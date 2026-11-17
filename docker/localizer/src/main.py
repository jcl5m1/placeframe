from __future__ import annotations

from asyncio import get_running_loop
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from os import environ
from pathlib import Path
from threading import Lock
from typing import Annotated
from uuid import UUID

from common.boto_clients import create_s3_client
from common.litestar import create_litestar_app
from common.multipart_requests import (
    MultipartRequestModel,
    MultipartRequestOperation,
    multipart_json,
    multipart_json_list,
)
from core.axis_convention import AxisConvention
from core.camera_config import PinholeCameraConfig
from core.reconstruction_metrics import ReconstructionMetrics
import numpy as np
from litestar import get, post
from litestar.datastructures import UploadFile
from litestar.enums import RequestEncodingType
from litestar.exceptions import HTTPException
from litestar.openapi.config import OpenAPIConfig
from litestar.openapi.spec import Server
from litestar.params import Body
from litestar.status_codes import HTTP_422_UNPROCESSABLE_ENTITY
from core.transform import Float3, Float4, Transform
from pydantic import BeforeValidator, Json
from scipy.spatial.transform import Rotation

from core.calibration import CalibrationArtifact
from .map import Map, load_map
from .schemas import LoadState, Localization, TrackedLocalization
from .settings import get_settings
from .tracker import FrameResult, Sessions, Tracker, pose_from_quaternion, sigma

RECONSTRUCTIONS_DIR = Path("/tmp/reconstructions")
CALIBRATION_GLOBAL_PATH = Path("/etc/placeframe/calibration/global.json")


_executor = ThreadPoolExecutor(max_workers=2)
_track_executor = ThreadPoolExecutor(max_workers=int(environ.get("LOCALIZER_TRACK_WORKERS", "2")))
_load_lock = Lock()
_load_state: dict[UUID, LoadState] = {}
_load_error: dict[UUID, str] = {}
_maps: dict[UUID, Map] = {}
_sessions = Sessions()

settings = get_settings()
s3_client = create_s3_client(
    minio_endpoint_url=settings.minio_endpoint_url,
    minio_access_key=settings.minio_access_key,
    minio_secret_key=settings.minio_secret_key,
)

pipeline_version: str = ""
calibration: CalibrationArtifact | None = None

if not environ.get("CODEGEN"):
    from core.calibration import load_global_calibration
    from .localize import load_models

    load_models()
    pipeline_version = environ["LOCALIZER_SHA"]
    calibration = load_global_calibration(CALIBRATION_GLOBAL_PATH, pipeline_version)
    print(f"Loaded global calibration (pipeline_version={pipeline_version[:12]}…)")


class LocalizationRequest(MultipartRequestModel):
    reconstruction_ids: Annotated[Json[list[UUID]], BeforeValidator(multipart_json_list)]
    metrics: Annotated[Json[dict[UUID, ReconstructionMetrics]], BeforeValidator(multipart_json)]
    camera_config: Annotated[Json[PinholeCameraConfig], BeforeValidator(multipart_json)]
    axis_convention: AxisConvention
    retrieval_top_k: int | None = None
    ransac_threshold: float | None = None
    image: UploadFile


@post("/localization", operation_class=MultipartRequestOperation)
async def localize_image(
    data: Annotated[LocalizationRequest, Body(media_type=RequestEncodingType.MULTI_PART)],
) -> list[Localization]:
    if environ.get("CODEGEN"):
        raise

    # Import here to avoid importing torch during codegen
    from .localize import LocalizationError, localize_image_against_reconstruction
    from .tracked import release_gpu_memory

    image = await data.image.read()

    localizations: list[Localization] = []
    errors: list[str] = []

    for id in data.reconstruction_ids:
        if id not in _maps:
            _maps[id] = load_map(id, s3_client, settings.reconstructions_bucket, RECONSTRUCTIONS_DIR, data.metrics[id])

        try:
            assert calibration is not None  # CODEGEN-guarded; runtime always has it
            result = localize_image_against_reconstruction(
                _maps[id],
                data.camera_config,
                data.axis_convention,
                image,
                data.retrieval_top_k,
                data.ransac_threshold,
                pipeline_version,
                calibration,
            )

            localizations.append(Localization(id=id, transform=result[0], metrics=result[1]))
        except LocalizationError as e:
            errors.append(f"Reconstruction {id}: {str(e)}")

    # Release PyTorch's cached GPU memory between requests: otherwise the localizer keeps 15-20 GB reserved, which a
    # reconstruction on the same GPU can't get.
    release_gpu_memory()
    if not localizations:
        raise HTTPException(status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail="; ".join(errors))

    return localizations


class TrackedLocalizationRequest(MultipartRequestModel):
    reconstruction_ids: Annotated[Json[list[UUID]], BeforeValidator(multipart_json_list)]
    metrics: Annotated[Json[dict[UUID, ReconstructionMetrics]], BeforeValidator(multipart_json)]
    camera_config: Annotated[Json[PinholeCameraConfig], BeforeValidator(multipart_json)]
    session_id: str
    odometry_from_camera: Annotated[Json[Transform], BeforeValidator(multipart_json)]
    image: UploadFile


# Localization of a moving client: each frame comes with its odometry pose, and a per-session multi-hypothesis
# tracker keeps every place the frames could be from, returning the most probable one (OpenCV axes only).
@post("/localization/track", operation_class=MultipartRequestOperation)
async def localize_tracked(
    data: Annotated[TrackedLocalizationRequest, Body(media_type=RequestEncodingType.MULTI_PART)],
) -> TrackedLocalization:
    if environ.get("CODEGEN"):
        raise

    from .tracked import localize_candidates

    image = await data.image.read()
    t, q = data.odometry_from_camera.translation, data.odometry_from_camera.rotation
    odom_from_camera = pose_from_quaternion([t.x, t.y, t.z], [q.x, q.y, q.z, q.w])
    results: list[tuple[UUID, Tracker, FrameResult, int]] = []
    for id in data.reconstruction_ids:
        if id not in _maps:
            _maps[id] = load_map(id, s3_client, settings.reconstructions_bucket, RECONSTRUCTIONS_DIR, data.metrics[id])
        tracker = _sessions.get(data.session_id, str(id))
        assert calibration is not None  # CODEGEN-guarded; runtime always has it
        # On a worker thread: one request's CPU work (decoding, pose estimation) overlaps another's GPU work, and
        # the server keeps answering while a frame is processed.
        candidates = await get_running_loop().run_in_executor(
            _track_executor,
            partial(
                localize_candidates,
                _maps[id],
                data.camera_config,
                image,
                tracker.predicted_map_from_camera(odom_from_camera),
                tracker.confident,
                pipeline_version,
                calibration,
            ),
        )
        results.append((id, tracker, tracker.update(odom_from_camera, candidates), len(candidates)))
        print(
            f"track session={data.session_id[:8]} map={str(id)[:8]} candidates="
            + ",".join(f"{c.source}:{c.confidence:.2f}/{c.num_inliers}" for c in candidates)
            + f" best={tracker.best.probability if tracker.best else 0:.3f} hyps={len(tracker.hypotheses)}"
        )
    id, tracker, frame, num_candidates = max(results, key=lambda r: r[1].best.probability if r[1].best else 0.0)
    best = tracker.best
    frame_info = {
        "frame_candidates": num_candidates,
        "frame_agreed": frame.agreed,
        "frame_confidence": frame.selected.confidence if frame.selected else None,
        "frame_inliers": frame.selected.num_inliers if frame.selected else None,
        "frame_source": frame.selected.source if frame.selected else None,
    }
    if best is None:
        return TrackedLocalization(
            id=None,
            probability=0.0,
            sigma_m=None,
            fixes=0,
            hypotheses=0,
            unknown=tracker.unknown,
            camera_from_map=None,
            odometry_from_map=None,
            **frame_info,
        )
    map_from_camera = np.linalg.inv(best.odom_from_map) @ odom_from_camera
    return TrackedLocalization(
        id=id,
        probability=best.probability,
        sigma_m=sigma(best),
        fixes=best.fixes,
        hypotheses=len(tracker.hypotheses),
        unknown=tracker.unknown,
        camera_from_map=_transform(np.linalg.inv(map_from_camera)),
        odometry_from_map=_transform(best.odom_from_map),
        **frame_info,
    )


@get("/version")
async def get_localizer_version() -> str:
    return pipeline_version


openapi_config = OpenAPIConfig("Localizer", "0.1.0", servers=[Server(url="http://localhost:8000")])


app = create_litestar_app([localize_image, localize_tracked, get_localizer_version], openapi_config)


def _transform(m: np.ndarray) -> Transform:
    q = Rotation.from_matrix(m[:3, :3]).as_quat()
    return Transform(
        translation=Float3(x=float(m[0, 3]), y=float(m[1, 3]), z=float(m[2, 3])),
        rotation=Float4(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])),
    )
