import json
import os
import time
import zlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, cast
from uuid import UUID

from common.multipart_requests import (
    MultipartRequestModel,
    MultipartRequestOperation,
    multipart_json,
    multipart_json_list,
)
from core.axis_convention import AxisConvention
from core.camera_config import PinholeCameraConfig
from core.localization_metrics import LocalizationMetrics
from core.transform import Float3, Float4, Transform
from datamodels.public_tables import Reconstruction
import httpx
from litestar import Request, Router, get, post
from litestar.datastructures import UploadFile
from litestar.di import Provide
from litestar.enums import RequestEncodingType
from litestar.exceptions import HTTPException
from litestar.params import Body
from litestar.status_codes import HTTP_404_NOT_FOUND, HTTP_422_UNPROCESSABLE_ENTITY, HTTP_502_BAD_GATEWAY
from placeframe_localizer_client import ApiClient, ApiException, Configuration
from placeframe_localizer_client.api.default_api import DefaultApi
from placeframe_localizer_client.models.axis_convention import AxisConvention as LocalizerAxisConvention
from placeframe_localizer_client.models.pinhole_camera_config import PinholeCameraConfig as LocalizerPinholeCameraConfig
from placeframe_localizer_client.models.reconstruction_metrics import (
    ReconstructionMetrics as LocalizerReconstructionMetrics,
)
from pydantic import BaseModel, BeforeValidator, Json
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_session
from ..settings import get_settings
from .localization_maps import fetch_localization_maps

settings = get_settings()
# Local addition: when set, every /localize request (image + parameters + response + latency) is written here
# so real client sessions can be replayed for localization metrics, regression tests and load simulation.
RECORD_DIR = os.environ.get("LOCALIZATION_RECORD_DIR")


class LocalizationRequest(MultipartRequestModel):
    map_ids: Annotated[Json[list[UUID]], BeforeValidator(multipart_json_list)]
    camera_config: Annotated[Json[PinholeCameraConfig], BeforeValidator(multipart_json)]
    axis_convention: AxisConvention
    retrieval_top_k: int | None = None
    ransac_threshold: float | None = None
    image: UploadFile


class MapLocalization(BaseModel):
    id: UUID
    camera_from_map_transform: Transform
    map_transform: Transform
    metrics: LocalizationMetrics


@post("/", operation_class=MultipartRequestOperation)
async def localize_image(
    request: Request[Any, Any, Any],
    session: AsyncSession,
    data: Annotated[LocalizationRequest, Body(media_type=RequestEncodingType.MULTI_PART)],
) -> list[MapLocalization]:
    started = time.monotonic()
    received_at = datetime.now(UTC)
    image = await data.image.read()
    try:
        result = await _localize(session, data, image)
    except HTTPException as e:
        _record(request, data, image, received_at, started, None, e.status_code)
        raise
    _record(request, data, image, received_at, started, result, 201)
    return result


async def _localize(session: AsyncSession, data: LocalizationRequest, image: bytes) -> list[MapLocalization]:
    reconstruction_id_to_map_id = {
        map.reconstruction_id: map for map in await fetch_localization_maps(session, data.map_ids)
    }
    manifest_rows = (
        await session.execute(
            select(Reconstruction.id, Reconstruction.manifest).where(
                Reconstruction.id.in_(reconstruction_id_to_map_id.keys())
            )
        )
    ).all()

    async with ApiClient(Configuration(host=str(settings.localizer_container_url))) as api_client:
        try:
            localizations = await DefaultApi(api_client).localize_image(
                reconstruction_ids=list(reconstruction_id_to_map_id.keys()),
                metrics={
                    str(row_id): LocalizerReconstructionMetrics.model_validate(manifest["metrics"])
                    for row_id, manifest in manifest_rows
                },
                camera_config=LocalizerPinholeCameraConfig.model_validate(data.camera_config.model_dump()),
                axis_convention=LocalizerAxisConvention(data.axis_convention.value),
                image=image,
                retrieval_top_k=data.retrieval_top_k,
                ransac_threshold=data.ransac_threshold,
            )

            return [
                MapLocalization(
                    id=reconstruction_id_to_map_id[localization.id].id,
                    camera_from_map_transform=Transform.model_validate(localization.transform.model_dump()),
                    map_transform=Transform(
                        translation=Float3(
                            x=reconstruction_id_to_map_id[localization.id].position_x,
                            y=reconstruction_id_to_map_id[localization.id].position_y,
                            z=reconstruction_id_to_map_id[localization.id].position_z,
                        ),
                        rotation=Float4(
                            x=reconstruction_id_to_map_id[localization.id].rotation_x,
                            y=reconstruction_id_to_map_id[localization.id].rotation_y,
                            z=reconstruction_id_to_map_id[localization.id].rotation_z,
                            w=reconstruction_id_to_map_id[localization.id].rotation_w,
                        ),
                    ),
                    metrics=LocalizationMetrics.model_validate(localization.metrics.model_dump()),
                )
                for localization in localizations
            ]

        except ApiException as e:
            status = cast(int | None, e.status)
            if status == 422:
                # The localizer's reason (no matches, confidence below the gate, ...) tells clients why a frame failed.
                raise HTTPException(status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail=_localizer_detail(e)) from e
            raise HTTPException(status_code=HTTP_502_BAD_GATEWAY, detail="Localization session backend error") from e


class TrackedLocalizationRequest(MultipartRequestModel):
    map_ids: Annotated[Json[list[UUID]], BeforeValidator(multipart_json_list)]
    camera_config: Annotated[Json[PinholeCameraConfig], BeforeValidator(multipart_json)]
    session_id: str
    odometry_from_camera: Annotated[Json[Transform], BeforeValidator(multipart_json)]
    image: UploadFile


class TrackedMapLocalization(BaseModel):
    map_id: UUID | None
    probability: float
    sigma_m: float | None
    fixes: int
    hypotheses: int
    unknown: float
    camera_from_map_transform: Transform | None
    odometry_from_map_transform: Transform | None
    map_transform: Transform | None
    frame_candidates: int
    frame_agreed: bool
    frame_confidence: float | None
    frame_inliers: int | None
    frame_source: str | None


# Localization of a moving client (OpenCV axes): each frame carries its pose in the client's odometry frame
# (ARCore/ARKit world) and a per-session id. The localizer tracks every place the frames could be from and returns
# the most probable one, with its probability; weak matches that fit the trajectory count, stray ones are ignored.
@post("/track", operation_class=MultipartRequestOperation)
async def localize_tracked(
    request: Request[Any, Any, Any],
    session: AsyncSession,
    data: Annotated[TrackedLocalizationRequest, Body(media_type=RequestEncodingType.MULTI_PART)],
) -> TrackedMapLocalization:
    started = time.monotonic()
    received_at = datetime.now(UTC)
    image = await data.image.read()
    maps = {m.reconstruction_id: m for m in await fetch_localization_maps(session, data.map_ids)}
    if not maps:  # unknown or inactive maps: nothing to localize against
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail=f"No active localization map with id {', '.join(str(i) for i in data.map_ids) or '(none given)'}",
        )
    manifest_rows = (
        await session.execute(
            select(Reconstruction.id, Reconstruction.manifest).where(Reconstruction.id.in_(maps.keys()))
        )
    ).all()
    async with httpx.AsyncClient(base_url=_tracking_localizer(data.session_id), timeout=60) as client:
        response = await client.post(
            "/localization/track",
            data={
                "reconstruction_ids": json.dumps([str(r) for r in maps]),
                "metrics": json.dumps({str(row_id): manifest["metrics"] for row_id, manifest in manifest_rows}),
                "camera_config": data.camera_config.model_dump_json(),
                "session_id": data.session_id,
                "odometry_from_camera": data.odometry_from_camera.model_dump_json(),
            },
            files={"image": ("frame.jpg", image, "image/jpeg")},
        )
    if response.status_code >= 400:
        _record_tracked(request, data, image, received_at, started, None, response.status_code)
        raise HTTPException(status_code=HTTP_502_BAD_GATEWAY, detail=f"Localizer: {response.text[:300]}")
    tracked = response.json()
    map_row = maps.get(UUID(tracked["id"])) if tracked["id"] else None
    result = TrackedMapLocalization(
        map_id=map_row.id if map_row else None,
        probability=tracked["probability"],
        sigma_m=tracked["sigma_m"],
        fixes=tracked["fixes"],
        hypotheses=tracked["hypotheses"],
        unknown=tracked["unknown"],
        camera_from_map_transform=tracked["camera_from_map"],
        odometry_from_map_transform=tracked["odometry_from_map"],
        map_transform=Transform(
            translation=Float3(x=map_row.position_x, y=map_row.position_y, z=map_row.position_z),
            rotation=Float4(x=map_row.rotation_x, y=map_row.rotation_y, z=map_row.rotation_z, w=map_row.rotation_w),
        )
        if map_row
        else None,
        frame_candidates=tracked["frame_candidates"],
        frame_agreed=tracked["frame_agreed"],
        frame_confidence=tracked["frame_confidence"],
        frame_inliers=tracked["frame_inliers"],
        frame_source=tracked["frame_source"],
    )
    _record_tracked(request, data, image, received_at, started, result, 201)
    return result


# Localizer processes for tracked sessions. A session's tracker state lives in one process, so each session always goes
# to the same one (by a hash of its id); several processes use a GPU more fully than one.
TRACK_URLS = [u.strip() for u in os.environ.get("LOCALIZER_TRACK_URLS", "").split(",") if u.strip()]


def _tracking_localizer(session_id: str) -> str:
    if not TRACK_URLS:
        return str(settings.localizer_container_url)
    return TRACK_URLS[zlib.crc32(session_id.encode()) % len(TRACK_URLS)]


def _record_tracked(
    request: Request[Any, Any, Any],
    data: TrackedLocalizationRequest,
    image: bytes,
    received_at: datetime,
    started: float,
    result: TrackedMapLocalization | None,
    status: int,
) -> None:
    if not RECORD_DIR or request.headers.get("x-placeframe-replay"):
        return
    day = Path(RECORD_DIR) / received_at.strftime("%Y-%m-%d")
    stem = received_at.strftime("%H%M%S_%f")
    entry = {
        "received_at": received_at.isoformat(),
        "endpoint": "track",
        "latency_ms": round((time.monotonic() - started) * 1000, 1),
        "status": status,
        "user_agent": request.headers.get("user-agent"),
        "image": f"{stem}.jpg",
        "image_bytes": len(image),
        "request": {
            "map_ids": [str(m) for m in data.map_ids],
            "camera_config": data.camera_config.model_dump(mode="json"),
            "axis_convention": "OPENCV",
            "session_id": data.session_id,
            "odometry_from_camera": data.odometry_from_camera.model_dump(mode="json"),
        },
        "response": result.model_dump(mode="json") if result is not None else None,
    }
    try:
        day.mkdir(parents=True, exist_ok=True)
        (day / f"{stem}.jpg").write_bytes(image)
        with (day / "requests.jsonl").open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"[localization-recorder] failed to record request: {e!r}")


def _record(
    request: Request[Any, Any, Any],
    data: LocalizationRequest,
    image: bytes,
    received_at: datetime,
    started: float,
    result: list[MapLocalization] | None,
    status: int,
) -> None:
    if not RECORD_DIR or request.headers.get("x-placeframe-replay"):
        return
    day = Path(RECORD_DIR) / received_at.strftime("%Y-%m-%d")
    stem = received_at.strftime("%H%M%S_%f")
    entry = {
        "received_at": received_at.isoformat(),
        "latency_ms": round((time.monotonic() - started) * 1000, 1),
        "status": status,
        "client": request.headers.get("x-forwarded-for") or (request.client.host if request.client else None),
        "user_agent": request.headers.get("user-agent"),
        "image": f"{stem}.jpg",
        "image_bytes": len(image),
        "image_content_type": data.image.content_type,
        "request": {
            "map_ids": [str(m) for m in data.map_ids],
            "camera_config": data.camera_config.model_dump(mode="json"),
            "axis_convention": data.axis_convention.value,
            "retrieval_top_k": data.retrieval_top_k,
            "ransac_threshold": data.ransac_threshold,
        },
        "response": [r.model_dump(mode="json") for r in result] if result is not None else None,
    }
    try:
        day.mkdir(parents=True, exist_ok=True)
        (day / f"{stem}.jpg").write_bytes(image)
        with (day / "requests.jsonl").open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"[localization-recorder] failed to record request: {e!r}")


@get("/version")
async def get_localizer_version() -> str:
    async with ApiClient(Configuration(host=str(settings.localizer_container_url))) as api_client:
        return await DefaultApi(api_client).get_localizer_version()


router = Router(
    "/localize",
    tags=["Localization"],
    dependencies={"session": Provide(get_session)},
    route_handlers=[localize_image, localize_tracked, get_localizer_version],
)


def _localizer_detail(e: ApiException) -> str:
    try:
        detail = json.loads(cast(str, e.body)).get("detail")
    except (TypeError, ValueError, AttributeError):
        return "Not localized"
    return detail if isinstance(detail, str) else "Not localized"
