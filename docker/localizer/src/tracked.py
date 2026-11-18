from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from os import environ
from threading import Lock
from typing import Any, cast

import numpy as np
from core.calibration import CalibrationArtifact
from core.camera_config import PinholeCameraConfig
from core.image_preprocess import (
    MaxTiles,
    NumImages,
    NumQueryTiles,
    canonicalize_image,
    canonicalize_intrinsics,
    tile_image,
)
from core.lightglue import Descriptors, Keypoints
from core.localization_metrics import RANSAC_THRESHOLD_DEFAULT
from core.model_wrappers import RetrievalDim
from core.tensor_types import TT
from core.opq import decode_descriptors
from numpy import asarray, float32, vstack
from pycolmap import AbsolutePoseEstimationOptions, RANSACOptions
from pycolmap import Camera as ColmapCamera
from pycolmap._core import Rigid3d, estimate_and_refine_absolute_pose, set_random_seed  # type: ignore  # noqa: PLC2701 — no public API
from torch import Tensor, cuda, inference_mode, manual_seed, topk  # type: ignore

from . import localize
from .build_metrics import build_localization_metrics
from .map import Map
from .torch_ops import amax, from_numpy, matmul, permute, stack, to, transpose
from .tracker import Candidate, Matrix4

# Candidate poses of one frame for the tracker: unlike the single-pose localizer, nothing is gated on confidence
# (the tracker weighs weak candidates by how well they fit the trajectory) and the retrieved images are split by
# place, so a frame that resembles two places yields a pose for each instead of one RANSAC winner.
RETRIEVAL_TOP_K = 16
PLACE_RADIUS_M = 4.0  # retrieved images within this distance of a group's first image form one place
MAX_PLACES = 4
CONFIDENT_RETRIEVAL_TOP_K = 6
# CPU work per request (see the pipeline repo's cost study): decoded map features stay on the GPU for recently
# matched images, and PyTorch's cached GPU memory is released only above a limit (it was released after every
# request so a reconstruction on the same GPU is not starved; the limit keeps that bound).
FEATURE_CACHE_IMAGES = int(environ.get("LOCALIZER_FEATURE_CACHE_IMAGES", "1000"))  # per map and process; 0: off
RELEASE_GPU_MEMORY_ABOVE_BYTES = int(float(environ.get("LOCALIZER_RELEASE_GPU_MEMORY_ABOVE_GB", "4")) * 2**30)
PRIOR_RADIUS_M = 4.0  # images this close to the predicted pose and facing a similar way are matched too
PRIOR_MAX_ANGLE_DEG = 75.0
PRIOR_TOP_K = 8
MIN_CORRESPONDENCES = 12
# Query images are resized to this shorter side (the single-frame localizer uses 1024). Phone frames are 480-1080 px:
# upscaling a 640x480 frame to 1024 only adds blur; on a recorded walk, 640 found the right pose 6% more often in
# a third less time.
QUERY_SHORTER_SIDE = 640
# LightGlue pairs per batch: GPU memory grows with batch × keypoints², so all 16-32 pairs at once needed 11-22 GB.
MATCH_BATCH_SIZE = 8


@dataclass
class _ImageGeometry:
    centers: np.ndarray  # (N, 3) camera centres in the map frame, ordered like map.ordered_image_ids
    directions: np.ndarray  # (N, 3) viewing directions


_geometry: dict[int, _ImageGeometry] = {}


def _turn(rows: list[list[float]]) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = rows
    return m


# The pose is estimated for the image turned upright (camera_config orientation); the client's odometry is of the
# sensor camera. Upright-camera-from-sensor-camera per orientation: a rotation about the optical axis.
_UPRIGHT_FROM_SENSOR = {
    "RIGHT_TOP": _turn([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    "LEFT_BOTTOM": _turn([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
    "BOTTOM_RIGHT": _turn([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
}


@inference_mode()
def localize_candidates(
    map: Map,
    camera: PinholeCameraConfig,
    image_buffer: bytes,
    predicted_map_from_camera: Matrix4 | None,
    confident: bool,
    pipeline_version: str,
    calibration: CalibrationArtifact,
) -> list[Candidate]:
    set_random_seed(localize.LOCALIZER_RANDOM_SEED)
    manual_seed(localize.LOCALIZER_RANDOM_SEED)
    image = canonicalize_image(image_buffer, camera.orientation, QUERY_SHORTER_SIDE)
    rgb = from_numpy(asarray(image, dtype=float32)).permute(2, 0, 1).div(255.0)
    query_keypoints, query_descriptors = localize.local_feature_extractor(rgb.unsqueeze(0).to(device=localize.DEVICE))
    tiles: list[TT[RetrievalDim]] = [
        localize.global_descriptor_extractor(
            from_numpy(asarray(t, dtype=float32)).permute(2, 0, 1).div(255.0).unsqueeze(0).to(device=localize.DEVICE)
        )
        for t in tile_image(image, QUERY_SHORTER_SIDE)
    ]
    database = _gpu_tile_descriptors(map)
    query_tiles: TT[NumQueryTiles, RetrievalDim] = stack(tiles, dim=0)
    pairs_similarity = permute(matmul(query_tiles, transpose(database, 1, 2)), (1, 0, 2))
    similarity = amax(pairs_similarity, dim=(0, 2)).cpu().numpy()
    # A confident track only needs to catch a different place appearing: fewer retrieved images (the images near the
    # predicted pose are matched anyway), so a confident query matches fewer pairs.
    top_k = CONFIDENT_RETRIEVAL_TOP_K if confident else RETRIEVAL_TOP_K
    retrieved = cast(list[int], topk(from_numpy(similarity), top_k).indices.tolist())  # type: ignore

    geometry = _image_geometry(map)
    groups = _places(retrieved, geometry)
    if predicted_map_from_camera is not None:
        near = _near_prediction(predicted_map_from_camera, geometry, similarity)
        if near:
            groups.insert(0, ("prior", near))
    indices = sorted({i for _, g in groups for i in g})
    image_ids = [map.ordered_image_ids[i] for i in indices]

    keypoints, descriptor_tensors = _map_features(map, image_ids)
    sizes = {str(i): map.image_sizes[str(i)] for i in image_ids}
    keypoints["query"] = query_keypoints.to(localize.DEVICE)
    descriptor_tensors["query"] = query_descriptors.to(localize.DEVICE)
    sizes["query"] = (image.height, image.width)
    pairs = [(str(i), "query") for i in image_ids]
    matches = localize.local_feature_matcher(pairs, keypoints, descriptor_tensors, sizes, MATCH_BATCH_SIZE)

    width, height, *params = canonicalize_intrinsics(camera, QUERY_SHORTER_SIDE)
    colmap_camera = ColmapCamera(width=width, height=height, model="PINHOLE", params=params)
    options = AbsolutePoseEstimationOptions()
    options.ransac = RANSACOptions()
    options.ransac.max_error = RANSAC_THRESHOLD_DEFAULT
    query_points = keypoints["query"].cpu().numpy()

    candidates: list[Candidate] = []
    for source, group in groups:
        query_indices: list[int] = []
        point_ids: list[int] = []
        num_matches = 0
        for image_id in (map.ordered_image_ids[i] for i in group):
            database_indices, query_match_indices = matches[(str(image_id), "query")]
            num_matches += int(database_indices.shape[0])
            for database_index, query_index in zip(database_indices, query_match_indices):
                point2d = map.images[image_id].points2D[int(database_index)]
                if point2d.has_point3D():
                    query_indices.append(int(query_index))
                    point_ids.append(int(point2d.point3D_id))
        if len(query_indices) < MIN_CORRESPONDENCES:
            continue
        points2d = query_points[query_indices]
        points3d = vstack([map.points3D[i].xyz for i in point_ids])
        result = cast(
            dict[str, Any] | None,
            estimate_and_refine_absolute_pose(points2d, points3d, colmap_camera, options, return_covariance=True),
        )
        if result is None:
            continue
        metrics = build_localization_metrics(
            result, points2d, points3d, colmap_camera, num_matches, width, height, pipeline_version, calibration, map
        )
        cam_from_world = cast(Rigid3d, result["cam_from_world"])
        camera_from_map = np.eye(4)
        camera_from_map[:3, :3] = cam_from_world.rotation.matrix()
        camera_from_map[:3, 3] = cam_from_world.translation
        map_from_camera = np.linalg.inv(camera_from_map) @ _UPRIGHT_FROM_SENSOR.get(camera.orientation, np.eye(4))
        candidates.append(
            Candidate(cast(Matrix4, map_from_camera), float(metrics.confidence_loose), metrics.num_inliers, source)
        )
    release_gpu_memory()
    return candidates


# The map's retrieval descriptors stay on the GPU (one copy per loaded map) instead of being uploaded per query.
def _gpu_tile_descriptors(map: Map) -> TT[NumImages, MaxTiles, RetrievalDim]:
    key = id(map)
    if key not in _gpu_descriptors:
        _gpu_descriptors[key] = to(from_numpy(map.tile_descriptors), localize.DEVICE)
    return _gpu_descriptors[key]


_gpu_descriptors: dict[int, TT[NumImages, MaxTiles, RetrievalDim]] = {}


def _image_geometry(map: Map) -> _ImageGeometry:
    key = id(map)
    if key not in _geometry:
        images = [map.images[i] for i in map.ordered_image_ids]
        _geometry[key] = _ImageGeometry(
            np.array([im.projection_center() for im in images]), np.array([im.viewing_direction() for im in images])
        )
    return _geometry[key]


# Retrieved images grouped by place, most similar first: each group starts at the most similar ungrouped image and
# takes the retrieved images within PLACE_RADIUS_M of it.
def _places(retrieved: list[int], geometry: _ImageGeometry) -> list[tuple[str, list[int]]]:
    groups: list[tuple[str, list[int]]] = []
    left: list[int] = list(retrieved)
    while left and len(groups) < MAX_PLACES:
        seed = geometry.centers[left[0]]
        group = [i for i in left if np.linalg.norm(geometry.centers[i] - seed) <= PLACE_RADIUS_M]
        groups.append(("retrieval", group))
        left = [i for i in left if i not in group]
    return groups


def _near_prediction(predicted: Matrix4, geometry: _ImageGeometry, similarity: np.ndarray) -> list[int]:
    center, direction = predicted[:3, 3], predicted[:3, 2]  # OpenCV camera: +z looks forward
    close = np.linalg.norm(geometry.centers - center, axis=1) <= PRIOR_RADIUS_M
    facing = geometry.directions @ direction >= np.cos(np.radians(PRIOR_MAX_ANGLE_DEG))
    nearby = np.flatnonzero(close & facing)
    return [int(i) for i in nearby[np.argsort(-similarity[nearby])][:PRIOR_TOP_K]]


# PyTorch keeps freed GPU memory cached; a long localization session otherwise holds 15-20 GB, which a
# reconstruction running on the same GPU then can't get.
def release_gpu_memory() -> None:
    if cuda.is_available() and cuda.memory_reserved() > RELEASE_GPU_MEMORY_ABOVE_BYTES:
        cuda.empty_cache()


_feature_cache: dict[int, OrderedDict[int, tuple[Tensor, Tensor]]] = {}
_feature_lock = Lock()


def _map_features(map: Map, image_ids: list[int]) -> tuple[Keypoints, Descriptors]:
    """Keypoints and decoded descriptors of map images, on the GPU; recently used images come from a cache."""
    with _feature_lock:
        cache = _feature_cache.setdefault(id(map), OrderedDict())
        found = {i: cache[i] for i in image_ids if i in cache}
        for i in found:
            cache.move_to_end(i)
    missing = [i for i in image_ids if i not in found]
    if missing:
        decoded = decode_descriptors(map.opq_matrix, map.product_quantizer, {i: map.pq_codes[i] for i in missing})
        new = {
            i: (from_numpy(map.keypoints[i]).to(localize.DEVICE), from_numpy(decoded[i]).to(localize.DEVICE))
            for i in missing
        }
        found |= new
        if FEATURE_CACHE_IMAGES:
            with _feature_lock:
                cache.update(new)
                while len(cache) > FEATURE_CACHE_IMAGES:
                    cache.popitem(last=False)
    return (Keypoints({str(i): found[i][0] for i in image_ids}), Descriptors({str(i): found[i][1] for i in image_ids}))
