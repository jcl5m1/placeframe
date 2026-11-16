from __future__ import annotations

from math import atan, degrees as degrees_scalar
from pathlib import Path
from typing import Dict, Optional

import torch
from numpy import arccos, clip, degrees, einsum, fill_diagonal, float64, stack, where
from numpy.linalg import norm
from numpy.typing import NDArray  # noqa: TID251 — Phase T piece 3 follow-up migration
from scipy.spatial.transform import Rotation
from torch import from_numpy, topk  # type: ignore

from .rig import Rig, Transform

PAIRS_FILE = "pairs.txt"

# A rig whose cameras share one optical centre and point more than this far apart covers a wide field
# around the frame (an Insta360 equirect split into pinhole views, a cubemap): which way the rig's
# reference camera faces says little about what the frame sees, so frame-level heading gating is wrong.
PANORAMIC_SPREAD_DEG = 90.0
SHARED_CENTRE_TOLERANCE_M = 0.05
# Two views are only worth matching if their fields of view overlap by at least this much.
MIN_FOV_OVERLAP_DEG = 20.0


def generate_image_pairs(rigs: Dict[str, Rig], neighbors_count: int, rotation_thresh_deg: float):
    camera_axes = {rig_id: _camera_axes_in_rig(rig) for rig_id, rig in rigs.items()}
    camera_fovs = {rig_id: _camera_fovs_deg(rig) for rig_id, rig in rigs.items()}
    panoramic = {rig_id: _is_panoramic(rig, camera_axes[rig_id]) for rig_id, rig in rigs.items()}

    frames = {
        (rig_id, frame_id): transform for rig_id, rig in rigs.items() for frame_id, transform in rig.frame_poses.items()
    }
    proximal_frame_pairs = pairs_from_poses(
        frames,
        neighbors_count,
        rotation_thresh_deg,
        camera_axes={key: camera_axes[key[0]] for key in frames},
        gate_frames={key: not panoramic[key[0]] for key in frames},
    )

    cross_frame_image_pairs_by_frame_proximity = [
        (f"{rig_id_a}/{camera_id_a}/{frame_id_a}.jpg", f"{rig_id_b}/{camera_id_b}/{frame_id_b}.jpg")
        for (rig_id_a, frame_id_a), (rig_id_b, frame_id_b) in proximal_frame_pairs
        for camera_id_a, camera_id_b in _overlapping_camera_pairs(
            frames[(rig_id_a, frame_id_a)],
            frames[(rig_id_b, frame_id_b)],
            camera_axes[rig_id_a],
            camera_axes[rig_id_b],
            camera_fovs[rig_id_a],
            camera_fovs[rig_id_b],
            panoramic[rig_id_a] or panoramic[rig_id_b],
        )
    ]

    intra_frame_image_pairs = [
        (f"{rig_id}/{camera_id_a}/{frame_id}.jpg", f"{rig_id}/{camera_id_b}/{frame_id}.jpg")
        for rig_id, rig in rigs.items()
        for frame_id in rig.frame_poses.keys()
        for camera_id_a, camera_id_b in _intra_frame_camera_pairs(camera_axes[rig_id], camera_fovs[rig_id])
    ]

    # Canonicalize and deduplicate
    return sorted({
        (a, b) if a <= b else (b, a)
        for a, b in cross_frame_image_pairs_by_frame_proximity + intra_frame_image_pairs
        if a != b
    })


def write_pairs(pairs: list[tuple[str, ...]], root_path: Path):
    path = root_path / PAIRS_FILE
    path.write_text("\n".join([" ".join(pair) for pair in pairs]))
    return PAIRS_FILE, path.read_bytes()


def pairs_from_poses(
    images: dict[tuple[str, str], Transform],
    num_neighbors: int,
    rotation_thresh_deg: float,
    camera_axes: Optional[dict[tuple[str, str], dict[str, NDArray[float64]]]] = None,
    gate_frames: Optional[dict[tuple[str, str], bool]] = None,
):
    names = list(images.keys())
    R_w_c = stack([images[name].rotation for name in names], axis=0)  # noqa: N806 — rotation matrix notation
    centers = stack([images[name].translation for name in names], axis=0)  # (N, 3)

    # Euclidean distances of camera centres
    diff = centers[:, None, :] - centers[None, :, :]
    dists = norm(diff, axis=-1)

    # Principal optical axis (Z direction) in world frame
    optical_axes = R_w_c[:, :, 2]
    cosines = clip(optical_axes @ optical_axes.T, -1.0, 1.0)
    ang_deg = degrees(arccos(cosines))

    scores = -from_numpy(dists)
    invalid = ang_deg >= rotation_thresh_deg
    if gate_frames is not None:
        # Panoramic frames see every heading: pair them by proximity alone.
        ungated = stack([not gate_frames[name] for name in names])
        invalid[ungated, :] = False
        invalid[:, ungated] = False
    fill_diagonal(invalid, True)
    invalid = from_numpy(invalid).to(scores.device)  # NEW: convert mask to torch bool tensor
    selected = pairs_from_score_matrix(scores, invalid, num_neighbors)

    return [(names[i], names[j]) for i, j in selected]


def pairs_from_retrieval(descriptors: Dict[str, torch.Tensor], num_neighbors: int, min_score: Optional[float] = None):
    names = list(descriptors.keys())
    tensor = torch.stack([descriptors[n] for n in names], dim=0)
    similarity = tensor @ tensor.t()

    invalid = torch.eye(len(names), dtype=torch.bool)
    if min_score is not None:
        invalid |= similarity < min_score

    selected = pairs_from_score_matrix(similarity, invalid, num_neighbors)
    return [(names[i], names[j]) for i, j in selected]


def pairs_from_score_matrix(scores: torch.Tensor, invalid: torch.Tensor, num_select: int):
    scores.masked_fill_(invalid, float("-inf"))
    top_k = topk(scores, min(num_select, scores.size(1)), dim=1)
    indices = top_k.indices.cpu().numpy()
    valid = top_k.values.isfinite().cpu().numpy()

    return [(int(i), int(indices[i, j])) for i, j in zip(*where(valid))]


def _camera_axes_in_rig(rig: Rig) -> dict[str, NDArray[float64]]:
    # cam_from_rig rotation's third row is the camera's optical axis expressed in the rig frame.
    return {
        camera_id: Rotation.from_quat([
            config.rotation.x,
            config.rotation.y,
            config.rotation.z,
            config.rotation.w,
        ]).as_matrix()[2]
        for camera_id, (config, _) in rig.cameras.items()
    }


def _camera_fovs_deg(rig: Rig) -> dict[str, float]:
    # The narrower of the horizontal and vertical fields of view (PINHOLE params: fx, fy, cx, cy).
    fovs = {}
    for camera_id, (_, colmap_camera) in rig.cameras.items():
        fx, fy = colmap_camera.params[0], colmap_camera.params[1]
        fovs[camera_id] = min(
            2 * degrees_scalar(atan(colmap_camera.width / (2 * fx))),
            2 * degrees_scalar(atan(colmap_camera.height / (2 * fy))),
        )
    return fovs


def _is_panoramic(rig: Rig, axes: dict[str, NDArray[float64]]) -> bool:
    if len(axes) < 3:
        return False
    translations = stack([
        [config.translation.x, config.translation.y, config.translation.z] for config, _ in rig.cameras.values()
    ])
    if norm(translations - translations.mean(axis=0), axis=1).max() > SHARED_CENTRE_TOLERANCE_M:
        return False
    axis_matrix = stack(list(axes.values()))
    spread = degrees(arccos(clip(einsum("id,jd->ij", axis_matrix, axis_matrix), -1.0, 1.0))).max()
    return bool(spread > PANORAMIC_SPREAD_DEG)


def _overlapping_camera_pairs(
    transform_a: Transform,
    transform_b: Transform,
    axes_a: dict[str, NDArray[float64]],
    axes_b: dict[str, NDArray[float64]],
    fovs_a: dict[str, float],
    fovs_b: dict[str, float],
    panoramic: bool,
):
    # Non-panoramic rigs keep every camera combination (the frame-level heading gate already applied).
    # Panoramic rigs only match views whose world-space fields of view overlap; the heading threshold
    # does not apply to them, since a panoramic frame sees every heading.
    if not panoramic:
        return [(camera_a, camera_b) for camera_a in axes_a for camera_b in axes_b]
    pairs = []
    for camera_a, axis_a in axes_a.items():
        world_axis_a = transform_a.rotation @ axis_a
        for camera_b, axis_b in axes_b.items():
            world_axis_b = transform_b.rotation @ axis_b
            angle = degrees_scalar(float(arccos(clip(world_axis_a @ world_axis_b, -1.0, 1.0))))
            if angle < _overlap_limit_deg(fovs_a[camera_a], fovs_b[camera_b]):
                pairs.append((camera_a, camera_b))
    return pairs


def _intra_frame_camera_pairs(axes: dict[str, NDArray[float64]], fovs: dict[str, float]):
    # Cameras of one frame are only worth matching where their views overlap (stereo pairs always do;
    # opposite-facing views of a panoramic rig never do).
    ids = list(axes)
    return [
        (ids[i], ids[j])
        for i in range(len(ids))
        for j in range(i + 1, len(ids))
        if degrees_scalar(float(arccos(clip(axes[ids[i]] @ axes[ids[j]], -1.0, 1.0))))
        < _overlap_limit_deg(fovs[ids[i]], fovs[ids[j]])
    ]


def _overlap_limit_deg(fov_a: float, fov_b: float) -> float:
    return (fov_a + fov_b) / 2 - MIN_FOV_OVERLAP_DEG
