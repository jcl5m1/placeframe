from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from numpy import eye, float64, ndarray
from pycolmap import (
    BundleAdjustmentConfig,
    BundleAdjustmentOptions,
    Database,
    IncrementalPipelineOptions,
    ObservationManager,
    PosePrior,
    PosePriorBundleAdjustmentOptions,
    PosePriorCoordinateSystem,
    Reconstruction,
    Rigid3d,
    Rotation3d,
    create_pose_prior_bundle_adjuster,
    incremental_mapping,
    logging,
    triangulate_points,
)

# Frames whose prior pose the images do not support (odometry glitch) keep fewer observations than this
# after outlier filtering; they are re-anchored on their nearest solved neighbour and retried once.
MIN_FRAME_OBSERVATIONS = 60
FILTER_MIN_TRIANGULATION_ANGLE_DEG = 1.5
BUNDLE_ADJUSTMENT_MAX_ITERATIONS = 50
GPU_SPARSE_SOLVER_MAX_IMAGES = 20000
# (max reprojection error px, max triangulation angle error deg) per round. Capture-time poses drift by
# metres and degrees, so the first round accepts loose tracks that let bundle adjustment pull the poses
# together; later rounds re-triangulate from the corrected poses with the mapper's usual strict limits.
ROUND_TOLERANCES = ((24.0, 8.0), (8.0, 4.0), (4.0, 2.0))


def prior_pose_mapping(
    database_path: Path,
    image_path: Path,
    output_path: Path,
    world_from_frame: dict[str, tuple[ndarray, ndarray]],
    pipeline_options: IncrementalPipelineOptions,
    prior_position_sigma_m: float,
    incremental_completion: bool = False,
    reanchor_dropped: bool = False,
    log: Callable[[str], None] = print,
    progress: Callable[[float], None] | None = None,
) -> dict[int, Reconstruction]:
    # Mapping from known poses instead of growing the map one frame at a time: every frame starts at its
    # capture-time pose (VIO/AR tracking), points are triangulated from the verified matches, then a few
    # bundle adjustments with pose-prior constraints remove drift. Frames the images contradict are
    # re-anchored on their nearest solved neighbour and retried; incremental_completion additionally hands
    # what is still unplaced to the incremental mapper (slow: one more whole-map adjustment).
    # world_from_frame keys are "<rig name>/<frame name>", values (rotation 3x3, translation 3) in the
    # database's world frame.
    t0 = perf_counter()
    reconstruction, pose_priors, frame_keys = _reconstruction_from_priors(
        database_path, world_from_frame, prior_position_sigma_m
    )
    log(f"[prior-mapper] {reconstruction.num_reg_frames()} frames posed from priors ({perf_counter() - t0:.0f}s)")

    work = output_path / "prior"
    work.mkdir(parents=True, exist_ok=True)
    strict = _tolerances(pipeline_options)
    for round_index, (max_reprojection_px, max_angle_deg) in enumerate(ROUND_TOLERANCES, start=1):
        last = round_index == len(ROUND_TOLERANCES)
        _set_tolerances(pipeline_options, strict if last else (max_reprojection_px, max_angle_deg))
        t = perf_counter()
        reconstruction = _triangulate(reconstruction, database_path, image_path, work, True, pipeline_options)
        log(
            f"[prior-mapper] round {round_index}: triangulated {reconstruction.num_points3D()} points at "
            f"{max_reprojection_px:.0f}px/{max_angle_deg:.0f}deg ({perf_counter() - t:.0f}s)"
        )
        t = perf_counter()
        solve = _pose_prior_bundle_adjustment(reconstruction, pose_priors, pipeline_options)
        removed = _filter(reconstruction, max_reprojection_px)
        dropped = _drop_unsupported_frames(reconstruction) if last else 0
        if progress is not None:
            progress(round_index / (len(ROUND_TOLERANCES) + 2))
        log(
            f"[prior-mapper] round {round_index}: bundle adjusted ({solve}), filtered {removed} observations, "
            f"dropped {dropped} unsupported frames ({perf_counter() - t:.0f}s)"
        )

    # Off by default: on the X5 merge it took 5 minutes to keep 1 of 50 dropped frames, which had no
    # usable matches at all rather than a wrong prior.
    reanchored = _reanchor_dropped_frames(reconstruction, frame_keys, world_from_frame) if reanchor_dropped else 0
    if reanchored:
        t = perf_counter()
        reconstruction = _triangulate(reconstruction, database_path, image_path, work, False, pipeline_options)
        _pose_prior_bundle_adjustment(reconstruction, pose_priors, pipeline_options)
        _filter(reconstruction, strict[0])
        dropped = _drop_unsupported_frames(reconstruction)
        log(
            f"[prior-mapper] re-anchored {reanchored} dropped frames on solved neighbours: "
            f"{reanchored - dropped} kept ({perf_counter() - t:.0f}s)"
        )
    t = perf_counter()
    # Final pass on the solved poses: extend and merge tracks across all views (each point observed by more
    # images gives the localizer more, stronger matches), as the incremental mapper does continuously.
    reconstruction = _triangulate(reconstruction, database_path, image_path, work, False, pipeline_options)
    log(
        f"[prior-mapper] tracks completed: {reconstruction.num_points3D()} points, mean track length "
        f"{reconstruction.compute_mean_track_length():.2f} ({perf_counter() - t:.0f}s)"
    )
    if progress is not None:
        progress((len(ROUND_TOLERANCES) + 1) / (len(ROUND_TOLERANCES) + 2))
    _set_tolerances(pipeline_options, strict)

    unregistered = reconstruction.num_frames() - reconstruction.num_reg_frames()
    if incremental_completion and unregistered:
        t = perf_counter()
        reconstruction.write(str(work))
        # The prior-supported frames are already solved: keep them fixed so completion only places the
        # dropped frames and their points, instead of re-solving the whole map in one large adjustment.
        pipeline_options.fix_existing_frames = True
        pipeline_options.multiple_models = False
        completed = incremental_mapping(
            database_path=str(database_path),
            image_path=str(image_path),
            output_path=str(output_path / "completed"),
            options=pipeline_options,
            input_path=str(work),
        )
        if completed:
            reconstruction = max(completed.values(), key=lambda r: r.num_reg_images())
        log(
            f"[prior-mapper] incremental completion: {unregistered} frames retried, now "
            f"{reconstruction.num_reg_frames()} registered ({perf_counter() - t:.0f}s)"
        )

    total_frames = reconstruction.num_frames()
    # Frames still unplaced keep no pose; saving and reloading leaves only the placed frames, which is what
    # the export code downstream expects (every frame and image in the result has a pose).
    final = output_path / "final"
    final.mkdir(parents=True, exist_ok=True)
    reconstruction.write(str(final))
    reconstruction = Reconstruction(str(final))
    log(
        f"[prior-mapper] done: {reconstruction.num_reg_frames()}/{total_frames} frames, "
        f"{reconstruction.num_points3D()} points in {perf_counter() - t0:.0f}s"
    )
    return {0: reconstruction}


def _reconstruction_from_priors(
    database_path: Path,
    world_from_frame: dict[str, tuple[ndarray, ndarray]],
    prior_position_sigma_m: float,
) -> tuple[Reconstruction, dict[int, PosePrior], dict[int, str]]:
    database = Database.open(str(database_path))
    reconstruction = Reconstruction()
    for camera in database.read_all_cameras():
        reconstruction.add_camera(camera)
    for rig in database.read_all_rigs():
        reconstruction.add_rig(rig)
    images = {image.image_id: image for image in database.read_all_images()}
    frames = database.read_all_frames()
    database.close()

    covariance = (prior_position_sigma_m**2) * eye(3, dtype=float64)
    pose_priors: dict[int, PosePrior] = {}
    frame_keys: dict[int, str] = {}
    posed: list[int] = []
    for frame in frames:
        frame_images = [images[data_id.id] for data_id in frame.data_ids if data_id.id in images]
        key = _frame_key(frame_images)
        prior = world_from_frame.get(key) if key else None
        if key:
            frame_keys[frame.frame_id] = key
        if prior is not None:
            rotation, translation = prior
            world_from_rig = Rigid3d(Rotation3d(rotation), translation)
            frame.rig_from_world = world_from_rig.inverse()
            posed.append(frame.frame_id)
        reconstruction.add_frame(frame)
        for image in frame_images:
            image.frame_id = frame.frame_id
            reconstruction.add_image(image)
            if prior is not None and cast(Any, image.camera_id) == _ref_camera_id(reconstruction, frame):
                pose_priors[image.image_id] = PosePrior(
                    position=prior[1].reshape(3, 1),
                    position_covariance=covariance,
                    coordinate_system=PosePriorCoordinateSystem.CARTESIAN,
                )
    for frame_id in posed:
        reconstruction.register_frame(frame_id)
    return reconstruction, pose_priors, frame_keys


def _tolerances(pipeline_options: IncrementalPipelineOptions) -> tuple[float, float]:
    return (
        float(pipeline_options.mapper.filter_max_reproj_error),
        float(pipeline_options.triangulation.create_max_angle_error),
    )


def _set_tolerances(pipeline_options: IncrementalPipelineOptions, tolerances: tuple[float, float]) -> None:
    max_reprojection_px, max_angle_deg = tolerances
    pipeline_options.mapper.filter_max_reproj_error = max_reprojection_px
    pipeline_options.triangulation.complete_max_reproj_error = max_reprojection_px
    pipeline_options.triangulation.merge_max_reproj_error = max_reprojection_px
    pipeline_options.triangulation.create_max_angle_error = max_angle_deg
    pipeline_options.triangulation.continue_max_angle_error = max_angle_deg


def _triangulate(
    reconstruction: Reconstruction,
    database_path: Path,
    image_path: Path,
    work: Path,
    clear_points: bool,
    pipeline_options: IncrementalPipelineOptions,
) -> Reconstruction:
    # COLMAP logs one INFO line per image while triangulating (hundreds of thousands per map): keep warnings.
    level = logging.minloglevel
    logging.minloglevel = 1
    try:
        return triangulate_points(
            reconstruction, str(database_path), str(image_path), str(work), clear_points, pipeline_options, False
        )
    finally:
        logging.minloglevel = level


def _pose_prior_bundle_adjustment(
    reconstruction: Reconstruction, pose_priors: dict[int, PosePrior], pipeline_options: IncrementalPipelineOptions
) -> str:
    options = cast(BundleAdjustmentOptions, pipeline_options.get_global_bundle_adjustment())
    options.solver_options.max_num_iterations = BUNDLE_ADJUSTMENT_MAX_ITERATIONS
    # COLMAP only uses the GPU direct sparse solver (cuDSS) up to 4000 images by default and falls back to an
    # iterative CPU solver above; merged maps exceed that, and a 24 GB GPU holds the factorization.
    options.max_num_images_direct_sparse_gpu_solver = GPU_SPARSE_SOLVER_MAX_IMAGES
    config = BundleAdjustmentConfig()
    for image_id in reconstruction.reg_image_ids():
        config.add_image(image_id)
    registered_priors = {
        i: p for i, p in pose_priors.items() if reconstruction.exists_image(i) and reconstruction.image(i).has_pose
    }
    summary = create_pose_prior_bundle_adjuster(
        options, PosePriorBundleAdjustmentOptions(), config, registered_priors, reconstruction
    ).solve()
    return (
        f"{summary.num_successful_steps + summary.num_unsuccessful_steps} iterations, "
        f"{summary.termination_type.name.lower()}, solver {summary.total_time_in_seconds:.0f}s, gpu {options.use_gpu}"
    )


def _filter(reconstruction: Reconstruction, max_reprojection_px: float) -> int:
    return ObservationManager(reconstruction).filter_all_points3D(
        max_reprojection_px, FILTER_MIN_TRIANGULATION_ANGLE_DEG
    )


def _drop_unsupported_frames(reconstruction: Reconstruction) -> int:
    dropped = 0
    for frame_id in list(reconstruction.reg_frame_ids()):
        frame = reconstruction.frame(frame_id)
        observations = sum(reconstruction.image(data_id.id).num_points3D for data_id in frame.image_ids)
        if observations < MIN_FRAME_OBSERVATIONS:
            reconstruction.deregister_frame(frame_id)
            dropped += 1
    return dropped


def _reanchor_dropped_frames(
    reconstruction: Reconstruction,
    frame_keys: dict[int, str],
    world_from_frame: dict[str, tuple[ndarray, ndarray]],
) -> int:
    # A dropped frame keeps its odometry motion relative to the nearest solved frame of the same recording:
    # world_from_dropped = world_from_solved * (prior_world_from_solved)^-1 * prior_world_from_dropped.
    solved = {frame_keys[frame_id]: frame_id for frame_id in reconstruction.reg_frame_ids() if frame_id in frame_keys}
    by_rig: dict[str, list[tuple[float, str]]] = {}
    for key in solved:
        rig, name = key.split("/")
        by_rig.setdefault(rig, []).append((_frame_time(name), key))
    reanchored = 0
    registered = set(reconstruction.reg_frame_ids())
    for frame_id in [f for f in reconstruction.frames if f not in registered and f in frame_keys]:
        key = frame_keys[frame_id]
        rig, name = key.split("/")
        if key not in world_from_frame or not by_rig.get(rig):
            continue
        anchor_key = min(by_rig[rig], key=lambda tk: abs(tk[0] - _frame_time(name)))[1]
        prior_anchor, prior_frame = _rigid(world_from_frame[anchor_key]), _rigid(world_from_frame[key])
        world_from_anchor = reconstruction.frame(solved[anchor_key]).rig_from_world.inverse()
        world_from_rig = world_from_anchor * prior_anchor.inverse() * prior_frame
        reconstruction.frame(frame_id).rig_from_world = world_from_rig.inverse()
        reconstruction.register_frame(frame_id)
        reanchored += 1
    return reanchored


def _frame_time(name: str) -> float:
    try:
        return float(name)
    except ValueError:
        return 0.0


def _rigid(pose: tuple[ndarray, ndarray]) -> Rigid3d:
    return Rigid3d(Rotation3d(pose[0]), pose[1])


def _frame_key(frame_images: list[Any]) -> str | None:
    # Image names are "<rig>/<camera>/<frame>.jpg"; every image of a frame shares rig and frame name.
    if not frame_images:
        return None
    rig, _, name = frame_images[0].name.split("/")
    return f"{rig}/{name.rsplit('.', 1)[0]}"


def _ref_camera_id(reconstruction: Reconstruction, frame: Any) -> int:
    return reconstruction.rig(frame.rig_id).ref_sensor_id.id
