from __future__ import annotations

import os
from io import BytesIO
from typing import Any

from numpy import asarray, float16, float32, load, ndarray, savez

# Stage checkpoints for workers that can be preempted (rented interruptible GPUs): extracted features and
# feature matches are written to object storage as they complete, so a re-leased job resumes after the last
# finished stage instead of starting over. Off unless RECONSTRUCTOR_CHECKPOINTS=1, because on a single local
# worker the upload costs more than it saves.
ENABLED = os.environ.get("RECONSTRUCTOR_CHECKPOINTS") == "1"
PREFIX = "checkpoints"


def load_features(
    s3_client: Any, bucket: str, reconstruction_id: object
) -> tuple[dict[str, ndarray], dict[str, ndarray], dict[str, ndarray], dict[str, tuple[int, int]]] | None:
    arrays = _get(s3_client, bucket, reconstruction_id, "features")
    if arrays is None:
        return None
    names = [str(n) for n in arrays["names"]]
    keypoints = {n: arrays[f"k{i}"] for i, n in enumerate(names)}
    descriptors = {n: arrays[f"d{i}"].astype(float32) for i, n in enumerate(names)}
    global_descriptors = {n: arrays[f"g{i}"] for i, n in enumerate(names)}
    sizes = {n: (int(arrays["sizes"][i][0]), int(arrays["sizes"][i][1])) for i, n in enumerate(names)}
    return keypoints, descriptors, global_descriptors, sizes


def save_features(
    s3_client: Any,
    bucket: str,
    reconstruction_id: object,
    keypoints: dict[str, ndarray],
    descriptors: dict[str, ndarray],
    global_descriptors: dict[str, ndarray],
    sizes: dict[str, tuple[int, int]],
) -> None:
    names = list(keypoints)
    arrays: dict[str, ndarray] = {"names": asarray(names), "sizes": asarray([sizes[n] for n in names])}
    for i, n in enumerate(names):
        arrays[f"k{i}"] = keypoints[n]
        arrays[f"d{i}"] = descriptors[n].astype(float16)  # matching is insensitive to half precision; halves size
        arrays[f"g{i}"] = global_descriptors[n]
    _put(s3_client, bucket, reconstruction_id, "features", arrays)


def load_matches(
    s3_client: Any, bucket: str, reconstruction_id: object
) -> dict[tuple[str, str], tuple[ndarray, ndarray]] | None:
    arrays = _get(s3_client, bucket, reconstruction_id, "matches")
    if arrays is None:
        return None
    pairs = arrays["pairs"]
    return {(str(a), str(b)): (arrays[f"a{i}"], arrays[f"b{i}"]) for i, (a, b) in enumerate(pairs)}


def save_matches(
    s3_client: Any, bucket: str, reconstruction_id: object, matches: dict[tuple[str, str], tuple[ndarray, ndarray]]
) -> None:
    pairs = list(matches)
    arrays: dict[str, ndarray] = {"pairs": asarray(pairs)}
    for i, pair in enumerate(pairs):
        arrays[f"a{i}"], arrays[f"b{i}"] = matches[pair]
    _put(s3_client, bucket, reconstruction_id, "matches", arrays)


def _get(s3_client: Any, bucket: str, reconstruction_id: object, name: str) -> Any:
    if not ENABLED:
        return None
    try:
        body = s3_client.get_object(Bucket=bucket, Key=f"{reconstruction_id}/{PREFIX}/{name}.npz")["Body"].read()
    except Exception:
        return None
    print(f"[checkpoint] resuming from {name} checkpoint ({len(body) / 1e6:.1f} MB)")
    return load(BytesIO(body), allow_pickle=False)


def _put(s3_client: Any, bucket: str, reconstruction_id: object, name: str, arrays: dict[str, ndarray]) -> None:
    if not ENABLED:
        return
    buffer = BytesIO()
    savez(buffer, **arrays)
    body = buffer.getvalue()
    s3_client.put_object(Bucket=bucket, Key=f"{reconstruction_id}/{PREFIX}/{name}.npz", Body=body)
    print(f"[checkpoint] saved {name} checkpoint ({len(body) / 1e6:.1f} MB)")
