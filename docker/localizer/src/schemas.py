from enum import Enum
from uuid import UUID

from core.localization_metrics import LocalizationMetrics
from core.transform import Transform
from pydantic import BaseModel


class LoadState(str, Enum):
    PENDING = "pending"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


class LoadStateResponse(BaseModel):
    status: LoadState
    error: str | None = None


class Localization(BaseModel):
    id: UUID
    transform: Transform
    metrics: LocalizationMetrics


class TrackedLocalization(BaseModel):
    # The most probable place of the client after this frame (see tracker.py); None fields until there is one.
    id: UUID | None  # reconstruction
    probability: float
    sigma_m: float | None
    fixes: int
    hypotheses: int
    unknown: float
    camera_from_map: Transform | None  # this frame's camera (OpenCV axes, sensor image orientation)
    odometry_from_map: Transform | None  # client odometry frame from map: compose with later odometry poses
    frame_candidates: int  # candidate poses this frame produced
    frame_agreed: bool  # one of them agreed with the most probable place
    frame_confidence: float | None  # that candidate's localization confidence
    frame_inliers: int | None
    frame_source: str | None  # "retrieval" or "prior" (images near the predicted pose)
