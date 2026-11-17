from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, pi, sqrt
from time import monotonic

import numpy as np
from scipy.spatial.transform import Rotation

# Tracking a moving client from noisy, sometimes wrong localizations (a similar-looking view elsewhere in the map).
#
# The client sends the pose of each frame in its own odometry frame (ARCore/ARKit world, metres). The estimated
# quantity is the alignment odometry-from-map: it stays fixed while the device moves (up to slow odometry drift),
# so every correct localization measures the same alignment and a wrong one measures a different one.
#
# Each hypothesis is one candidate alignment with an isotropic Kalman variance (metres², at the device) and a
# probability. A frame's candidate poses update every hypothesis they agree with (the best-fitting candidate is
# fused, the hypothesis' probability rises); hypotheses with no agreeing candidate treat the frame's matches as
# false. "Unknown" is the probability that the device is somewhere no hypothesis covers (uniform over the map): it
# seeds a new hypothesis from each candidate, so consistent localizations elsewhere take over after two or three
# frames while a single stray match never does, and weak candidates that agree with the predicted trajectory still
# count.

Vector3 = np.ndarray[tuple[int], np.dtype[np.float64]]
Matrix4 = np.ndarray[tuple[int, int], np.dtype[np.float64]]

MAP_VOLUME_M3 = 2000.0  # spread of a false match: uniform over a house-sized map
MAX_CORRECT_PROBABILITY = 0.75  # one localization alone never makes a hypothesis confident
MIN_CORRECT_PROBABILITY = 0.02
MIN_UNKNOWN = 0.01  # always allow for being somewhere else
DRIFT_M2_PER_M = 0.002  # alignment variance added per metre moved (≈ 4.5 cm after 1 m)
GATE_SIGMAS = 3.0
GATE_ANGLE_RAD = 15 * pi / 180
MERGE_DISTANCE_M = 0.5
MERGE_ANGLE_RAD = 10 * pi / 180
MAX_HYPOTHESES = 8
MIN_HYPOTHESIS_PROBABILITY = 1e-4
SESSION_TTL_S = 15 * 60
CONFIDENT_PROBABILITY = 0.95
# Faster than anyone walks: the client's odometry re-anchored itself (ARCore/ARKit correct their world frame after
# relocalizing; seen as a 5 m step in one second), so every alignment is off by about the jump.
JUMP_SPEED_M_S = 3.0
JUMP_MIN_M = 1.0


@dataclass
class Candidate:
    map_from_camera: Matrix4
    confidence: float
    num_inliers: int
    source: str  # "retrieval" or "prior" (images near the predicted pose)


@dataclass
class Hypothesis:
    odom_from_map: Matrix4
    variance: float
    probability: float
    fixes: int = 1
    reanchoring: bool = False  # after an odometry jump: no heading gate until the next agreeing localization


@dataclass
class FrameResult:
    agreed: bool  # a candidate of this frame agreed with the most probable hypothesis
    selected: Candidate | None  # that candidate


@dataclass
class Tracker:
    hypotheses: list[Hypothesis] = field(default_factory=list)
    unknown: float = 1.0
    last_odom_position: Vector3 | None = None
    last_time: float | None = None
    jumps: int = 0
    last_seen: float = field(default_factory=monotonic)
    agreed: int = 0
    disagreed: int = 0

    @property
    def best(self) -> Hypothesis | None:
        return max(self.hypotheses, key=lambda h: h.probability, default=None)

    @property
    def confident(self) -> bool:
        best = self.best
        return best is not None and best.probability >= CONFIDENT_PROBABILITY and best.fixes >= 3

    def predicted_map_from_camera(self, odom_from_camera: Matrix4, min_probability: float = 0.5) -> Matrix4 | None:
        best = self.best
        if best is None or best.probability < min_probability:
            return None
        return np.linalg.inv(best.odom_from_map) @ odom_from_camera

    def update(self, odom_from_camera: Matrix4, candidates: list[Candidate], now: float | None = None) -> FrameResult:
        self.last_seen = monotonic()
        self._predict(odom_from_camera[:3, 3], self.last_seen if now is None else now)
        clutter = 1 / MAP_VOLUME_M3
        selected: dict[int, Candidate] = {}
        for h in self.hypotheses:
            best_term, best_candidate, best_gain_input = 0.0, None, None
            for c in candidates:
                p = _correct_probability(c.confidence)
                noise = _measurement_variance(c.confidence)
                predicted = h.odom_from_map @ c.map_from_camera  # where h puts the device for this candidate
                s = h.variance + noise
                d2 = float(np.sum((predicted[:3, 3] - odom_from_camera[:3, 3]) ** 2))
                angle_gated = not h.reanchoring and _angle(predicted, odom_from_camera) > GATE_ANGLE_RAD
                if d2 > GATE_SIGMAS**2 * s or angle_gated:
                    continue
                term = p * exp(-0.5 * d2 / s) / (2 * pi * s) ** 1.5
                if term > best_term:
                    best_term, best_candidate, best_gain_input = term, c, (predicted, s)
            if best_candidate is None or best_gain_input is None:
                h.probability *= clutter  # under h every candidate of this frame is a false match
                continue
            p = _correct_probability(best_candidate.confidence)
            h.probability *= (1 - p) * clutter + best_term
            predicted, s = best_gain_input
            gain = 1.0 if h.reanchoring else h.variance / s  # Kalman gain; a re-anchoring takes the new alignment
            h.odom_from_map = _nudge(predicted, odom_from_camera, gain) @ h.odom_from_map
            h.variance = _measurement_variance(best_candidate.confidence) if h.reanchoring else h.variance * (1 - gain)
            h.reanchoring = False
            h.fixes += 1
            selected[id(h)] = best_candidate
        # Under "unknown", each candidate is correct (a new place, density 1/volume) or false (also 1/volume).
        probabilities = [_correct_probability(c.confidence) for c in candidates]
        scale = 1 / max(1.0, sum(probabilities))
        seeds = [
            (
                Hypothesis(
                    odom_from_camera @ np.linalg.inv(c.map_from_camera),
                    _measurement_variance(c.confidence),
                    self.unknown * clutter * p * scale,
                ),
                c,
            )
            for c, p in zip(candidates, probabilities)
        ]
        self.hypotheses += [h for h, _ in seeds]
        self.unknown *= clutter * (1 - sum(probabilities) * scale)
        self._normalize()
        merged_into = self._merge(odom_from_camera[:3, 3])
        best = self.best
        chosen = selected.get(id(best)) if best is not None else None
        if best is not None and chosen is None:
            chosen = next((c for h, c in seeds if merged_into.get(id(h), h) is best), None)
        agreed = chosen is not None
        if candidates:
            self.agreed += agreed
            self.disagreed += not agreed
        return FrameResult(agreed, chosen)

    def _predict(self, position: Vector3, now: float) -> None:
        if self.last_odom_position is not None and self.last_time is not None:
            moved = float(np.linalg.norm(position - self.last_odom_position))
            speed = moved / max(now - self.last_time, 0.05)
            jumped = moved > JUMP_MIN_M and speed > JUMP_SPEED_M_S
            self.jumps += jumped
            for h in self.hypotheses:
                h.variance += moved**2 if jumped else DRIFT_M2_PER_M * moved
                h.reanchoring = h.reanchoring or jumped
        self.last_odom_position = position.copy()
        self.last_time = now

    def _normalize(self) -> None:
        total = self.unknown + sum(h.probability for h in self.hypotheses)
        if total <= 0:
            self.hypotheses.clear()
            self.unknown = 1.0
            return
        self.unknown /= total
        for h in self.hypotheses:
            h.probability /= total
        if self.unknown < MIN_UNKNOWN:
            scale = (1 - MIN_UNKNOWN) / (1 - self.unknown)
            for h in self.hypotheses:
                h.probability *= scale
            self.unknown = MIN_UNKNOWN

    # Hypotheses describing the same alignment (compared at the device) become one; unlikely ones are dropped.
    # Returns which hypothesis each merged one went into.
    def _merge(self, at: Vector3) -> dict[int, Hypothesis]:
        self.hypotheses.sort(key=lambda h: -h.probability)
        kept: list[Hypothesis] = []
        merged_into: dict[int, Hypothesis] = {}
        for h in self.hypotheses:
            same = next((k for k in kept if _same_alignment(k.odom_from_map, h.odom_from_map, at)), None)
            if same is None:
                kept.append(h)
                continue
            same.probability += h.probability
            same.fixes = max(same.fixes, h.fixes)
            merged_into[id(h)] = same
        self.hypotheses = [h for h in kept if h.probability >= MIN_HYPOTHESIS_PROBABILITY][:MAX_HYPOTHESES]
        return merged_into


class Sessions:
    def __init__(self) -> None:
        self._trackers: dict[tuple[str, str], Tracker] = {}

    def get(self, session_id: str, reconstruction_id: str) -> Tracker:
        now = monotonic()
        for key in [k for k, t in self._trackers.items() if now - t.last_seen > SESSION_TTL_S]:
            del self._trackers[key]
        return self._trackers.setdefault((session_id, reconstruction_id), Tracker())


def _correct_probability(confidence: float) -> float:
    return min(MAX_CORRECT_PROBABILITY, max(MIN_CORRECT_PROBABILITY, confidence))


def _measurement_variance(confidence: float) -> float:
    return (0.1 + 0.3 * (1 - min(1.0, max(0.0, confidence)))) ** 2


def _angle(a: Matrix4, b: Matrix4) -> float:
    return float(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude())


# The rigid motion that moves `source` the fraction `gain` of the way to `target`, rotating about source's position.
def _nudge(source: Matrix4, target: Matrix4, gain: float) -> Matrix4:
    rotation = Rotation.from_rotvec(Rotation.from_matrix(target[:3, :3] @ source[:3, :3].T).as_rotvec() * gain)
    motion = np.eye(4)
    motion[:3, :3] = rotation.as_matrix()
    position = source[:3, 3]
    motion[:3, 3] = position + gain * (target[:3, 3] - position) - motion[:3, :3] @ position
    return motion


def _same_alignment(a: Matrix4, b: Matrix4, at: Vector3) -> bool:
    difference = a @ np.linalg.inv(b)  # identity when the alignments agree
    moved = difference[:3, :3] @ at + difference[:3, 3] - at
    return float(np.linalg.norm(moved)) <= MERGE_DISTANCE_M and _angle(difference, np.eye(4)) <= MERGE_ANGLE_RAD


def pose_from_quaternion(translation: list[float], quaternion_xyzw: list[float]) -> Matrix4:
    m = np.eye(4)
    m[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
    m[:3, 3] = translation
    return m


def sigma(h: Hypothesis) -> float:
    return sqrt(h.variance)
