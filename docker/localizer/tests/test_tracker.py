from math import cos, sin

import numpy as np
from src.tracker import Candidate, Matrix4, Tracker

TRUTH = np.array([[cos(0.7), 0, sin(0.7), 3], [0, 1, 0, 0.2], [-sin(0.7), 0, cos(0.7), -5], [0, 0, 0, 1]])


def test_consistent_localizations_win_and_stray_matches_are_suppressed() -> None:
    rng = np.random.default_rng(1)
    tracker = Tracker()
    stray_agreed = 0
    for i in range(40):
        odom = _odom(i)
        candidates = [_correct(odom, rng, 0.6)]
        if i % 4 == 3:
            candidates = [_wrong(rng, 0.7)]  # a stronger match somewhere else
        result = tracker.update(odom, candidates)
        if i % 4 == 3 and i > 3:
            stray_agreed += result.agreed
    best = tracker.best
    assert best is not None
    assert best.probability > 0.9
    assert _error(best.odom_from_map, _odom(39)) < 0.1
    assert stray_agreed == 0


def test_weak_candidate_that_fits_the_trajectory_is_selected_over_a_stronger_one_elsewhere() -> None:
    rng = np.random.default_rng(2)
    tracker = Tracker()
    for i in range(6):
        tracker.update(_odom(i), [_correct(_odom(i), rng, 0.7)])
    odom = _odom(6)
    weak = _correct(odom, rng, 0.08)
    result = tracker.update(odom, [_wrong(rng, 0.6), weak])
    assert result.agreed
    assert result.selected is weak


def test_a_wrong_first_match_is_replaced_after_a_few_consistent_ones() -> None:
    rng = np.random.default_rng(3)
    tracker = Tracker()
    tracker.update(_odom(0), [_wrong(rng, 0.9)])
    first = tracker.best
    for i in range(1, 4):
        tracker.update(_odom(i), [_correct(_odom(i), rng, 0.6)])
    best = tracker.best
    assert best is not None and best is not first
    assert best.probability > 0.8
    assert _error(best.odom_from_map, _odom(3)) < 0.15


def test_an_odometry_jump_is_followed_by_the_next_good_localization() -> None:
    rng = np.random.default_rng(5)
    tracker = Tracker()
    for i in range(8):
        tracker.update(_odom(i), [_correct(_odom(i), rng, 0.7)], now=float(i))
    # The client's odometry re-anchors: from frame 8 on, its world frame is shifted 5 m and turned 40°.
    shift = np.array([[cos(0.7), 0, sin(0.7), 5], [0, 1, 0, 0], [-sin(0.7), 0, cos(0.7), 1], [0, 0, 0, 1.0]])
    for i in range(8, 11):
        odom = shift @ _odom(i)
        truth_candidate = Candidate(np.linalg.inv(TRUTH) @ _odom(i), 0.7, 100, "retrieval")
        result = tracker.update(odom, [truth_candidate], now=float(i))
        assert result.agreed
    best = tracker.best
    assert best is not None and best.probability > 0.9
    assert tracker.jumps == 1
    estimated = np.linalg.inv(best.odom_from_map) @ (shift @ _odom(10))
    assert np.linalg.norm(estimated[:3, 3] - (np.linalg.inv(TRUTH) @ _odom(10))[:3, 3]) < 0.1


def test_one_localization_alone_is_not_confident() -> None:
    tracker = Tracker()
    tracker.update(_odom(0), [_correct(_odom(0), np.random.default_rng(4), 0.95)])
    best = tracker.best
    assert best is not None and best.probability <= 0.75


def _odom(i: int) -> Matrix4:
    a = i * 0.1
    return np.array([[cos(a), 0, sin(a), i * 0.3], [0, 1, 0, 1.4], [-sin(a), 0, cos(a), i * 0.1], [0, 0, 0, 1]])


def _correct(odom: Matrix4, rng: np.random.Generator, confidence: float) -> Candidate:
    m = np.linalg.inv(TRUTH) @ odom
    m[:3, 3] += rng.uniform(-0.05, 0.05, 3)
    return Candidate(m, confidence, 100, "retrieval")


def _wrong(rng: np.random.Generator, confidence: float) -> Candidate:
    a = rng.uniform(0, 6)
    m = np.array([[cos(a), 0, sin(a), 0], [0, 1, 0, 0], [-sin(a), 0, cos(a), 0], [0, 0, 0, 1.0]])
    m[:3, 3] = [rng.uniform(0, 20), 0, rng.uniform(0, 20)]
    return Candidate(m, confidence, 100, "retrieval")


def _error(odom_from_map: Matrix4, odom: Matrix4) -> float:
    estimated = np.linalg.inv(odom_from_map) @ odom
    truth = np.linalg.inv(TRUTH) @ odom
    return float(np.linalg.norm(estimated[:3, 3] - truth[:3, 3]))
