"""Tests for the pure algorithmic pieces of sync.py (no network calls)."""

from sync import sliding_window_best_time


def test_constant_pace_matches_expected_time():
    # 5 m/s for 400s -> 1000m should take exactly 200s regardless of window start
    distances = [i * 5 for i in range(400)]
    times = list(range(400))
    assert sliding_window_best_time(distances, times, 1000) == 200


def test_finds_embedded_fast_segment_not_whole_activity_average():
    # slow (600m/300s) -> fast (1000m/125s) -> slow (600m/300s)
    distances, times = [], []
    d = 0
    for t in range(0, 300):
        d += 2
        distances.append(d); times.append(t)
    for t in range(300, 425):
        d += 8
        distances.append(d); times.append(t)
    for t in range(425, 725):
        d += 2
        distances.append(d); times.append(t)

    best = sliding_window_best_time(distances, times, 1000)
    whole_activity_equivalent = times[-1] / distances[-1] * 1000

    assert best is not None
    assert best <= 126  # true fast segment took 125s
    assert best < whole_activity_equivalent  # must beat the naive average


def test_returns_none_when_activity_too_short():
    assert sliding_window_best_time([0, 100, 200], [0, 10, 20], 5000) is None


def test_returns_none_for_empty_input():
    assert sliding_window_best_time([], [], 1000) is None


def test_exact_boundary_distance():
    # exactly 1000m covered by the whole series, no more no less
    distances = [0, 500, 1000]
    times = [0, 60, 130]
    assert sliding_window_best_time(distances, times, 1000) == 130
