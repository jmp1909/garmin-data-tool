"""Tests for analytics.py against hand-computed fixtures -- the point is to
prove the math is right, not just that it produces a plausible-looking chart."""

import math

import pandas as pd
import pytest

import analytics as A


def acts_df(rows):
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def daily_df(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


# --------------------------------------------------------------------------
# compute_daily_load
# --------------------------------------------------------------------------

def test_daily_load_matches_hand_computed_trimp():
    acts = acts_df([{"date": "2026-07-01", "avg_hr": 150, "duration_min": 30, "max_hr": 190}])
    daily = daily_df([{"date": "2026-07-01", "rhr.value": 50}])

    load = A.compute_daily_load(acts, daily)

    hrr = (150 - 50) / (190 - 50)
    expected = 30 * hrr * 0.64 * math.exp(1.92 * hrr)
    assert load.loc[pd.Timestamp("2026-07-01")] == pytest.approx(expected, rel=1e-9)


def test_daily_load_sums_two_activities_same_day():
    acts = acts_df([
        {"date": "2026-07-01", "avg_hr": 150, "duration_min": 30, "max_hr": 190},
        {"date": "2026-07-01", "avg_hr": 140, "duration_min": 20, "max_hr": 190},
    ])
    daily = daily_df([{"date": "2026-07-01", "rhr.value": 50}])

    load = A.compute_daily_load(acts, daily)

    hrr1, hrr2 = (150 - 50) / 140, (140 - 50) / 140
    expected = 30 * hrr1 * 0.64 * math.exp(1.92 * hrr1) + 20 * hrr2 * 0.64 * math.exp(1.92 * hrr2)
    assert load.loc[pd.Timestamp("2026-07-01")] == pytest.approx(expected, rel=1e-9)


def test_daily_load_falls_back_to_default_resting_hr_when_missing():
    acts = acts_df([{"date": "2026-07-01", "avg_hr": 150, "duration_min": 30, "max_hr": 190}])
    daily = daily_df([{"date": "2026-07-01", "rhr.value": None}])

    load = A.compute_daily_load(acts, daily, resting_hr_default=55.0)

    hrr = (150 - 55) / (190 - 55)
    expected = 30 * hrr * 0.64 * math.exp(1.92 * hrr)
    assert load.loc[pd.Timestamp("2026-07-01")] == pytest.approx(expected, rel=1e-9)


def test_daily_load_empty_activities_returns_empty_series():
    assert A.compute_daily_load(acts_df([]), daily_df([])).empty


# --------------------------------------------------------------------------
# compute_pmc
# --------------------------------------------------------------------------

def test_pmc_first_day_equals_that_days_load():
    load = pd.Series([100.0], index=[pd.Timestamp("2026-07-01")])
    pmc = A.compute_pmc(load, ctl_days=42, atl_days=7)
    assert pmc.loc[pd.Timestamp("2026-07-01"), "ctl"] == pytest.approx(100.0)
    assert pmc.loc[pd.Timestamp("2026-07-01"), "atl"] == pytest.approx(100.0)
    assert pmc.loc[pd.Timestamp("2026-07-01"), "tsb"] == pytest.approx(0.0)


def test_pmc_recursive_ewm_matches_hand_computation():
    # y_0 = x_0; y_t = alpha*x_t + (1-alpha)*y_{t-1}
    load = pd.Series([100.0, 0.0, 0.0], index=pd.date_range("2026-07-01", periods=3))
    pmc = A.compute_pmc(load, ctl_days=42, atl_days=7)

    alpha_ctl = 1 / 42
    day1_ctl = 100.0
    day2_ctl = alpha_ctl * 0.0 + (1 - alpha_ctl) * day1_ctl
    day3_ctl = alpha_ctl * 0.0 + (1 - alpha_ctl) * day2_ctl

    assert pmc["ctl"].iloc[1] == pytest.approx(day2_ctl)
    assert pmc["ctl"].iloc[2] == pytest.approx(day3_ctl)


def test_pmc_fills_gaps_with_zero_load_rest_days():
    # two activity days three days apart -- the gap day must count as zero load
    load = pd.Series([50.0, 50.0], index=[pd.Timestamp("2026-07-01"), pd.Timestamp("2026-07-04")])
    pmc = A.compute_pmc(load)
    assert len(pmc) == 4  # 07-01, 02, 03, 04
    assert pmc.loc[pd.Timestamp("2026-07-02"), "ctl"] < pmc.loc[pd.Timestamp("2026-07-01"), "ctl"]


def test_pmc_empty_input():
    result = A.compute_pmc(pd.Series(dtype=float))
    assert result.empty


# --------------------------------------------------------------------------
# compute_acwr
# --------------------------------------------------------------------------

def test_acwr_constant_load_converges_to_one():
    load = pd.Series([10.0] * 30, index=pd.date_range("2026-07-01", periods=30))
    acwr = A.compute_acwr(load, acute_days=7, chronic_days=28)
    assert acwr.iloc[-1] == pytest.approx(1.0, rel=1e-6)


def test_acwr_hand_computed_short_series():
    load = pd.Series([10.0, 20.0, 30.0], index=pd.date_range("2026-07-01", periods=3))
    acwr = A.compute_acwr(load, acute_days=2, chronic_days=3)
    # day 3: acute = mean(last 2) = mean(20,30) = 25; chronic = mean(10,20,30) = 20
    assert acwr.iloc[-1] == pytest.approx(25 / 20)


@pytest.mark.parametrize("ratio,expected", [
    (0.5, "warning"),
    (0.8, "good"),
    (1.0, "good"),
    (1.3, "good"),
    (1.4, "warning"),
    (1.6, "critical"),
    (None, None),
])
def test_classify_acwr_boundaries(ratio, expected):
    assert A.classify_acwr(ratio) == expected


@pytest.mark.parametrize("tsb,expected", [
    (10, "good"),
    (-5, "good"),
    (-15, "warning"),
    (-35, "critical"),
    (None, None),
])
def test_classify_tsb_boundaries(tsb, expected):
    assert A.classify_tsb(tsb) == expected


# --------------------------------------------------------------------------
# heat_adjusted_pace / efficiency_factor
# --------------------------------------------------------------------------

def test_heat_adjusted_pace_hand_computed():
    acts = acts_df([{"date": "2026-07-01", "avg_pace_min_per_km": 6.0, "weather": {"temp_c": 25.0}}])
    result = A.heat_adjusted_pace(acts, threshold_c=15.0, slowdown_per_degree=0.005)
    expected = 6.0 / (1 + 0.005 * (25.0 - 15.0))
    assert result.loc[pd.Timestamp("2026-07-01"), "adjusted_pace"] == pytest.approx(expected)


def test_heat_adjusted_pace_below_threshold_is_unchanged():
    acts = acts_df([{"date": "2026-07-01", "avg_pace_min_per_km": 6.0, "weather": {"temp_c": 10.0}}])
    result = A.heat_adjusted_pace(acts, threshold_c=15.0)
    assert result.loc[pd.Timestamp("2026-07-01"), "adjusted_pace"] == pytest.approx(6.0)


def test_heat_adjusted_pace_skips_activities_without_weather():
    acts = acts_df([{"date": "2026-07-01", "avg_pace_min_per_km": 6.0, "weather": None}])
    result = A.heat_adjusted_pace(acts)
    assert result.empty


def test_efficiency_factor_hand_computed():
    acts = acts_df([{"date": "2026-07-01", "avg_pace_min_per_km": 5.0, "avg_hr": 150}])
    ef = A.efficiency_factor(acts)
    speed_m_s = 1000 / (5.0 * 60)
    assert ef.loc[pd.Timestamp("2026-07-01")] == pytest.approx(round(speed_m_s / 150, 4))


# --------------------------------------------------------------------------
# Riegel race prediction
# --------------------------------------------------------------------------

def test_riegel_predict_exact_formula():
    # 20:00 5K -> predicted 10K
    predicted = A.riegel_predict(1200, 5000, 10000, exponent=1.06)
    assert predicted == pytest.approx(1200 * 2 ** 1.06)


def test_riegel_predict_same_distance_returns_same_time():
    assert A.riegel_predict(1200, 5000, 5000) == pytest.approx(1200)


def test_predict_race_times_uses_longest_known_reference():
    # both 5k and 10k known -- must extrapolate from 10k (the longer one)
    best_efforts = {"5k": 1200, "10k": 2500}
    result = A.predict_race_times(best_efforts)
    expected_5k = round(A.riegel_predict(2500, 10000, 5000), 1)  # predict_race_times rounds to 1dp
    assert result["5k"] == pytest.approx(expected_5k)


def test_predict_race_times_empty_input():
    assert A.predict_race_times({}) == {}
    assert A.predict_race_times(None) == {}


# --------------------------------------------------------------------------
# Best efforts leaderboard
# --------------------------------------------------------------------------

def test_leaderboard_sorted_ascending_and_limited():
    acts = acts_df([
        {"date": "2026-07-01", "name": "slow", "distance_km": 5, "best_efforts": {"5k": 2000}},
        {"date": "2026-07-02", "name": "fast", "distance_km": 5, "best_efforts": {"5k": 1500}},
        {"date": "2026-07-03", "name": "mid", "distance_km": 5, "best_efforts": {"5k": 1800}},
    ])
    board = A.best_efforts_leaderboard(acts, "5k", top_n=2)
    assert list(board["name"]) == ["fast", "mid"]
    assert list(board["time_sec"]) == [1500, 1800]


def test_leaderboard_excludes_activities_missing_that_distance():
    acts = acts_df([
        {"date": "2026-07-01", "name": "has5k", "distance_km": 5, "best_efforts": {"5k": 1500}},
        {"date": "2026-07-02", "name": "no5k", "distance_km": 2, "best_efforts": {"1k": 300}},
    ])
    board = A.best_efforts_leaderboard(acts, "5k")
    assert list(board["name"]) == ["has5k"]


def test_leaderboard_empty_when_no_data():
    acts = acts_df([{"date": "2026-07-01", "name": "x", "distance_km": 1, "best_efforts": None}])
    board = A.best_efforts_leaderboard(acts, "5k")
    assert board.empty


# --------------------------------------------------------------------------
# this_week_vs_alltime_best
# --------------------------------------------------------------------------

def test_this_week_vs_alltime_picks_correct_week_bucket():
    # week of the "current" date (max date in acts) vs an older week
    acts = acts_df([
        {"date": "2026-06-01", "best_efforts": {"5k": 1400}},  # older week, faster (all-time best)
        {"date": "2026-07-27", "best_efforts": {"5k": 1600}},  # same week as max date
        {"date": "2026-07-31", "best_efforts": {"5k": 1550}},  # same week as max date (current week best)
    ])
    this_week, all_time = A.this_week_vs_alltime_best(acts, "5k")
    assert all_time == 1400
    assert this_week == 1550


def test_this_week_vs_alltime_no_data_returns_none_none():
    acts = acts_df([{"date": "2026-07-01", "best_efforts": None}])
    assert A.this_week_vs_alltime_best(acts, "5k") == (None, None)


# --------------------------------------------------------------------------
# Polarization
# --------------------------------------------------------------------------

def test_polarization_hand_computed():
    acts = acts_df([
        {"date": "2026-07-01", "hr_zones": {"zone_1": 100, "zone_2": 100, "zone_3": 50, "zone_4": 25, "zone_5": 25}},
    ])
    result = A.polarization(acts)
    # easy=200, moderate=50, hard=50, total=300 -- polarization() rounds to 1dp
    assert result["easy_pct"] == pytest.approx(round(200 / 300 * 100, 1))
    assert result["moderate_pct"] == pytest.approx(round(50 / 300 * 100, 1))
    assert result["hard_pct"] == pytest.approx(round(50 / 300 * 100, 1))


def test_polarization_no_hr_zone_data_returns_empty_dict():
    acts = acts_df([{"date": "2026-07-01", "hr_zones": None}])
    assert A.polarization(acts) == {}
