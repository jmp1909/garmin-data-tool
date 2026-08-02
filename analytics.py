"""
Training-load and performance models computed from your own synced data --
none of this comes from Garmin directly. Pure functions over pandas
DataFrames/Series, independent of Streamlit and the Garmin client, so they
can be unit tested in isolation (see test_analytics.py).

Methodology notes (the honest version, not just what the chart says):

- Daily training load uses the classic Banister exponential TRIMP:
  load = duration_min * HRr * 0.64 * exp(1.92 * HRr), where HRr is heart
  rate reserve fraction. The 0.64/1.92 coefficients are the commonly-cited
  male values; the model doesn't correct for sex-specific coefficients,
  which is a known simplification.
- Fitness/Fatigue/Form (CTL/ATL/TSB) is the standard Performance Manager
  Chart model: CTL is an exponentially-weighted 42-day average of load,
  ATL a 7-day EWA, TSB = CTL - ATL.
- ACWR (acute:chronic workload ratio) is 7-day average load over 28-day
  average load -- the standard sports-science injury-risk heuristic, with
  0.8-1.3 considered the "sweet spot" per the published literature.
- Heat adjustment is a simple linear approximation (~0.5%/deg C above
  15C), not a validated physiological model -- treat it as directional.
- Race predictions use Riegel's formula (T2 = T1 * (D2/D1)^1.06) fitted to
  your own best-known effort, shown alongside Garmin's own estimate.
"""

import math

import pandas as pd

from sync import BEST_EFFORT_DISTANCES_M

STANDARD_DISTANCES_M = {"5k": 5000, "10k": 10000, "half": 21097.5, "marathon": 42195}


# --------------------------------------------------------------------------
# Training load / PMC / ACWR
# --------------------------------------------------------------------------

def compute_daily_load(acts: pd.DataFrame, daily: pd.DataFrame, max_hr=None, resting_hr_default=55.0) -> pd.Series:
    """Banister TRIMP-based training load per activity, summed by day."""
    if acts.empty:
        return pd.Series(dtype=float)
    if max_hr is None:
        observed = acts["max_hr"].dropna() if "max_hr" in acts.columns else pd.Series(dtype=float)
        max_hr = observed.max() if not observed.empty else 190.0

    def resting_hr_for(day):
        if "rhr.value" in daily.columns and day in daily.index:
            v = daily.loc[day, "rhr.value"]
            if pd.notna(v):
                return v
        return resting_hr_default

    dates, loads = [], []
    for _, row in acts.iterrows():
        avg_hr, duration_min = row.get("avg_hr"), row.get("duration_min")
        if pd.isna(avg_hr) or pd.isna(duration_min):
            continue
        day = pd.Timestamp(row["date"]).normalize()
        resting_hr = resting_hr_for(day)
        hr_reserve = max_hr - resting_hr
        if hr_reserve <= 0:
            continue
        hrr = max(0.0, min(1.0, (avg_hr - resting_hr) / hr_reserve))
        load = duration_min * hrr * 0.64 * math.exp(1.92 * hrr)
        dates.append(day)
        loads.append(load)

    if not loads:
        return pd.Series(dtype=float)
    return pd.Series(loads, index=dates).groupby(level=0).sum().sort_index()


def compute_pmc(daily_load: pd.Series, ctl_days: int = 42, atl_days: int = 7) -> pd.DataFrame:
    """Fitness (CTL), fatigue (ATL), form (TSB) -- reindexed to a continuous
    daily series with zero-load rest days, since the exponential decay only
    means what it's supposed to mean over an unbroken calendar."""
    if daily_load.empty:
        return pd.DataFrame(columns=["ctl", "atl", "tsb"])
    full_range = pd.date_range(daily_load.index.min(), daily_load.index.max(), freq="D")
    load = daily_load.reindex(full_range, fill_value=0.0)
    ctl = load.ewm(alpha=1 / ctl_days, adjust=False).mean()
    atl = load.ewm(alpha=1 / atl_days, adjust=False).mean()
    return pd.DataFrame({"ctl": ctl, "atl": atl, "tsb": ctl - atl})


def compute_acwr(daily_load: pd.Series, acute_days: int = 7, chronic_days: int = 28) -> pd.Series:
    if daily_load.empty:
        return pd.Series(dtype=float)
    full_range = pd.date_range(daily_load.index.min(), daily_load.index.max(), freq="D")
    load = daily_load.reindex(full_range, fill_value=0.0)
    acute = load.rolling(acute_days, min_periods=1).mean()
    chronic = load.rolling(chronic_days, min_periods=1).mean().replace(0, pd.NA)
    return (acute / chronic).rename("acwr")


def classify_tsb(tsb):
    if tsb is None or pd.isna(tsb):
        return None
    if tsb < -30:
        return "critical"
    if tsb < -10:
        return "warning"
    return "good"


def classify_acwr(ratio):
    if ratio is None or pd.isna(ratio):
        return None
    if ratio < 0.8:
        return "warning"
    if ratio <= 1.3:
        return "good"
    if ratio <= 1.5:
        return "warning"
    return "critical"


# --------------------------------------------------------------------------
# Pace quality
# --------------------------------------------------------------------------

def heat_adjusted_pace(acts: pd.DataFrame, threshold_c: float = 15.0, slowdown_per_degree: float = 0.005) -> pd.DataFrame:
    """Returns a date-indexed DataFrame with raw and heat-adjusted pace
    (min/km) for activities that have weather data attached."""
    if acts.empty or "weather" not in acts.columns:
        return pd.DataFrame(columns=["raw_pace", "adjusted_pace"])
    rows = []
    for _, row in acts.iterrows():
        weather, pace = row.get("weather"), row.get("avg_pace_min_per_km")
        if not isinstance(weather, dict) or pd.isna(pace):
            continue
        temp_c = weather.get("temp_c")
        if temp_c is None:
            continue
        excess = max(0.0, temp_c - threshold_c)
        adjusted = pace / (1 + slowdown_per_degree * excess)
        rows.append({"date": pd.Timestamp(row["date"]).normalize(), "raw_pace": pace, "adjusted_pace": adjusted, "temp_c": temp_c})
    if not rows:
        return pd.DataFrame(columns=["raw_pace", "adjusted_pace"])
    return pd.DataFrame(rows).set_index("date").sort_index()


def efficiency_factor(acts: pd.DataFrame) -> pd.Series:
    """Speed (m/s) per heartbeat, per activity -- rising means the same
    effort is producing more speed."""
    if acts.empty:
        return pd.Series(dtype=float)
    dates, values = [], []
    for _, row in acts.iterrows():
        pace, hr = row.get("avg_pace_min_per_km"), row.get("avg_hr")
        if pd.isna(pace) or pd.isna(hr) or hr in (0, None) or pace <= 0:
            continue
        speed_m_s = 1000 / (pace * 60)
        dates.append(pd.Timestamp(row["date"]).normalize())
        values.append(round(speed_m_s / hr, 4))
    if not values:
        return pd.Series(dtype=float)
    return pd.Series(values, index=dates).sort_index()


# --------------------------------------------------------------------------
# Race prediction
# --------------------------------------------------------------------------

def riegel_predict(known_time_sec, known_distance_m, target_distance_m, exponent: float = 1.06):
    if not known_time_sec or not known_distance_m:
        return None
    return known_time_sec * (target_distance_m / known_distance_m) ** exponent


def predict_race_times(all_time_best_efforts: dict, exponent: float = 1.06) -> dict:
    """all_time_best_efforts: {"1k": sec, "3k": sec, "5k": sec, "10k": sec}.
    Extrapolates from the longest known effort, since Riegel is most
    reliable predicting from a similar-or-longer reference distance."""
    known = {BEST_EFFORT_DISTANCES_M[k]: v for k, v in (all_time_best_efforts or {}).items()
             if v and k in BEST_EFFORT_DISTANCES_M}
    if not known:
        return {}
    reference_dist = max(known)
    reference_time = known[reference_dist]
    return {label: round(riegel_predict(reference_time, reference_dist, target_m, exponent), 1)
            for label, target_m in STANDARD_DISTANCES_M.items()}


# --------------------------------------------------------------------------
# Best efforts
# --------------------------------------------------------------------------

def best_efforts_leaderboard(acts: pd.DataFrame, distance_label: str, top_n: int = 5) -> pd.DataFrame:
    rows = []
    for _, row in acts.iterrows():
        be = row.get("best_efforts")
        if not isinstance(be, dict) or distance_label not in be:
            continue
        rows.append({
            "date": row["date"], "name": row.get("name"), "distance_km": row.get("distance_km"),
            "run_type": row.get("run_type"), "time_sec": be[distance_label],
        })
    cols = ["date", "name", "distance_km", "run_type", "time_sec"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows).sort_values("time_sec").head(top_n).reset_index(drop=True)
    df.index = df.index + 1
    return df


def this_week_vs_alltime_best(acts: pd.DataFrame, distance_label: str):
    """Returns (this_week_best_sec, all_time_best_sec), either possibly None."""
    pairs = [(row["date"], row["best_efforts"][distance_label]) for _, row in acts.iterrows()
             if isinstance(row.get("best_efforts"), dict) and distance_label in row["best_efforts"]]
    if not pairs:
        return None, None
    all_time_best = min(t for _, t in pairs)
    current_week = acts["date"].max().to_period("W")
    this_week_times = [t for d, t in pairs if pd.Timestamp(d).to_period("W") == current_week]
    this_week_best = min(this_week_times) if this_week_times else None
    return this_week_best, all_time_best


# --------------------------------------------------------------------------
# Polarization
# --------------------------------------------------------------------------

def polarization(acts: pd.DataFrame) -> dict:
    """% of total HR-zone time across all enriched activities spent easy
    (Z1-2), moderate (Z3), hard (Z4-5) -- the 80/20 polarized-training read."""
    easy = moderate = hard = 0.0
    if "hr_zones" not in acts.columns:
        return {}
    for _, row in acts.iterrows():
        zones = row.get("hr_zones")
        if not isinstance(zones, dict):
            continue
        easy += zones.get("zone_1", 0) + zones.get("zone_2", 0)
        moderate += zones.get("zone_3", 0)
        hard += zones.get("zone_4", 0) + zones.get("zone_5", 0)
    total = easy + moderate + hard
    if total == 0:
        return {}
    return {
        "easy_pct": round(easy / total * 100, 1),
        "moderate_pct": round(moderate / total * 100, 1),
        "hard_pct": round(hard / total * 100, 1),
    }
