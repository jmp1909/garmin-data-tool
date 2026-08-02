"""
Pulls training data from Garmin Connect and appends it to data/garmin_history.json.

Run standalone:
    python sync.py

Or import run_sync() from dashboard.py to trigger a sync from the Streamlit app.
"""

import json
import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from garminconnect import Garmin

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = Path(os.getenv("GARMIN_DATA_FILE", BASE_DIR / "data" / "garmin_history.json"))
TOKEN_STORE = Path(os.getenv("GARMIN_TOKEN_STORE", BASE_DIR / ".garmin_tokens"))
BACKFILL_DAYS = int(os.getenv("GARMIN_BACKFILL_DAYS", 84))  # ~12 weeks, used on first run only
ACTIVITY_LOOKBACK_DAYS = int(os.getenv("GARMIN_ACTIVITY_LOOKBACK_DAYS", 10))
# One-time deep pull so PMC/ACWR (need ~6 weeks of load history) and the all-time
# best-efforts leaderboard aren't starved by the normal 10-day rolling window.
ACTIVITY_BACKFILL_DAYS = int(os.getenv("GARMIN_ACTIVITY_BACKFILL_DAYS", 365))
# Per-activity enrichment (best efforts, weather, HR zones) costs 3 extra API
# calls each -- bounded separately so a multi-year history doesn't turn a sync
# into thousands of requests.
DETAIL_ENRICHMENT_DAYS = int(os.getenv("GARMIN_DETAIL_ENRICHMENT_DAYS", 120))

BEST_EFFORT_DISTANCES_M = {"1k": 1000, "3k": 3000, "5k": 5000, "10k": 10000}

log = logging.getLogger("garmin_sync")


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------

def prompt_mfa_console():
    try:
        return input("Enter Garmin MFA code: ").strip()
    except EOFError:
        raise RuntimeError(
            "MFA required but no interactive input is available (e.g. running "
            "from the dashboard without a terminal). Run `python sync.py` "
            "directly in a terminal once to refresh the cached session."
        )


def get_client() -> Garmin:
    client = Garmin(
        email=os.environ["GARMIN_EMAIL"],
        password=os.environ["GARMIN_PASSWORD"],
        prompt_mfa=prompt_mfa_console,
    )
    client.login(str(TOKEN_STORE))  # resumes a cached session if valid, else does full auth
    return client


# --------------------------------------------------------------------------
# Fetch helpers
# --------------------------------------------------------------------------

def safe_call(label, fn, *args, **kwargs):
    """Calls fn and returns its result, or None (with a warning) if it fails.
    Keeps one bad metric from aborting the whole sync."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        log.warning("Failed to fetch %s: %s", label, e)
        return None


def first_present(d, *dotted_keys):
    """Returns the first non-None value found at any of the given dotted key
    paths in d. The unofficial Garmin API's field names shift between account
    types/firmware, so fetch_* functions probe a few known variants."""
    if not isinstance(d, dict):
        return None
    for key in dotted_keys:
        cur = d
        for part in key.split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
            if cur is None:
                break
        if cur is not None:
            return cur
    return None


def fetch_vo2max(client, date_str):
    raw = safe_call("vo2max", client.get_max_metrics, date_str)
    if not raw:
        return None
    entry = raw[0] if isinstance(raw, list) and raw else raw
    value = first_present(entry, "generic.vo2MaxPreciseValue", "generic.vo2MaxValue", "vo2MaxValue")
    return {"value": value} if value is not None else None


def fetch_race_predictions(client, date_str):
    raw = safe_call("race_predictions", client.get_race_predictions, date_str, date_str, "daily")
    if not raw:
        return None
    entry = raw[0] if isinstance(raw, list) and raw else raw
    if not isinstance(entry, dict):
        return None
    result = {
        "time_5k_sec": first_present(entry, "time5K", "raceTime5K"),
        "time_10k_sec": first_present(entry, "time10K", "raceTime10K"),
        "time_half_sec": first_present(entry, "timeHalfMarathon", "raceTimeHalfMarathon"),
        "time_marathon_sec": first_present(entry, "timeMarathon", "raceTimeMarathon"),
    }
    return result if any(v is not None for v in result.values()) else None


def fetch_hrv(client, date_str):
    raw = safe_call("hrv", client.get_hrv_data, date_str)
    if not isinstance(raw, dict):
        return None
    summary = raw.get("hrvSummary", raw)
    last_night = first_present(summary, "lastNightAvg")
    weekly = first_present(summary, "weeklyAvg")
    status = first_present(summary, "status")
    if last_night is None and weekly is None and status is None:
        return None
    return {"last_night_avg": last_night, "weekly_avg": weekly, "status": status}


def fetch_rhr(client, date_str):
    raw = safe_call("rhr", client.get_rhr_day, date_str)
    if not isinstance(raw, dict):
        return None
    value = raw.get("restingHeartRate")
    if value is None:
        try:
            value = raw["allMetrics"]["metricsMap"]["WELLNESS_RESTING_HEART_RATE"][0]["value"]
        except (KeyError, IndexError, TypeError):
            value = None
    return {"value": value} if value is not None else None


def fetch_sleep(client, date_str):
    raw = safe_call("sleep", client.get_sleep_data, date_str)
    if not isinstance(raw, dict):
        return None
    dto = raw.get("dailySleepDTO", raw)
    score = first_present(dto, "sleepScores.overall.value", "sleepScores.overallScore")
    seconds = first_present(dto, "sleepTimeSeconds")
    duration_hours = round(seconds / 3600, 2) if isinstance(seconds, (int, float)) else None
    if score is None and duration_hours is None:
        return None
    return {"score": score, "duration_hours": duration_hours}


def fetch_stress(client, date_str):
    raw = safe_call("stress", client.get_all_day_stress, date_str)
    if not isinstance(raw, dict):
        return None
    avg = first_present(raw, "avgStressLevel", "overallStressLevel")
    return {"avg_stress": avg} if avg is not None else None


def fetch_body_battery(client, date_str):
    raw = safe_call("body_battery", client.get_body_battery, date_str, date_str)
    if not isinstance(raw, list) or not raw:
        return None
    entry = raw[0]
    if not isinstance(entry, dict):
        return None
    charged, drained = entry.get("charged"), entry.get("drained")
    end_level = None
    values = entry.get("bodyBatteryValuesArray") or []
    if values and isinstance(values[-1], list) and len(values[-1]) >= 2:
        end_level = values[-1][1]
    if charged is None and drained is None and end_level is None:
        return None
    return {"charged": charged, "drained": drained, "end_level": end_level}


def sliding_window_best_time(distances_m, times_s, target_m):
    """Minimum elapsed time (seconds) to cover target_m anywhere in the
    activity, via a two-pointer scan over monotonic cumulative distance/time
    arrays. Returns None if the activity never covers that distance."""
    n = len(distances_m)
    if n == 0 or distances_m[-1] < target_m:
        return None
    best = None
    j = 0
    for i in range(n):
        if j < i:
            j = i
        while j < n and distances_m[j] - distances_m[i] < target_m:
            j += 1
        if j >= n:
            break
        duration = times_s[j] - times_s[i]
        if best is None or duration < best:
            best = duration
    return best


def fetch_activity_best_efforts(client, activity_id):
    """Scans the raw per-point activity stream (not Garmin's own lap
    boundaries, which are inconsistent/manual) for the fastest continuous
    1k/3k/5k/10k segment anywhere in the activity."""
    raw = safe_call("activity_details", client.get_activity_details, activity_id)
    if not isinstance(raw, dict):
        return None
    descriptors = raw.get("metricDescriptors") or []
    index_by_key = {d.get("key"): d.get("metricsIndex") for d in descriptors if isinstance(d, dict)}
    dist_idx, dur_idx = index_by_key.get("sumDistance"), index_by_key.get("sumDuration")
    if dist_idx is None or dur_idx is None:
        return None

    distances, times = [], []
    for point in raw.get("activityDetailMetrics") or []:
        metrics = point.get("metrics") if isinstance(point, dict) else None
        if not metrics or len(metrics) <= max(dist_idx, dur_idx):
            continue
        d, t = metrics[dist_idx], metrics[dur_idx]
        if isinstance(d, (int, float)) and isinstance(t, (int, float)):
            distances.append(d)
            times.append(t)
    if len(distances) < 2:
        return None

    result = {}
    for label, target_m in BEST_EFFORT_DISTANCES_M.items():
        best = sliding_window_best_time(distances, times, target_m)
        if best is not None:
            result[label] = round(best, 1)
    return result or None


def fetch_activity_weather(client, activity_id):
    raw = safe_call("activity_weather", client.get_activity_weather, activity_id)
    if not isinstance(raw, dict):
        return None
    temp_f = first_present(raw, "temp")
    humidity = first_present(raw, "relativeHumidity")
    if temp_f is None:
        return None
    return {"temp_c": round((temp_f - 32) * 5 / 9, 1), "humidity_pct": humidity}


def fetch_activity_hr_zones(client, activity_id):
    raw = safe_call("activity_hr_zones", client.get_activity_hr_in_timezones, activity_id)
    if not isinstance(raw, list) or not raw:
        return None
    zones = {}
    for z in raw:
        num, secs = z.get("zoneNumber"), z.get("secsInZone")
        if num is not None and secs is not None:
            zones[f"zone_{num}"] = round(secs, 1)
    return zones or None


def extract_activity(raw):
    if not isinstance(raw, dict):
        return None
    activity_type = ((raw.get("activityType") or {}).get("typeKey")) or ""
    if "running" not in activity_type:
        return None
    activity_id = raw.get("activityId")
    if activity_id is None:
        return None

    distance_m = raw.get("distance")
    duration_s = raw.get("duration")
    distance_km = round(distance_m / 1000, 2) if isinstance(distance_m, (int, float)) else None
    duration_min = round(duration_s / 60, 1) if isinstance(duration_s, (int, float)) else None

    avg_speed = raw.get("averageSpeed")  # meters/sec
    avg_pace = None
    if isinstance(avg_speed, (int, float)) and avg_speed > 0:
        avg_pace = round((1000 / avg_speed) / 60, 2)  # minutes per km

    return {
        "activity_id": activity_id,
        "date": (raw.get("startTimeLocal") or "")[:10],
        "start_time_local": raw.get("startTimeLocal"),
        "name": raw.get("activityName"),
        "activity_type": activity_type,
        "distance_km": distance_km,
        "duration_min": duration_min,
        "avg_pace_min_per_km": avg_pace,
        "avg_hr": raw.get("averageHR"),
        "max_hr": raw.get("maxHR"),
        "total_ascent_m": raw.get("elevationGain"),
        "cadence_spm": raw.get("averageRunningCadenceInStepsPerMinute"),
    }


# --------------------------------------------------------------------------
# Sync steps
# --------------------------------------------------------------------------

def compute_rolling_averages(data):
    """Recomputes trailing-7-day averages for rhr/sleep/stress across all
    stored days, so re-running (or backfilling) never drifts out of sync."""
    for date_str, entry in data["daily"].items():
        d = date.fromisoformat(date_str)
        window = [(d - timedelta(days=i)).isoformat() for i in range(7)]

        rhr_vals = [data["daily"][ds]["rhr"]["value"] for ds in window
                    if ds in data["daily"] and data["daily"][ds].get("rhr", {}).get("value") is not None]
        sleep_vals = [data["daily"][ds]["sleep"]["duration_hours"] for ds in window
                      if ds in data["daily"] and data["daily"][ds].get("sleep", {}).get("duration_hours") is not None]
        stress_vals = [data["daily"][ds]["stress"]["avg_stress"] for ds in window
                       if ds in data["daily"] and data["daily"][ds].get("stress", {}).get("avg_stress") is not None]

        if rhr_vals and "rhr" in entry:
            entry["rhr"]["avg_7day"] = round(sum(rhr_vals) / len(rhr_vals), 1)
        if sleep_vals and "sleep" in entry:
            entry["sleep"]["avg_7day_duration_hours"] = round(sum(sleep_vals) / len(sleep_vals), 2)
        if stress_vals and "stress" in entry:
            entry["stress"]["avg_7day"] = round(sum(stress_vals) / len(stress_vals), 1)


def sync_wellness(client, data, days):
    today = date.today()
    for offset in range(days):
        date_str = (today - timedelta(days=offset)).isoformat()
        entry = data["daily"].get(date_str, {})

        vo2max = fetch_vo2max(client, date_str)
        if vo2max:
            entry["vo2max"] = vo2max
        race = fetch_race_predictions(client, date_str)
        if race:
            entry["race_predictions"] = race
        hrv = fetch_hrv(client, date_str)
        if hrv:
            entry["hrv"] = hrv
        rhr = fetch_rhr(client, date_str)
        if rhr:
            entry["rhr"] = rhr
        sleep = fetch_sleep(client, date_str)
        if sleep:
            entry["sleep"] = sleep
        stress = fetch_stress(client, date_str)
        if stress:
            entry["stress"] = stress
        battery = fetch_body_battery(client, date_str)
        if battery:
            entry["body_battery"] = battery

        entry["synced_at"] = datetime.now().astimezone().isoformat()
        data["daily"][date_str] = entry
        time.sleep(0.4)  # conservative courtesy delay; Garmin documents no explicit rate limit

    compute_rolling_averages(data)


def sync_activities(client, data, lookback_days, enrich_days):
    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)
    enrich_cutoff = end_date - timedelta(days=enrich_days)
    raw_list = safe_call(
        "activities", client.get_activities_by_date, start_date.isoformat(), end_date.isoformat()
    ) or []
    for raw in raw_list:
        activity = extract_activity(raw)
        if activity is None:
            continue
        activity_id = activity["activity_id"]
        try:
            activity_date = date.fromisoformat(activity["date"])
        except (ValueError, TypeError):
            activity_date = None

        if activity_date and activity_date >= enrich_cutoff:
            best_efforts = fetch_activity_best_efforts(client, activity_id)
            if best_efforts:
                activity["best_efforts"] = best_efforts
            time.sleep(0.4)
            weather = fetch_activity_weather(client, activity_id)
            if weather:
                activity["weather"] = weather
            time.sleep(0.4)
            hr_zones = fetch_activity_hr_zones(client, activity_id)
            if hr_zones:
                activity["hr_zones"] = hr_zones
            time.sleep(0.4)

        data["activities"][str(activity_id)] = activity


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------

def load_history(path: Path) -> dict:
    if not path.exists():
        return {"daily": {}, "activities": {}, "meta": {"schema_version": 1}}
    return json.loads(path.read_text())


def save_history(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(path)  # atomic on the same filesystem


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def run_sync() -> dict:
    """Runs a full sync and returns the updated history dict. Safe to import
    and call from dashboard.py as well as running this file directly."""
    load_dotenv()
    client = get_client()
    data = load_history(DATA_FILE)

    is_first_run = len(data["daily"]) == 0
    sync_wellness(client, data, days=BACKFILL_DAYS if is_first_run else 1)

    activities_backfilled = data["meta"].get("activities_backfilled", False)
    activity_lookback = ACTIVITY_LOOKBACK_DAYS if activities_backfilled else ACTIVITY_BACKFILL_DAYS
    sync_activities(client, data, lookback_days=activity_lookback, enrich_days=DETAIL_ENRICHMENT_DAYS)
    if not activities_backfilled:
        data["meta"]["activities_backfilled"] = True

    data["meta"]["last_sync"] = datetime.now().astimezone().isoformat()
    save_history(DATA_FILE, data)
    return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_sync()
    log.info(
        "Sync complete: %d daily entries, %d activities",
        len(result["daily"]), len(result["activities"]),
    )
