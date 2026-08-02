"""Generates a fully synthetic garmin_history.json for the public demo.

This is NOT real data with the numbers scaled or offset — scaling a real
runner's data preserves their actual day-to-day training pattern (which days
they ran, the exact shape of their fitness curve), so it would still describe
a real person. Instead this simulates an independent fictional runner from a
parametric model with its own randomness, seeded for reproducibility.

The schema has no GPS/location/address fields at all (see sync.py) so that
specific risk class doesn't apply, but activity names are kept generic
("Running", Garmin's own default) rather than invented place names anyway.

Run:
    python scripts/generate_sample_data.py
Writes docs/sample_data/garmin_history.json.
"""

import json
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 20260802  # fixed for reproducibility, unrelated to any real date
random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_FILE = BASE_DIR / "docs" / "sample_data" / "garmin_history.json"

NUM_WEEKS = 16
TODAY = date(2026, 7, 15)  # fixed fictional "today" so output is deterministic
START = TODAY - timedelta(weeks=NUM_WEEKS)

RUNS_PER_WEEK = 4
DETAIL_ENRICHMENT_DAYS = 120  # matches sync.py default; all activities enriched here


def build_daily():
    daily = {}
    vo2max_start, vo2max_end = 41.5, 46.0
    hrv_base = 52
    rhr_start, rhr_end = 57, 50
    n_days = (TODAY - START).days + 1

    for i in range(n_days):
        d = START + timedelta(days=i)
        progress = i / max(n_days - 1, 1)

        vo2max = round(vo2max_start + (vo2max_end - vo2max_start) * progress + random.gauss(0, 0.3), 1)
        rhr_val = round(rhr_start + (rhr_end - rhr_start) * progress + random.gauss(0, 1.5))
        hrv_last_night = round(hrv_base + 8 * progress + random.gauss(0, 6))
        sleep_hours = round(max(5.0, min(9.0, random.gauss(7.1, 0.6))), 1)
        stress_avg = max(5, min(70, round(random.gauss(30 - 5 * progress, 8))))

        entry = {
            "vo2max": {"value": vo2max},
            "race_predictions": {
                "time_5k_sec": round((25.5 - 2.2 * progress) * 60 + random.gauss(0, 20)),
                "time_10k_sec": round((53.5 - 4.8 * progress) * 60 + random.gauss(0, 40)),
                "time_half_sec": round((119 - 10 * progress) * 60 + random.gauss(0, 90)),
                "time_marathon_sec": round((252 - 20 * progress) * 60 + random.gauss(0, 180)),
            },
            "hrv": {
                "last_night_avg": hrv_last_night,
                "weekly_avg": round(hrv_base + 8 * progress),
                "status": random.choices(
                    ["BALANCED", "UNBALANCED", "LOW"], weights=[0.7, 0.22, 0.08]
                )[0],
            },
            "rhr": {"value": rhr_val, "avg_7day": round(rhr_start + (rhr_end - rhr_start) * progress, 1)},
            "sleep": {
                "score": max(40, min(95, round(random.gauss(75, 10)))),
                "duration_hours": sleep_hours,
                "avg_7day_duration_hours": round(7.0 + 0.2 * progress, 1),
            },
            "stress": {"avg_stress": stress_avg, "avg_7day": round(30 - 5 * progress)},
            "synced_at": f"{d.isoformat()}T06:05:00-04:00",
        }
        daily[d.isoformat()] = entry
    return daily


RUN_TYPES = ["Easy/Recovery", "Easy/Recovery", "Tempo/Speed", "Long Run"]


def build_activities():
    activities = {}
    activity_id = 900000000001
    n_days = (TODAY - START).days + 1

    weekday_pool = list(range(7))
    for week in range(NUM_WEEKS):
        run_days = sorted(random.sample(weekday_pool, RUNS_PER_WEEK))
        week_start = START + timedelta(weeks=week)
        progress = week / max(NUM_WEEKS - 1, 1)
        base_weekly_km = 18 + 26 * progress  # build from ~18km/wk to ~44km/wk

        for slot, wd in enumerate(run_days):
            d = week_start + timedelta(days=wd)
            if d > TODAY:
                continue
            run_type = RUN_TYPES[slot % len(RUN_TYPES)]

            if run_type == "Long Run":
                distance_km = round(base_weekly_km * 0.32 + random.gauss(0, 1.0), 2)
                pace = round(6.4 - 0.5 * progress + random.gauss(0, 0.15), 2)
            elif run_type == "Tempo/Speed":
                distance_km = round(base_weekly_km * 0.18 + random.gauss(0, 0.6), 2)
                pace = round(5.1 - 0.5 * progress + random.gauss(0, 0.1), 2)
            else:
                distance_km = round(base_weekly_km * 0.22 + random.gauss(0, 0.8), 2)
                pace = round(6.0 - 0.4 * progress + random.gauss(0, 0.15), 2)

            distance_km = max(2.0, distance_km)
            pace = max(3.8, pace)
            duration_min = round(distance_km * pace, 1)
            avg_hr = {
                "Long Run": round(random.gauss(148, 6)),
                "Tempo/Speed": round(random.gauss(168, 6)),
                "Easy/Recovery": round(random.gauss(138, 6)),
            }[run_type]
            max_hr = avg_hr + round(random.gauss(18, 4))

            activity = {
                "activity_id": activity_id,
                "date": d.isoformat(),
                "start_time_local": f"{d.isoformat()}T06:30:00",
                "name": "Running",
                "activity_type": "running",
                "distance_km": round(distance_km, 2),
                "duration_min": duration_min,
                "avg_pace_min_per_km": pace,
                "avg_hr": avg_hr,
                "max_hr": max_hr,
                "total_ascent_m": max(0, round(random.gauss(45, 30))),
                "cadence_spm": round(random.gauss(170, 4)),
            }

            if (TODAY - d).days <= DETAIL_ENRICHMENT_DAYS:
                fast_factor = {"Long Run": 0.94, "Tempo/Speed": 0.90, "Easy/Recovery": 0.96}[run_type]
                best_efforts = {}
                for label, km in (("1k", 1), ("3k", 3), ("5k", 5), ("10k", 10)):
                    if km <= distance_km:
                        best_efforts[label] = round(km * pace * 60 * fast_factor, 1)
                if best_efforts:
                    activity["best_efforts"] = best_efforts

                activity["weather"] = {
                    "temp_c": round(random.gauss(17, 6), 1),
                    "humidity_pct": max(20, min(95, round(random.gauss(55, 15)))),
                }

                zone_split = {
                    "Long Run": [0.05, 0.35, 0.40, 0.15, 0.05],
                    "Tempo/Speed": [0.02, 0.10, 0.28, 0.40, 0.20],
                    "Easy/Recovery": [0.15, 0.55, 0.25, 0.04, 0.01],
                }[run_type]
                total_s = duration_min * 60
                activity["hr_zones"] = {
                    f"zone_{i+1}": round(total_s * frac, 1) for i, frac in enumerate(zone_split)
                }

            activities[str(activity_id)] = activity
            activity_id += 1

    return activities


def main():
    daily = build_daily()
    activities = build_activities()
    history = {
        "daily": daily,
        "activities": activities,
        "meta": {
            "schema_version": 1,
            "last_sync": f"{TODAY.isoformat()}T06:05:00-04:00",
            "note": "Synthetic sample data for the public demo — not a real athlete.",
        },
    }
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(history, indent=2))
    print(f"Wrote {len(daily)} daily entries and {len(activities)} activities to {OUT_FILE}")


if __name__ == "__main__":
    main()
