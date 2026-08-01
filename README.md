# Garmin Running Dashboard

A personal, local dashboard that pulls your running data from Garmin Connect
and tracks it over time — VO2max, race predictions, HRV/RHR/sleep/stress
recovery signals, mileage, run consistency, and pace-vs-HR trends — so you
don't have to check the Garmin Connect app manually.

Every widget is optional. Open the sidebar **⚙️ Settings** panel to pick which
sections show up, set your race date and goal time, tune the weekly run
target, and adjust the run-type detection thresholds — your choices are saved
locally and reloaded next time you open the dashboard.

## How it works

```
Garmin Connect  --sync.py-->  data/garmin_history.json  --dashboard.py-->  Streamlit UI
```

`sync.py` pulls data via the (unofficial) `python-garminconnect` library and
appends it to a local JSON file, keyed by date, so history accumulates rather
than being overwritten. Activities are deduplicated by Garmin's own activity
ID. `dashboard.py` reads that JSON and renders the charts — **opening the
dashboard automatically triggers a sync** if it hasn't synced yet today, so
there's no separate scheduled task to manage.

## Features

- **Overview** — quick stats (total distance, this week's mileage, longest
  run this week, weekly-target streak) and a rule-based daily readiness read
- **Training Load** — ramp-rate injury-risk warning, weekly/monthly mileage,
  weekly run count vs. your target
- **Race Fitness** — VO2max trend, marathon/half-marathon and 5K/10K race
  predictors (with an optional goal-time reference line), personal records
- **Recovery** — HRV & resting heart rate, sleep vs. rolling average, stress
- **Runs** — recent runs table, pace-vs-heart-rate scatter (aerobic
  decoupling), a GitHub-style run-consistency calendar

## Prerequisites

- Python 3.10+
- A Garmin Connect account with running activity history

## First-time setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and fill in your real Garmin credentials:

```
GARMIN_EMAIL=you@example.com
GARMIN_PASSWORD=your-garmin-password
```

Credentials are read from `.env` at runtime and never leave your machine or
get hardcoded anywhere. `.env` is gitignored.

## First launch

```bash
streamlit run dashboard.py
```

The first launch triggers a **~12-week backfill** sync automatically (so the
dashboard has useful trend data right away instead of slowly filling in over
the next 12 weeks) — this can take a little while and may prompt you for an
MFA code **in the terminal** if your Garmin account has two-factor auth
enabled. After that, a `.garmin_tokens/` folder is created holding your
cached session, so future syncs won't need your password or MFA code again
until that session expires.

Set your race date, goal time, and which sections you want in the sidebar
**⚙️ Settings** panel — none of it is hardcoded.

## Day-to-day use

Just open the dashboard:

```bash
streamlit run dashboard.py
```

It automatically re-syncs once per calendar day. To force a fresh pull
mid-session (e.g. right after finishing a run), click **🔄 Sync now** in the
sidebar.

On Windows, `run_dashboard.bat` does the same thing and can be turned into a
desktop shortcut for one-click launching.

## Running sync.py standalone

To pull fresh data without opening the dashboard:

```bash
python sync.py
```

## Customizing your dashboard

Everything below is set from the sidebar **⚙️ Settings** panel (no code
editing required) and persisted to `data/config.json`, which is gitignored —
your personal preferences never get committed:

- **Race date** and **goal time** — drive the countdown, training-phase
  label, and the goal reference line on the race predictor chart
- **Weekly run target** — used by the streak counter and the weekly-run-count
  chart's target line
- **Run-type thresholds** (long-run distance factor, tempo pace percentile) —
  tune how "Long Run" / "Tempo/Speed" / "Easy/Recovery" get inferred from
  your own recent pace and distance distribution, since Garmin activity
  titles are often just "Running"
- **Sections shown** — a checkbox per widget, grouped to match the page

### Adding your own widget

Each section in `dashboard.py` is a small function registered with a
decorator:

```python
@section("my_widget", "Training Load", "My Widget")
def render_my_widget(ctx):
    st.subheader("My Widget")
    ...
```

That's the entire integration point — the same registry drives both the page
layout and the Settings-panel checkboxes, so a new widget shows up in both
automatically. `ctx` carries the loaded `daily`/`acts` DataFrames and the
current config values; see the existing sections for examples.

## Data & privacy

These are gitignored and never committed:

- `.env` — your Garmin credentials
- `.garmin_tokens/` — your cached Garmin session (as sensitive as a password)
- `data/garmin_history.json` — your personal training history
- `data/config.json` — your personal settings (race date, goal time, etc.)

## Troubleshooting

- **Login/MFA errors when run from the dashboard**: MFA codes can only be
  entered in an interactive terminal. Run `python sync.py` directly in a
  terminal once to refresh the session, then the dashboard can reuse it.
- **Force a clean re-login**: delete the `.garmin_tokens/` folder and sync
  again.
- **One metric is missing for some days**: `sync.py` logs a warning and
  continues if a single metric fails to fetch or parse (the Garmin API is
  unofficial and its field names occasionally shift) — it won't halt the
  whole sync.

## Advanced configuration

Via `.env`:

- `GARMIN_BACKFILL_DAYS` (default 84) — how far back the very first sync
  reaches.
- `GARMIN_ACTIVITY_LOOKBACK_DAYS` (default 10) — activities are always
  re-fetched over this trailing window so a missed sync day self-heals.

## License

MIT — see [LICENSE](LICENSE).
