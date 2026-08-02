# Garmin Running Dashboard

**[Live demo →](https://jmp1909.github.io/garmin-data-tool/)** (sample data, not a real athlete's activity)

A personal, local dashboard that pulls your running data from Garmin Connect
and turns it into actual training analysis — not just a re-skin of what
Garmin Connect already shows you. It computes fitness/fatigue/form, injury-risk
load ratios, heat-adjusted pace, and a from-scratch race predictor from your
own best efforts, none of which Garmin exposes directly.

Every widget is optional. Open the sidebar **Settings** panel to pick which
sections show up, set your race date and goal time, tune the weekly run
target, and adjust the run-type detection thresholds — your choices are saved
locally and reloaded next time you open the dashboard. The status chips above
the tabs are contextual to whichever tab you're on.

## How it works

```
Garmin Connect --sync.py--> data/garmin_history.json --analytics.py--> dashboard.py --> Streamlit UI
```

`sync.py` pulls data via the (unofficial) `python-garminconnect` library and
appends it to a local JSON file, keyed by date, so history accumulates rather
than being overwritten. Activities are deduplicated by Garmin's own activity
ID. `analytics.py` is a pure, unit-tested module of training-load models
computed from that history. `dashboard.py` reads both and renders the UI —
**opening the dashboard automatically triggers a sync** if it hasn't synced
yet today, so there's no separate scheduled task to manage.

## Features

**Load** — fitness/fatigue/form (the standard Performance Manager Chart
model), acute:chronic workload ratio with its injury-risk band, weekly/monthly
mileage, weekly run count vs. your target.

**Fitness** — VO2max trend, marathon/half-marathon and 5K/10K race predictors
(with an optional goal-time reference line), a race-predictor comparison
across three independent methods (Riegel and Cameron from your own best
efforts, Daniels/Gilbert VDOT from Garmin's measured VO2max) shown next to
Garmin's own estimate, heat-adjusted pace, efficiency factor, personal
records.

**Best Efforts** — fastest *continuous* 1K/3K/5K/10K segment found anywhere
inside any run (not Garmin's own lap markers, which are inconsistent between
manual and auto laps) — this week vs. all-time, plus a leaderboard.

**Recovery** — HRV & resting heart rate, sleep vs. rolling average, stress,
and a weekly training polarization chart (the 80/20 easy/hard split).

**Runs** — recent runs table, pace-vs-heart-rate scatter (aerobic
decoupling).

Every chart with a line or area gets a crosshair + hover tooltip showing
every series' value at that point together, not just the date. Metrics with
non-obvious methodology (PMC, ACWR, heat adjustment, Riegel, Cameron,
Daniels/VDOT) have an (ℹ️) next to the title explaining exactly how they're
computed — see also the docstring at the top of `analytics.py` for the full
methodology notes, including the simplifications each model makes.

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

The first launch triggers two one-time deep pulls: a ~12-week wellness
backfill and a (separately gated) activity history backfill bounded by
`GARMIN_ACTIVITY_BACKFILL_DAYS` (default 365, naturally capped by however
much history your account actually has) — the fitness/fatigue/form model
needs several weeks of load history to mean anything, and the best-efforts
leaderboard needs your real activity history, not just the last 10 days.
Recent activities (within `GARMIN_DETAIL_ENRICHMENT_DAYS`, default 120) get
three extra per-activity calls each (best efforts, weather, HR zones), so
this first sync can take a few minutes — it may also prompt for an MFA code
**in the terminal** if your Garmin account has two-factor auth enabled. After
that, a `.garmin_tokens/` folder is created holding your cached session, so
future syncs won't need your password or MFA code again until it expires.

Set your race date, goal time, and which sections you want in the sidebar
**Settings** panel — none of it is hardcoded.

## Day-to-day use

Just open the dashboard:

```bash
streamlit run dashboard.py
```

It automatically re-syncs once per calendar day. To force a fresh pull
mid-session (e.g. right after finishing a run), click **Sync now** in the
sidebar.

On Windows, `run_dashboard.bat` does the same thing and can be turned into a
desktop shortcut for one-click launching.

## Running sync.py standalone

To pull fresh data without opening the dashboard:

```bash
python sync.py
```

## Customizing your dashboard

Everything below is set from the sidebar **Settings** panel (no code
editing required) and persisted to `data/config.json`, which is gitignored —
your personal preferences never get committed:

- **Race date** and **goal time** — drive the countdown, training-phase rail,
  and the goal reference line on the race predictor chart
- **Training block length** — how many weeks back the Base/Build/Peak/Taper
  rail starts, counted from race day
- **Weekly run target** — used by the streak counter and the weekly-run-count
  chart's target line
- **Run-type thresholds** (long-run distance factor, tempo pace percentile) —
  tune how "Long Run" / "Tempo/Speed" / "Easy/Recovery" get inferred from
  your own recent pace and distance distribution, since Garmin activity
  titles are often just "Running"
- **Sections shown** — a checkbox per widget, grouped by tab

### Adding your own widget

Each section in `dashboard.py` is a small function registered with a
decorator:

```python
@section("my_widget", "Load", "My Widget")
def render_my_widget(ctx):
    st.subheader("My Widget")
    ...
```

That's the entire integration point — the same registry drives the tab
layout and the Settings-panel checkboxes, so a new widget shows up in both
automatically. `ctx` carries the loaded `daily`/`acts` DataFrames, the
computed analytics (`pmc`, `acwr`, `heat`, `ef`, `weekly_polarization`,
`riegel_predictions`, `cameron_predictions`, `daniels_predictions`), and the
current config values; see the existing sections for examples. The five tabs
are `Load`, `Fitness`, `Best Efforts`, `Recovery`, `Runs`.

### Adding your own analytics model

`analytics.py` is plain pandas functions with no Streamlit or Garmin
dependency, so it's straightforward to extend and test in isolation — see
`test_analytics.py` for the pattern (hand-computed fixtures, not just
plausible-looking output).

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
- **A chart looks empty right after enabling it**: some metrics (PMC, best
  efforts, heat-adjusted pace, polarization) only have data for activities
  that were "enriched" during sync — see `GARMIN_DETAIL_ENRICHMENT_DAYS`
  below.

## Advanced configuration

Via `.env`:

- `GARMIN_BACKFILL_DAYS` (default 84) — how far back the first wellness sync
  reaches.
- `GARMIN_ACTIVITY_LOOKBACK_DAYS` (default 10) — activities are always
  re-fetched over this trailing window on every sync, so a missed day
  self-heals.
- `GARMIN_ACTIVITY_BACKFILL_DAYS` (default 365) — how far back the one-time
  activity history backfill reaches (bounded by your account's actual
  history).
- `GARMIN_DETAIL_ENRICHMENT_DAYS` (default 120) — activities within this many
  days of today get three extra per-activity API calls (best efforts,
  weather, HR zones) on every sync; older activities keep their basic
  distance/pace/HR but won't feed the best-efforts leaderboard, heat
  adjustment, or polarization chart. Raise this if you want deeper history
  for those specific features, at the cost of a slower sync.

## Development

```bash
pip install -r requirements-dev.txt
pytest
```

`test_sync.py` and `test_analytics.py` check the pure algorithmic pieces
(the best-effort sliding-window scan, PMC/ACWR/Riegel math, leaderboard
sorting) against hand-computed values — no Garmin account or network access
needed to run them.

## License

MIT — see [LICENSE](LICENSE).
