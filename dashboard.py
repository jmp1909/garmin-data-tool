"""
Personal Garmin Connect running dashboard.

Reads data/garmin_history.json (auto-syncing it first if it's stale) and
renders trend charts. Which sections appear, your race date, weekly run
target, and run-type thresholds are all configurable from the sidebar
Settings panel -- your choices are saved to data/config.json.

Run with:

    streamlit run dashboard.py
"""

import json
from datetime import date

import altair as alt
import pandas as pd
import streamlit as st

from sync import BASE_DIR, DATA_FILE, run_sync

CONFIG_FILE = BASE_DIR / "data" / "config.json"

DEFAULT_WEEKLY_RUN_TARGET = 4
DEFAULT_LONG_RUN_DISTANCE_FACTOR = 1.3   # vs. trailing 8-week median distance
DEFAULT_TEMPO_PACE_PERCENTILE = 0.25     # faster than this percentile of trailing paces = tempo/speed

st.set_page_config(page_title="Running Dashboard", layout="wide")

# --------------------------------------------------------------------------
# Color palette (validated categorical/status/sequential slots, light+dark
# variants) -- see dataviz skill palette.md. Streamlit >=1.36 exposes the
# viewer's active theme via st.context.theme; fall back to light if absent.
# --------------------------------------------------------------------------

try:
    DARK = st.context.theme.type == "dark"
except Exception:
    DARK = False

CATEGORICAL = ["#3987e5", "#d95926", "#199e70"] if DARK else ["#2a78d6", "#eb6834", "#1baf7a"]
STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_CRITICAL = "#d03b3b"
AXIS_GRID = "#2c2c2a" if DARK else "#e1e0d9"
AXIS_DOMAIN = "#383835" if DARK else "#c3c2b7"
AXIS_LABEL = "#c3c2b7" if DARK else "#52514e"
SEQUENTIAL_BLUES = ["#104281", "#1c5cab", "#3987e5", "#86b6ef"] if DARK else ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"]


def marathon_theme():
    return {
        "config": {
            "background": None,
            "axis": {
                "gridColor": AXIS_GRID,
                "domainColor": AXIS_DOMAIN,
                "tickColor": AXIS_DOMAIN,
                "labelColor": AXIS_LABEL,
                "titleColor": AXIS_LABEL,
                "labelFontSize": 11,
                "titleFontSize": 12,
            },
            "legend": {"labelColor": AXIS_LABEL, "titleColor": AXIS_LABEL},
            "line": {"strokeWidth": 2},
            "circle": {"opacity": 0.75},
            "range": {"category": CATEGORICAL},
        }
    }


alt.themes.register("marathon", marathon_theme)
alt.themes.enable("marathon")


# --------------------------------------------------------------------------
# Section registry -- each widget registers itself here with a stable id,
# a sidebar group, and a display label. The same registry drives both the
# page rendering order and the Settings-panel checkboxes, so adding a new
# widget to the dashboard means writing one function and nothing else.
# --------------------------------------------------------------------------

SECTIONS = []
GROUP_ORDER = ["Overview", "Training Load", "Race Fitness", "Recovery", "Runs"]


def section(section_id, group, label, default_enabled=True):
    def decorator(fn):
        SECTIONS.append({"id": section_id, "group": group, "label": label, "default": default_enabled, "render": fn})
        return fn
    return decorator


# --------------------------------------------------------------------------
# Data loading + auto-sync
# --------------------------------------------------------------------------

def load_data_raw() -> dict:
    if not DATA_FILE.exists():
        return {"daily": {}, "activities": {}, "meta": {}}
    return json.loads(DATA_FILE.read_text())


def load_frames(raw: dict):
    daily = pd.json_normalize(list(raw["daily"].values()))
    if not daily.empty:
        daily["date"] = pd.to_datetime(list(raw["daily"].keys()))
        daily = daily.set_index("date").sort_index()

    acts = pd.DataFrame.from_dict(raw["activities"], orient="index")
    if not acts.empty:
        acts["date"] = pd.to_datetime(acts["date"])
        acts = acts.sort_values("date")
    return daily, acts


def infer_run_type(acts: pd.DataFrame, long_run_factor: float, tempo_percentile: float) -> pd.Series:
    recent = acts[acts["date"] >= acts["date"].max() - pd.Timedelta(weeks=8)]
    median_dist = recent["distance_km"].median()
    tempo_cutoff = recent["avg_pace_min_per_km"].quantile(tempo_percentile)

    def classify(row):
        if pd.notna(row["distance_km"]) and row["distance_km"] >= median_dist * long_run_factor:
            return "Long Run"
        if pd.notna(row["avg_pace_min_per_km"]) and row["avg_pace_min_per_km"] <= tempo_cutoff:
            return "Tempo/Speed"
        return "Easy/Recovery"

    return acts.apply(classify, axis=1)


def format_pace(min_per_km):
    if pd.isna(min_per_km):
        return "--"
    minutes = int(min_per_km)
    seconds = round((min_per_km - minutes) * 60)
    return f"{minutes}:{seconds:02d}/km"


def parse_hms(text: str):
    try:
        parts = [int(p) for p in text.strip().split(":")]
    except ValueError:
        return None
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    else:
        return None
    if h < 0 or not (0 <= m < 60) or not (0 <= s < 60):
        return None
    return h * 3600 + m * 60 + s


def seconds_to_hms(total_seconds):
    if not total_seconds:
        return ""
    total_seconds = int(round(total_seconds))
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def load_config() -> dict:
    if not CONFIG_FILE.exists():
        return {}
    try:
        return json.loads(CONFIG_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_config(cfg: dict):
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


def set_config_value(config: dict, key: str, value):
    """Persists a single config key if it changed. Returns the value."""
    if config.get(key) != value:
        config[key] = value
        save_config(config)
    return value


def training_phase(weeks_to_go):
    if weeks_to_go is None or weeks_to_go < 0:
        return "Race day!"
    if weeks_to_go <= 2:
        return "Taper"
    if weeks_to_go <= 6:
        return "Peak"
    if weeks_to_go <= 14:
        return "Build"
    return "Base"


def compute_readiness(daily: pd.DataFrame):
    """Rule-based good/warning/critical read combining HRV status, RHR trend,
    and sleep vs rolling average. Heuristic, not medical advice."""
    if daily.empty:
        return None
    latest = daily.iloc[-1]
    score = 0
    reasons = []

    hrv_status = latest.get("hrv.status")
    if hrv_status == "BALANCED":
        score += 1
    elif hrv_status in ("UNBALANCED", "LOW"):
        score -= 1
        reasons.append(f"HRV status: {hrv_status}")

    if "rhr.avg_7day" in daily.columns:
        rhr_series = daily["rhr.avg_7day"].dropna()
        if len(rhr_series) >= 8:
            recent_rhr, prior_rhr = rhr_series.iloc[-1], rhr_series.iloc[-8]
            if recent_rhr > prior_rhr + 2:
                score -= 1
                reasons.append("resting HR trending up")
            elif recent_rhr < prior_rhr - 1:
                score += 1

    sleep_last, sleep_avg = latest.get("sleep.duration_hours"), latest.get("sleep.avg_7day_duration_hours")
    if pd.notna(sleep_last) and pd.notna(sleep_avg):
        if sleep_last < sleep_avg - 1:
            score -= 1
            reasons.append("slept below your recent average")
        elif sleep_last >= sleep_avg:
            score += 1

    if score >= 1:
        return "good", "Ready to train", reasons
    if score <= -2:
        return "critical", "Signs of fatigue — consider an easy day", reasons
    return "warning", "Mixed signals — listen to your body", reasons


def compute_streak(weekly_counts: pd.Series, target: int) -> int:
    complete_weeks = weekly_counts.iloc[:-1] if len(weekly_counts) > 0 else weekly_counts
    streak = 0
    for count in reversed(complete_weeks.tolist()):
        if count >= target:
            streak += 1
        else:
            break
    return streak


def build_calendar_df(acts: pd.DataFrame, weeks: int = 13) -> pd.DataFrame:
    end = acts["date"].max().normalize()
    start = (end - pd.Timedelta(weeks=weeks - 1)).normalize()
    start = start - pd.Timedelta(days=start.weekday())  # snap back to Monday
    all_days = pd.date_range(start=start, end=end, freq="D")
    daily_km = acts.groupby(acts["date"].dt.normalize())["distance_km"].sum()
    cal = pd.DataFrame({"date": all_days})
    cal["distance_km"] = cal["date"].map(daily_km).fillna(0)
    cal["week"] = cal["date"] - pd.to_timedelta(cal["date"].dt.weekday, unit="D")
    cal["weekday"] = cal["date"].dt.strftime("%a")
    return cal


def themed_line_chart(df_wide: pd.DataFrame, rename: dict, y_title: str, colors=None):
    """Melts a wide date-indexed frame into long format and returns a themed
    Altair line chart, or None if no data is present for the given columns."""
    present = [c for c in rename if c in df_wide.columns]
    if not present:
        return None
    value_vars = [rename[c] for c in present]
    long_df = df_wide[present].reset_index().rename(columns={"date": "date", **rename})
    long_df = long_df.melt(id_vars="date", value_vars=value_vars, var_name="Series", value_name="value")
    long_df = long_df.dropna(subset=["value"])
    if long_df.empty:
        return None

    encode_kwargs = dict(
        x=alt.X("date:T", title=None),
        y=alt.Y("value:Q", title=y_title),
        tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("Series:N"), alt.Tooltip("value:Q", format=".1f")],
    )
    if len(value_vars) > 1:
        encode_kwargs["color"] = alt.Color("Series:N", title=None, scale=alt.Scale(domain=value_vars, range=colors or CATEGORICAL))
    else:
        encode_kwargs["color"] = alt.value((colors or CATEGORICAL)[0])
    return alt.Chart(long_df).mark_line(point=True).encode(**encode_kwargs)


# --------------------------------------------------------------------------
# Section renderers
# --------------------------------------------------------------------------

@section("quick_stats", "Overview", "Quick Stats")
def render_quick_stats(ctx):
    acts, weekly_periods, weekly_counts = ctx["acts"], ctx["weekly_periods"], ctx["weekly_counts"]
    stat_cols = st.columns(4)
    stat_cols[0].metric("Total distance", f"{acts['distance_km'].sum():.1f} km" if not acts.empty else "--")
    stat_cols[1].metric("This week", f"{weekly_periods.iloc[-1]:.1f} km" if not weekly_periods.empty else "--")
    long_run = ctx["long_run_this_week"]
    stat_cols[2].metric("Long run this week", f"{long_run:.1f} km" if long_run is not None else "--")
    streak = compute_streak(weekly_counts, ctx["weekly_run_target"]) if not weekly_counts.empty else None
    stat_cols[3].metric("Weekly streak", f"{streak} wk" if streak is not None else "--")


@section("readiness", "Overview", "Today's Readiness")
def render_readiness(ctx):
    st.subheader("Today's Readiness")
    readiness = compute_readiness(ctx["daily"])
    if readiness:
        level, headline, reasons = readiness
        text = headline + (" — " + "; ".join(reasons) if reasons else "")
        icon = {"good": "✅", "warning": "⚠️", "critical": "🔴"}[level]
        {"good": st.success, "warning": st.warning, "critical": st.error}[level](text, icon=icon)
    else:
        st.caption("Not enough data yet for a readiness read.")


@section("ramp_rate_warning", "Training Load", "Ramp-Rate Warning")
def render_ramp_rate_warning(ctx):
    weekly_periods = ctx["weekly_periods"]
    if len(weekly_periods) < 3:
        return
    last_complete, prior_complete = weekly_periods.iloc[-2], weekly_periods.iloc[-3]
    if prior_complete > 0:
        ramp_pct = (last_complete - prior_complete) / prior_complete * 100
        if ramp_pct > 10:
            st.warning(
                f"Mileage jumped {ramp_pct:.0f}% last week ({prior_complete:.1f} → {last_complete:.1f} km) "
                "— the classic overuse-injury zone is a >10% weekly increase.",
                icon="⚠️",
            )


@section("weekly_mileage", "Training Load", "Weekly Mileage")
def render_weekly_mileage(ctx):
    acts, weekly_periods = ctx["acts"], ctx["weekly_periods"]
    st.subheader("Weekly Mileage")
    if acts.empty:
        st.caption("No activities yet.")
        return
    weekly_recent = weekly_periods.tail(12)
    weekly_df = pd.DataFrame({
        "week_start": weekly_recent.index.start_time,
        "week_end": weekly_recent.index.end_time.normalize(),
        "distance_km": weekly_recent.values,
    })
    weekly_df["week_label"] = (
        weekly_df["week_start"].dt.strftime("%b %d") + " - " + weekly_df["week_end"].dt.strftime("%b %d")
    )
    weekly_chart = (
        alt.Chart(weekly_df)
        .mark_bar(color=CATEGORICAL[0], cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("week_start:T", title="Week"),
            y=alt.Y("distance_km:Q", title="km"),
            tooltip=[
                alt.Tooltip("week_label:N", title="Week"),
                alt.Tooltip("distance_km:Q", title="Distance (km)", format=".1f"),
            ],
        )
    )
    st.altair_chart(weekly_chart, width="stretch")
    st.caption(f"Current week ({weekly_df['week_label'].iloc[-1]}) is still in progress — its bar will keep growing.")


@section("monthly_mileage", "Training Load", "Monthly Mileage")
def render_monthly_mileage(ctx):
    acts, monthly_periods = ctx["acts"], ctx["monthly_periods"]
    st.subheader("Monthly Mileage")
    if acts.empty:
        st.caption("No activities yet.")
        return
    monthly_df = monthly_periods.reset_index()
    monthly_df.columns = ["month_period", "distance_km"]
    monthly_df["month"] = monthly_df["month_period"].apply(lambda p: p.start_time)
    monthly_df["month_label"] = monthly_df["month_period"].astype(str)
    monthly_chart = (
        alt.Chart(monthly_df)
        .mark_bar(color=CATEGORICAL[1], cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("month:T", title="Month"),
            y=alt.Y("distance_km:Q", title="km"),
            tooltip=[alt.Tooltip("month_label:N", title="Month"), alt.Tooltip("distance_km:Q", title="km", format=".1f")],
        )
    )
    st.altair_chart(monthly_chart, width="stretch")


@section("weekly_run_count", "Training Load", "Weekly Run Count")
def render_weekly_run_count(ctx):
    acts, weekly_counts, target = ctx["acts"], ctx["weekly_counts"], ctx["weekly_run_target"]
    st.subheader("Weekly Run Count")
    if acts.empty:
        st.caption("No activities yet.")
        return
    counts_df = weekly_counts.tail(12).reset_index()
    counts_df.columns = ["week_period", "runs"]
    counts_df["week"] = counts_df["week_period"].apply(lambda p: p.start_time)
    count_bars = (
        alt.Chart(counts_df)
        .mark_bar(color=CATEGORICAL[2], cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("week:T", title="Week"),
            y=alt.Y("runs:Q", title="Runs", axis=alt.Axis(tickMinStep=1, format="d")),
            tooltip=[alt.Tooltip("week:T", title="Week"), alt.Tooltip("runs:Q", title="Runs")],
        )
    )
    target_rule = (
        alt.Chart(pd.DataFrame({"target": [target]}))
        .mark_rule(strokeDash=[4, 4], color=STATUS_WARNING)
        .encode(y="target:Q")
    )
    st.altair_chart(count_bars + target_rule, width="stretch")
    st.caption(f"Current streak: {compute_streak(weekly_counts, target)} week(s) hitting {target}+ runs. Dashed line marks your target.")


@section("vo2max", "Race Fitness", "VO2max Trend")
def render_vo2max(ctx):
    st.subheader("VO2max Trend")
    chart = themed_line_chart(ctx["daily"], {"vo2max.value": "VO2max"}, "VO2max")
    if chart is not None:
        st.altair_chart(chart, width="stretch")
    else:
        st.caption("No VO2max data yet.")


@section("race_predictor_long", "Race Fitness", "Race Predictor: Marathon & Half Marathon")
def render_race_predictor_long(ctx):
    daily = ctx["daily"]
    st.subheader("Race Predictor: Marathon & Half Marathon")
    cols = ["race_predictions.time_marathon_sec", "race_predictions.time_half_sec"]
    race_wide = daily[cols] / 60 if all(c in daily.columns for c in cols) else pd.DataFrame()
    chart = themed_line_chart(
        race_wide, {"race_predictions.time_marathon_sec": "Marathon", "race_predictions.time_half_sec": "Half Marathon"},
        "Predicted time (min)",
    )
    if chart is None:
        st.caption("No race predictor data yet.")
        return
    layers = [chart]
    goal_sec = ctx["goal_marathon_sec"]
    if goal_sec:
        goal_minutes = goal_sec / 60
        layers.append(alt.Chart(pd.DataFrame({"minutes": [goal_minutes]})).mark_rule(strokeDash=[4, 4], color=STATUS_GOOD).encode(y="minutes:Q"))
        layers.append(
            alt.Chart(pd.DataFrame({"minutes": [goal_minutes], "date": [daily.index.max()]}))
            .mark_text(text=f"Goal: {seconds_to_hms(goal_sec)}", align="right", baseline="bottom", dy=-4, color=STATUS_GOOD)
            .encode(x="date:T", y="minutes:Q")
        )
    st.altair_chart(alt.layer(*layers), width="stretch")


@section("race_predictor_short", "Race Fitness", "Race Predictor: 5K & 10K")
def render_race_predictor_short(ctx):
    daily = ctx["daily"]
    st.subheader("Race Predictor: 5K & 10K")
    cols = ["race_predictions.time_5k_sec", "race_predictions.time_10k_sec"]
    race_wide = daily[cols] / 60 if all(c in daily.columns for c in cols) else pd.DataFrame()
    chart = themed_line_chart(race_wide, {"race_predictions.time_5k_sec": "5K", "race_predictions.time_10k_sec": "10K"}, "Predicted time (min)")
    if chart is not None:
        st.altair_chart(chart, width="stretch")
    else:
        st.caption("No race predictor data yet.")


@section("personal_records", "Race Fitness", "Personal Records")
def render_personal_records(ctx):
    acts, daily = ctx["acts"], ctx["daily"]
    st.subheader("Personal Records")
    if acts.empty or not acts["avg_pace_min_per_km"].notna().any():
        st.caption("No activities yet.")
        return
    fastest_idx = acts["avg_pace_min_per_km"].idxmin()
    longest_idx = acts["distance_km"].idxmax()
    pr_cols = st.columns(3)
    pr_cols[0].metric("Fastest Pace", format_pace(acts.loc[fastest_idx, "avg_pace_min_per_km"]))
    pr_cols[0].caption(acts.loc[fastest_idx, "date"].strftime("%b %d, %Y"))
    pr_cols[1].metric("Longest Run", f"{acts.loc[longest_idx, 'distance_km']:.1f} km")
    pr_cols[1].caption(acts.loc[longest_idx, "date"].strftime("%b %d, %Y"))
    if "vo2max.value" in daily.columns and daily["vo2max.value"].notna().any():
        vo2_idx = daily["vo2max.value"].idxmax()
        pr_cols[2].metric("Highest VO2max", f"{daily.loc[vo2_idx, 'vo2max.value']:.1f}")
        pr_cols[2].caption(vo2_idx.strftime("%b %d, %Y"))
    else:
        pr_cols[2].metric("Highest VO2max", "--")


@section("hrv_rhr", "Recovery", "HRV & Resting Heart Rate")
def render_hrv_rhr(ctx):
    daily = ctx["daily"]
    st.subheader("HRV & Resting Heart Rate (last 8 weeks)")
    chart = themed_line_chart(
        daily.tail(56), {"hrv.weekly_avg": "HRV (weekly avg, ms)", "rhr.avg_7day": "Resting HR (7-day avg, bpm)"}, None
    )
    if chart is None:
        st.caption("No HRV/RHR data yet.")
        return
    st.altair_chart(chart, width="stretch")
    if "hrv.status" in daily.columns:
        latest_status = daily["hrv.status"].dropna()
        if not latest_status.empty:
            st.metric("Latest HRV status", latest_status.iloc[-1])


@section("sleep", "Recovery", "Sleep Duration")
def render_sleep(ctx):
    st.subheader("Sleep Duration vs 7-Day Rolling Average")
    chart = themed_line_chart(
        ctx["daily"], {"sleep.duration_hours": "Nightly Duration (hrs)", "sleep.avg_7day_duration_hours": "7-Day Avg (hrs)"}, "hours"
    )
    if chart is not None:
        st.altair_chart(chart, width="stretch")
    else:
        st.caption("No sleep data yet.")


@section("stress", "Recovery", "Stress Trend")
def render_stress(ctx):
    st.subheader("Stress Trend")
    chart = themed_line_chart(ctx["daily"], {"stress.avg_stress": "Daily Stress", "stress.avg_7day": "7-Day Avg"}, "stress score")
    if chart is not None:
        st.altair_chart(chart, width="stretch")
    else:
        st.caption("No stress data yet.")


@section("recent_runs", "Runs", "Recent Runs")
def render_recent_runs(ctx):
    acts = ctx["acts"]
    st.subheader("Recent Runs")
    if acts.empty:
        st.caption("No activities yet.")
        return
    recent_runs = acts.sort_values("date", ascending=False).head(20).copy()
    recent_runs["pace"] = recent_runs["avg_pace_min_per_km"].apply(format_pace)
    display_cols = {"date": "Date", "run_type": "Type", "distance_km": "Distance (km)", "pace": "Pace", "avg_hr": "Avg HR"}
    st.dataframe(
        recent_runs[list(display_cols.keys())].rename(columns=display_cols),
        hide_index=True,
        column_config={"Date": st.column_config.DateColumn(format="YYYY-MM-DD")},
    )


@section("pace_hr_scatter", "Runs", "Pace vs Heart Rate")
def render_pace_hr_scatter(ctx):
    acts = ctx["acts"]
    st.subheader("Pace vs Heart Rate (Aerobic Decoupling)")
    st.caption("Same pace at a lower heart rate over time suggests improving aerobic fitness.")
    scatter_df = acts.dropna(subset=["avg_pace_min_per_km", "avg_hr"]) if not acts.empty else acts
    if scatter_df.empty:
        st.caption("Not enough activity data yet.")
        return
    run_type_domain = ["Easy/Recovery", "Long Run", "Tempo/Speed"]
    scatter_chart = (
        alt.Chart(scatter_df)
        .mark_circle(size=80)
        .encode(
            x=alt.X("avg_pace_min_per_km:Q", title="Avg Pace (min/km)", scale=alt.Scale(reverse=True)),
            y=alt.Y("avg_hr:Q", title="Avg HR (bpm)"),
            color=alt.Color("run_type:N", title="Run Type", scale=alt.Scale(domain=run_type_domain, range=CATEGORICAL)),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("distance_km:Q", title="Distance (km)"),
                alt.Tooltip("avg_pace_min_per_km:Q", title="Pace (min/km)"),
                alt.Tooltip("avg_hr:Q", title="Avg HR"),
                alt.Tooltip("run_type:N", title="Type"),
            ],
        )
        .interactive()
    )
    st.altair_chart(scatter_chart, width="stretch")


@section("calendar_heatmap", "Runs", "Run Consistency Calendar")
def render_calendar_heatmap(ctx):
    acts = ctx["acts"]
    st.subheader("Run Consistency Calendar")
    if acts.empty:
        st.caption("No activities yet.")
        return
    cal_df = build_calendar_df(acts, weeks=13)
    weekday_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    heatmap = (
        alt.Chart(cal_df)
        .mark_rect(cornerRadius=2)
        .encode(
            x=alt.X("week:T", title=None, axis=alt.Axis(format="%b %d")),
            y=alt.Y("weekday:O", sort=weekday_order, title=None),
            color=alt.Color("distance_km:Q", title="km", scale=alt.Scale(range=SEQUENTIAL_BLUES)),
            tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("distance_km:Q", title="km", format=".1f")],
        )
    )
    st.altair_chart(heatmap, width="stretch")


# --------------------------------------------------------------------------
# Sidebar: race date, training phase, goal time, settings, sync controls
# --------------------------------------------------------------------------

st.sidebar.header("Running Dashboard")
config = load_config()

race_date_val = date.fromisoformat(config["race_date"]) if config.get("race_date") else None
new_race_date = st.sidebar.date_input("Race date", value=race_date_val)
race_date = new_race_date
if race_date != race_date_val:
    set_config_value(config, "race_date", race_date.isoformat() if race_date else None)

days_to_go = (race_date - date.today()).days if race_date else None
st.sidebar.metric("Days to race", days_to_go if days_to_go is not None else "—")
st.sidebar.metric("Training phase", training_phase(days_to_go / 7) if days_to_go is not None else "—")

goal_input = st.sidebar.text_input("Goal time (H:MM:SS)", value=seconds_to_hms(config.get("goal_marathon_time_sec")), placeholder="3:30:00")
goal_marathon_sec = parse_hms(goal_input) if goal_input else None
if goal_input and goal_marathon_sec is None:
    st.sidebar.caption("⚠️ Couldn't parse that as H:MM:SS")
elif goal_marathon_sec != config.get("goal_marathon_time_sec"):
    set_config_value(config, "goal_marathon_time_sec", goal_marathon_sec)

with st.sidebar.expander("⚙️ Settings"):
    weekly_run_target = st.number_input(
        "Weekly run target", min_value=1, max_value=14, value=config.get("weekly_run_target", DEFAULT_WEEKLY_RUN_TARGET)
    )
    set_config_value(config, "weekly_run_target", weekly_run_target)

    st.caption("Advanced: run-type inference")
    long_run_factor = st.slider(
        "Long-run distance factor", 1.0, 2.0, value=config.get("long_run_distance_factor", DEFAULT_LONG_RUN_DISTANCE_FACTOR), step=0.05
    )
    set_config_value(config, "long_run_distance_factor", long_run_factor)
    tempo_percentile = st.slider(
        "Tempo/speed pace percentile", 0.05, 0.5, value=config.get("tempo_pace_percentile", DEFAULT_TEMPO_PACE_PERCENTILE), step=0.05
    )
    set_config_value(config, "tempo_pace_percentile", tempo_percentile)

    st.caption("Sections shown")
    enabled_sections = config.get("enabled_sections", {})
    for group in GROUP_ORDER:
        group_sections = [s for s in SECTIONS if s["group"] == group]
        if not group_sections:
            continue
        st.caption(f"**{group}**")
        for s in group_sections:
            current = enabled_sections.get(s["id"], s["default"])
            new_val = st.checkbox(s["label"], value=current, key=f"chk_{s['id']}")
            if new_val != current:
                enabled_sections[s["id"]] = new_val
    set_config_value(config, "enabled_sections", enabled_sections)

raw = load_data_raw()
last_sync_date = (raw.get("meta", {}).get("last_sync") or "")[:10]
needs_sync = last_sync_date != date.today().isoformat()
force_sync = st.sidebar.button("🔄 Sync now")

if needs_sync or force_sync:
    with st.spinner("Syncing with Garmin Connect..."):
        raw = run_sync()

if raw.get("meta", {}).get("last_sync"):
    st.sidebar.caption(f"Last synced: {raw['meta']['last_sync'][:16].replace('T', ' ')}")

st.title("🏃 Running Dashboard")

daily, acts = load_frames(raw)

if daily.empty and acts.empty:
    st.info("No data yet. Click **Sync now** in the sidebar to pull your first batch from Garmin Connect.")
    st.stop()

if not acts.empty:
    acts["run_type"] = infer_run_type(acts, long_run_factor, tempo_percentile)


# --------------------------------------------------------------------------
# Derived aggregates shared across sections
# --------------------------------------------------------------------------

if not acts.empty:
    weekly_periods = acts.groupby(acts["date"].dt.to_period("W"))["distance_km"].sum()
    weekly_counts = acts.groupby(acts["date"].dt.to_period("W"))["activity_id"].count()
    monthly_periods = acts.groupby(acts["date"].dt.to_period("M"))["distance_km"].sum()
    current_week_period = acts["date"].max().to_period("W")
    long_run_this_week = acts.loc[acts["date"].dt.to_period("W") == current_week_period, "distance_km"].max()
else:
    weekly_periods = pd.Series(dtype=float)
    weekly_counts = pd.Series(dtype="int64")
    monthly_periods = pd.Series(dtype=float)
    long_run_this_week = None

ctx = {
    "daily": daily,
    "acts": acts,
    "weekly_periods": weekly_periods,
    "weekly_counts": weekly_counts,
    "monthly_periods": monthly_periods,
    "long_run_this_week": long_run_this_week,
    "weekly_run_target": weekly_run_target,
    "goal_marathon_sec": goal_marathon_sec,
}


# --------------------------------------------------------------------------
# Render enabled sections, grouped
# --------------------------------------------------------------------------

for group in GROUP_ORDER:
    group_sections = [s for s in SECTIONS if s["group"] == group]
    visible = [s for s in group_sections if enabled_sections.get(s["id"], s["default"])]
    if not visible:
        continue
    if group != "Overview":
        st.header(group)
    for s in visible:
        s["render"](ctx)
    if group != "Overview":
        st.divider()
