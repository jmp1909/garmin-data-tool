"""
Personal Garmin Connect running dashboard.

Reads data/garmin_history.json (auto-syncing it first if it's stale) and
renders training-load, fitness, recovery, and best-effort analytics. Race
date, goal time, weekly run target, run-type thresholds, and which widgets
appear are all configurable from the sidebar Settings panel and persisted to
data/config.json.

Run with:

    streamlit run dashboard.py
"""

import json
from datetime import date, timedelta

import altair as alt
import pandas as pd
import streamlit as st

import analytics as A
from sync import BASE_DIR, DATA_FILE, run_sync

CONFIG_FILE = BASE_DIR / "data" / "config.json"

DEFAULT_WEEKLY_RUN_TARGET = 4
DEFAULT_LONG_RUN_DISTANCE_FACTOR = 1.3
DEFAULT_TEMPO_PACE_PERCENTILE = 0.25
DEFAULT_PHASE_LOOKBACK_WEEKS = 20  # left edge of the training-phase rail, counted back from race day

# Phase boundaries, days before race day (must agree with training_phase())
PHASE_TAPER_DAYS = 14
PHASE_PEAK_DAYS = 42
PHASE_BUILD_DAYS = 98

BEST_EFFORT_LABELS = {"1k": "1 km", "3k": "3 km", "5k": "5 km", "10k": "10 km"}

st.set_page_config(page_title="Running Dashboard", layout="wide")

# --------------------------------------------------------------------------
# Palette -- validated categorical/status/sequential slots, light+dark
# variants (dataviz skill palette.md). "The blue line" is the through-motif:
# the phase rail, the primary chart series, and the active-tab marker.
# --------------------------------------------------------------------------

try:
    DARK = st.context.theme.type == "dark"
except Exception:
    DARK = False

ACCENT = "#5b8def" if DARK else "#1e48c8"
CATEGORICAL = ["#3987e5", "#d95926", "#199e70"] if DARK else ["#2a78d6", "#eb6834", "#1baf7a"]
STATUS_GOOD = "#3fae63" if DARK else "#157f3d"
STATUS_WARNING = "#d99527" if DARK else "#b45309"
STATUS_CRITICAL = "#e06a5e" if DARK else "#b42318"
SEQUENTIAL_BLUES = ["#104281", "#1c5cab", "#3987e5", "#86b6ef"] if DARK else ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"]

INK = "#ecefef" if DARK else "#14181a"
INK_2 = "#a3adb1" if DARK else "#5a6468"
INK_3 = "#79848a" if DARK else "#8b9599"
SURFACE = "#171b1e" if DARK else "#ffffff"
GROUND = "#0e1113" if DARK else "#f6f7f6"
RULE = "#262c30" if DARK else "#e2e6e5"
RULE_STRONG = "#333b40" if DARK else "#cfd5d4"
ACCENT_SOFT = "rgba(91,141,239,0.12)" if DARK else "rgba(30,72,200,0.08)"

MONO = 'ui-monospace, "SF Mono", "Cascadia Mono", "Roboto Mono", Menlo, Consolas, monospace'
SEV_COLOR = {"good": STATUS_GOOD, "warning": STATUS_WARNING, "critical": STATUS_CRITICAL, "accent": ACCENT, None: INK_3}


def marathon_theme():
    return {
        "config": {
            "background": None,
            "axis": {
                "gridColor": RULE, "domainColor": RULE_STRONG, "tickColor": RULE_STRONG,
                "labelColor": INK_2, "titleColor": INK_2, "labelFontSize": 11, "titleFontSize": 12,
                "labelFont": MONO, "titleFont": MONO,
            },
            "legend": {"labelColor": INK_2, "titleColor": INK_2, "labelFont": MONO},
            "line": {"strokeWidth": 2},
            "circle": {"opacity": 0.85},
            "range": {"category": CATEGORICAL},
        }
    }


alt.themes.register("marathon", marathon_theme)
alt.themes.enable("marathon")

st.markdown(f"""
<style>
  .stApp {{ background: {GROUND}; }}
  [data-testid="stMetricValue"], .rd-num {{ font-family: {MONO}; font-variant-numeric: tabular-nums; }}
  .rd-clock {{
    background: {SURFACE}; border: 1px solid {RULE}; border-radius: 4px;
    padding: 22px 24px 18px; margin-bottom: 18px;
  }}
  .rd-clock-top {{ display: flex; align-items: flex-end; justify-content: space-between; gap: 20px; flex-wrap: wrap; }}
  .rd-eyebrow {{ font-family: {MONO}; font-size: 11px; letter-spacing: .14em; text-transform: uppercase; color: {INK_3}; display: block; margin-bottom: 4px; }}
  .rd-days {{ font-family: {MONO}; font-size: clamp(40px,7vw,64px); font-weight: 600; line-height: .9; letter-spacing: -.02em; color: {INK}; }}
  .rd-days sup {{ font-size: .28em; font-weight: 500; letter-spacing: .12em; text-transform: uppercase; color: {INK_3}; margin-left: 8px; }}
  .rd-target {{ text-align: right; }}
  .rd-target .name {{ font-size: 14px; font-weight: 600; color: {INK}; }}
  .rd-target .sub {{ font-family: {MONO}; font-size: 12px; color: {INK_2}; }}
  .rd-target .goal {{ font-family: {MONO}; font-size: 12px; color: {ACCENT}; }}
  .rd-rail-track {{ position: relative; height: 3px; background: {RULE}; border-radius: 2px; margin-top: 18px; }}
  .rd-rail-fill {{ position: absolute; inset: 0 auto 0 0; background: {ACCENT}; border-radius: 2px; }}
  .rd-rail-now {{ position: absolute; top: 50%; width: 10px; height: 10px; border-radius: 50%; background: {ACCENT}; border: 2px solid {SURFACE}; transform: translate(-50%,-50%); }}
  .rd-rail-mark {{ position: absolute; top: -3px; width: 1px; height: 9px; background: {RULE_STRONG}; }}
  .rd-rail-labels {{ display: flex; justify-content: space-between; font-family: {MONO}; font-size: 10px; letter-spacing: .1em; text-transform: uppercase; color: {INK_3}; margin-top: 7px; }}
  .rd-rail-labels span.on {{ color: {ACCENT}; font-weight: 600; }}
  .rd-chips {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px,1fr)); gap: 10px; margin-bottom: 6px; }}
  .rd-chip {{ background: {SURFACE}; border: 1px solid {RULE}; border-radius: 4px; padding: 12px 14px; position: relative; overflow: hidden; }}
  .rd-chip::before {{ content: ""; position: absolute; inset: 0 auto 0 0; width: 2px; background: var(--sev); }}
  .rd-chip .k {{ font-family: {MONO}; font-size: 10px; letter-spacing: .12em; text-transform: uppercase; color: {INK_3}; display: block; }}
  .rd-chip .v {{ font-family: {MONO}; font-size: 22px; font-weight: 600; color: {INK}; }}
  .rd-chip .v small {{ font-size: .5em; font-weight: 400; color: {INK_3}; margin-left: 3px; }}
  .rd-chip .s {{ font-size: 11.5px; color: {INK_2}; display: flex; align-items: center; gap: 5px; margin-top: 2px; }}
  .rd-dot {{ width: 6px; height: 6px; border-radius: 50%; background: var(--sev); flex: none; }}
  .rd-hint {{ font-size: 12px; color: {INK_2}; max-width: 72ch; margin: -6px 0 4px; }}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Section registry -- each widget registers with a stable id, a tab, and a
# label. The same registry drives page layout and Settings checkboxes.
# --------------------------------------------------------------------------

SECTIONS = []
TAB_ORDER = ["Load", "Fitness", "Best Efforts", "Recovery", "Runs"]


def section(section_id, tab, label, default_enabled=True):
    def decorator(fn):
        SECTIONS.append({"id": section_id, "tab": tab, "label": label, "default": default_enabled, "render": fn})
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


def format_duration(total_seconds):
    if total_seconds is None or pd.isna(total_seconds):
        return "--"
    total_seconds = int(round(total_seconds))
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


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
    if config.get(key) != value:
        config[key] = value
        save_config(config)
    return value


def training_phase(days_to_go):
    if days_to_go is None or days_to_go < 0:
        return "Race day!"
    if days_to_go <= PHASE_TAPER_DAYS:
        return "Taper"
    if days_to_go <= PHASE_PEAK_DAYS:
        return "Peak"
    if days_to_go <= PHASE_BUILD_DAYS:
        return "Build"
    return "Base"


def compute_readiness(daily: pd.DataFrame):
    """Rule-based good/warning/critical read combining HRV status, RHR trend,
    and sleep vs rolling average. Heuristic, not medical advice."""
    if daily.empty:
        return None
    latest = daily.iloc[-1]
    score, reasons = 0, []

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
    start = start - pd.Timedelta(days=start.weekday())
    all_days = pd.date_range(start=start, end=end, freq="D")
    daily_km = acts.groupby(acts["date"].dt.normalize())["distance_km"].sum()
    cal = pd.DataFrame({"date": all_days})
    cal["distance_km"] = cal["date"].map(daily_km).fillna(0)
    cal["week"] = cal["date"] - pd.to_timedelta(cal["date"].dt.weekday, unit="D")
    cal["weekday"] = cal["date"].dt.strftime("%a")
    return cal


def chip(key, value_html, status_html, sev=None):
    color = SEV_COLOR.get(sev, INK_3)
    return f"""<div class="rd-chip" style="--sev:{color}">
      <span class="k">{key}</span>
      <span class="v">{value_html}</span>
      <span class="s"><span class="rd-dot"></span>{status_html}</span>
    </div>"""


def info_popover(body_markdown: str, label: str = "ℹ️"):
    with st.popover(label, use_container_width=False):
        st.markdown(body_markdown)


def crosshair_chart(long_df, y_title=None, colors=None, zero_line=False):
    """A themed multi-series line chart with a crosshair + hover tooltip
    (nearest-point rule, standard Vega-Lite pattern) -- expects columns
    date/Series/value. Returns None if there's no data to show."""
    if long_df is None or long_df.empty:
        return None
    series = sorted(long_df["Series"].unique().tolist())
    palette = colors or CATEGORICAL

    base = alt.Chart(long_df)
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["date"], empty=False)

    color_enc = alt.Color("Series:N", title=None, scale=alt.Scale(domain=series, range=palette[:len(series)])) \
        if len(series) > 1 else alt.value(palette[0])

    layers = []
    if zero_line:
        layers.append(alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=RULE_STRONG, strokeDash=[3, 3]).encode(y="y:Q"))

    line = base.mark_line(point=False).encode(x=alt.X("date:T", title=None), y=alt.Y("value:Q", title=y_title), color=color_enc)
    selectors = base.mark_point(opacity=0).encode(x="date:T").add_params(nearest)
    points = line.mark_point(size=45).encode(opacity=alt.condition(nearest, alt.value(1), alt.value(0)))
    tooltip = [alt.Tooltip("date:T", title="Date")] + ([alt.Tooltip("Series:N")] if len(series) > 1 else []) + [alt.Tooltip("value:Q", format=".2f")]
    rule = base.mark_rule(color=RULE_STRONG).encode(
        x="date:T", opacity=alt.condition(nearest, alt.value(0.5), alt.value(0)), tooltip=tooltip
    ).add_params(nearest)

    layers += [line, selectors, points, rule]
    return alt.layer(*layers).properties(height=280)


def melt_for_chart(df_wide: pd.DataFrame, rename: dict) -> pd.DataFrame:
    present = [c for c in rename if c in df_wide.columns]
    if not present:
        return pd.DataFrame(columns=["date", "Series", "value"])
    value_vars = [rename[c] for c in present]
    long_df = df_wide[present].reset_index().rename(columns={df_wide.index.name or "index": "date", **rename})
    long_df = long_df.melt(id_vars="date", value_vars=value_vars, var_name="Series", value_name="value")
    return long_df.dropna(subset=["value"])


# --------------------------------------------------------------------------
# Load tab
# --------------------------------------------------------------------------

@section("pmc", "Load", "Fitness, Fatigue & Form")
def render_pmc(ctx):
    st.subheader("Fitness, Fatigue & Form")
    info_popover(
        "**CTL** (fitness) is a 42-day exponentially-weighted average of daily training load. "
        "**ATL** (fatigue) is the same over 7 days. **TSB** (form) = CTL − ATL. Positive form means "
        "rested; sustained negative form means you're digging into fatigue. This is the standard "
        "Performance Manager Chart model — Garmin doesn't expose it."
    )
    pmc = ctx["pmc"]
    if pmc.empty:
        st.caption("Not enough training history yet.")
        return
    long_df = pmc.reset_index().rename(columns={"index": "date"}).melt(
        id_vars="date", value_vars=["ctl", "atl", "tsb"], var_name="Series", value_name="value"
    )
    long_df["Series"] = long_df["Series"].map({"ctl": "Fitness (CTL)", "atl": "Fatigue (ATL)", "tsb": "Form (TSB)"})
    chart = crosshair_chart(long_df, colors=[CATEGORICAL[0], INK_3, STATUS_GOOD], zero_line=True)
    if chart is not None:
        st.altair_chart(chart, width="stretch")
    latest_tsb = pmc["tsb"].iloc[-1]
    st.caption(f"Current form: {latest_tsb:+.1f}")


@section("acwr", "Load", "Load Ratio (ACWR)")
def render_acwr(ctx):
    st.subheader("Load Ratio (ACWR)")
    info_popover(
        "Acute (7-day average) load ÷ chronic (28-day average) load — the standard sports-science "
        "injury-risk heuristic. **0.8–1.3 is the published sweet spot**; sustained time above 1.5 is "
        "where injury risk climbs fastest."
    )
    acwr = ctx["acwr"]
    if acwr.empty:
        st.caption("Not enough training history yet.")
        return
    long_df = acwr.reset_index().rename(columns={"index": "date", "acwr": "value"})
    long_df["Series"] = "ACWR"
    chart = crosshair_chart(long_df, colors=[ACCENT])
    if chart is not None:
        band = alt.Chart(pd.DataFrame({"lo": [0.8], "hi": [1.3]})).mark_rect(opacity=0.08, color=STATUS_GOOD).encode(y="lo:Q", y2="hi:Q")
        ceiling = alt.Chart(pd.DataFrame({"y": [1.3]})).mark_rule(strokeDash=[4, 3], color=STATUS_WARNING).encode(y="y:Q")
        st.altair_chart(band + ceiling + chart, width="stretch")
    latest = acwr.iloc[-1]
    sev = A.classify_acwr(latest)
    label = {"good": "in the sweet spot", "warning": "outside the sweet spot", "critical": "high injury-risk zone"}.get(sev, "")
    getattr(st, {"good": "success", "warning": "warning", "critical": "error"}.get(sev, "info"))(
        f"Current ratio {latest:.2f} — {label}.", icon={"good": "✅", "warning": "⚠️", "critical": "🔴"}.get(sev, "ℹ️")
    )


@section("weekly_mileage", "Load", "Weekly Mileage")
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
    weekly_df["week_label"] = weekly_df["week_start"].dt.strftime("%b %d") + " - " + weekly_df["week_end"].dt.strftime("%b %d")
    chart = (
        alt.Chart(weekly_df).mark_bar(color=CATEGORICAL[0], cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(x=alt.X("week_start:T", title="Week"), y=alt.Y("distance_km:Q", title="km"),
                tooltip=[alt.Tooltip("week_label:N", title="Week"), alt.Tooltip("distance_km:Q", title="Distance (km)", format=".1f")])
    )
    st.altair_chart(chart, width="stretch")
    st.caption(f"Current week ({weekly_df['week_label'].iloc[-1]}) is still in progress — its bar will keep growing.")


@section("monthly_mileage", "Load", "Monthly Mileage")
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
    chart = (
        alt.Chart(monthly_df).mark_bar(color=CATEGORICAL[1], cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(x=alt.X("month:T", title="Month"), y=alt.Y("distance_km:Q", title="km"),
                tooltip=[alt.Tooltip("month_label:N", title="Month"), alt.Tooltip("distance_km:Q", title="km", format=".1f")])
    )
    st.altair_chart(chart, width="stretch")


@section("weekly_run_count", "Load", "Weekly Run Count")
def render_weekly_run_count(ctx):
    acts, weekly_counts, target = ctx["acts"], ctx["weekly_counts"], ctx["weekly_run_target"]
    st.subheader("Weekly Run Count")
    if acts.empty:
        st.caption("No activities yet.")
        return
    counts_df = weekly_counts.tail(12).reset_index()
    counts_df.columns = ["week_period", "runs"]
    counts_df["week"] = counts_df["week_period"].apply(lambda p: p.start_time)
    bars = (
        alt.Chart(counts_df).mark_bar(color=CATEGORICAL[2], cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(x=alt.X("week:T", title="Week"), y=alt.Y("runs:Q", title="Runs", axis=alt.Axis(tickMinStep=1, format="d")),
                tooltip=[alt.Tooltip("week:T", title="Week"), alt.Tooltip("runs:Q", title="Runs")])
    )
    target_rule = alt.Chart(pd.DataFrame({"target": [target]})).mark_rule(strokeDash=[4, 4], color=STATUS_WARNING).encode(y="target:Q")
    st.altair_chart(bars + target_rule, width="stretch")
    st.caption(f"Current streak: {compute_streak(weekly_counts, target)} week(s) hitting {target}+ runs. Dashed line marks your target.")


# --------------------------------------------------------------------------
# Fitness tab
# --------------------------------------------------------------------------

@section("vo2max", "Fitness", "VO2max Trend")
def render_vo2max(ctx):
    st.subheader("VO2max Trend")
    chart = crosshair_chart(melt_for_chart(ctx["daily"], {"vo2max.value": "VO2max"}), colors=[CATEGORICAL[0]])
    if chart is not None:
        st.altair_chart(chart, width="stretch")
    else:
        st.caption("No VO2max data yet.")


@section("race_predictor_long", "Fitness", "Race Predictor: Marathon & Half Marathon")
def render_race_predictor_long(ctx):
    daily = ctx["daily"]
    st.subheader("Race Predictor: Marathon & Half Marathon")
    cols = ["race_predictions.time_marathon_sec", "race_predictions.time_half_sec"]
    race_wide = daily[cols] / 60 if all(c in daily.columns for c in cols) else pd.DataFrame()
    long_df = melt_for_chart(race_wide, {"race_predictions.time_marathon_sec": "Marathon", "race_predictions.time_half_sec": "Half Marathon"})
    chart = crosshair_chart(long_df, y_title="Predicted time (min)")
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


@section("race_predictor_short", "Fitness", "Race Predictor: 5K & 10K")
def render_race_predictor_short(ctx):
    daily = ctx["daily"]
    st.subheader("Race Predictor: 5K & 10K")
    cols = ["race_predictions.time_5k_sec", "race_predictions.time_10k_sec"]
    race_wide = daily[cols] / 60 if all(c in daily.columns for c in cols) else pd.DataFrame()
    chart = crosshair_chart(melt_for_chart(race_wide, {"race_predictions.time_5k_sec": "5K", "race_predictions.time_10k_sec": "10K"}), y_title="Predicted time (min)")
    if chart is not None:
        st.altair_chart(chart, width="stretch")
    else:
        st.caption("No race predictor data yet.")


@section("own_predictor", "Fitness", "Your Model vs. Garmin")
def render_own_predictor(ctx):
    st.subheader("Your Model vs. Garmin")
    info_popover(
        "Riegel's formula (T₂ = T₁ × (D₂/D₁)^1.06) fitted to your own best-known effort, shown "
        "next to Garmin's own estimate. Garmin leans on VO2max-derived potential; this leans on what "
        "you've actually run. Where they disagree is the interesting part."
    )
    own = ctx["own_predictions"]
    daily = ctx["daily"]
    garmin_cols = {"5k": "race_predictions.time_5k_sec", "10k": "race_predictions.time_10k_sec",
                   "half": "race_predictions.time_half_sec", "marathon": "race_predictions.time_marathon_sec"}
    latest = daily.iloc[-1] if not daily.empty else {}
    rows = []
    for label, col in garmin_cols.items():
        garmin_val = latest.get(col) if isinstance(latest, pd.Series) else None
        rows.append({"Distance": BEST_EFFORT_LABELS.get(label, label.title()) if label in BEST_EFFORT_LABELS else label.title(),
                      "Garmin": format_duration(garmin_val), "Your model": format_duration(own.get(label))})
    if not own and all(pd.isna(r["Garmin"]) or r["Garmin"] == "--" for r in rows):
        st.caption("Not enough data yet — need at least one recorded best effort and/or Garmin prediction.")
        return
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


@section("heat_adjusted", "Fitness", "Heat-Adjusted Pace")
def render_heat_adjusted(ctx):
    st.subheader("Heat-Adjusted Pace")
    info_popover(
        "A simple linear approximation (~0.5%/°C above 15°C) — not a validated physiological model, "
        "just a directional correction so hot-weather runs don't read as a fitness plateau."
    )
    heat = ctx["heat"]
    if heat.empty:
        st.caption("No weather-enriched runs yet.")
        return
    long_df = heat.reset_index().rename(columns={"index": "date"}).melt(
        id_vars=["date", "temp_c"], value_vars=["raw_pace", "adjusted_pace"], var_name="Series", value_name="value"
    )
    long_df["Series"] = long_df["Series"].map({"raw_pace": "Raw pace", "adjusted_pace": "Heat-adjusted"})
    chart = crosshair_chart(long_df, y_title="min/km", colors=[INK_3, ACCENT])
    if chart is not None:
        st.altair_chart(chart, width="stretch")


@section("efficiency_factor", "Fitness", "Efficiency Factor")
def render_efficiency_factor(ctx):
    st.subheader("Efficiency Factor")
    info_popover("Speed (m/s) per heartbeat. Rising means the same effort produces more speed — the cleanest single signal of aerobic gain.")
    ef = ctx["ef"]
    if ef.empty:
        st.caption("No data yet.")
        return
    long_df = ef.reset_index()
    long_df.columns = ["date", "value"]
    long_df["Series"] = "EF"
    chart = crosshair_chart(long_df, colors=[ACCENT])
    if chart is not None:
        st.altair_chart(chart, width="stretch")


@section("personal_records", "Fitness", "Personal Records")
def render_personal_records(ctx):
    acts, daily = ctx["acts"], ctx["daily"]
    st.subheader("Personal Records")
    if acts.empty or not acts["avg_pace_min_per_km"].notna().any():
        st.caption("No activities yet.")
        return
    fastest_idx = acts["avg_pace_min_per_km"].idxmin()
    longest_idx = acts["distance_km"].idxmax()
    cols = st.columns(3)
    cols[0].metric("Fastest Pace", format_pace(acts.loc[fastest_idx, "avg_pace_min_per_km"]))
    cols[0].caption(acts.loc[fastest_idx, "date"].strftime("%b %d, %Y"))
    cols[1].metric("Longest Run", f"{acts.loc[longest_idx, 'distance_km']:.1f} km")
    cols[1].caption(acts.loc[longest_idx, "date"].strftime("%b %d, %Y"))
    if "vo2max.value" in daily.columns and daily["vo2max.value"].notna().any():
        vo2_idx = daily["vo2max.value"].idxmax()
        cols[2].metric("Highest VO2max", f"{daily.loc[vo2_idx, 'vo2max.value']:.1f}")
        cols[2].caption(vo2_idx.strftime("%b %d, %Y"))
    else:
        cols[2].metric("Highest VO2max", "--")


# --------------------------------------------------------------------------
# Best Efforts tab
# --------------------------------------------------------------------------

@section("best_efforts", "Best Efforts", "Best Efforts")
def render_best_efforts(ctx):
    acts = ctx["acts"]
    st.subheader("Best Efforts")
    info_popover(
        "Fastest **continuous** segment at each distance, found by scanning the raw GPS/time stream of "
        "every run — not Garmin's own lap markers, which are inconsistent between manual and auto laps. "
        "So a fast 5K buried inside a long run still counts."
    )
    if acts.empty or "best_efforts" not in acts.columns:
        st.caption("No enriched runs yet.")
        return

    dist_label = st.segmented_control("Distance", list(BEST_EFFORT_LABELS.values()), default="5 km", key="be_distance")
    dist_key = {v: k for k, v in BEST_EFFORT_LABELS.items()}.get(dist_label, "5k")

    this_week, all_time = A.this_week_vs_alltime_best(acts, dist_key)
    stat_cols = st.columns(2)
    stat_cols[0].metric(f"This week's best {dist_label}", format_duration(this_week) if this_week else "no effort")
    delta = None
    if this_week and all_time and this_week > all_time:
        delta = f"+{this_week - all_time:.0f}s vs all-time"
    elif this_week and all_time and this_week == all_time:
        delta = "matches all-time best"
    stat_cols[1].metric(f"All-time best {dist_label}", format_duration(all_time) if all_time else "--", delta=delta if this_week != all_time else None)

    st.markdown("**Leaderboard**")
    board = A.best_efforts_leaderboard(acts, dist_key, top_n=8)
    if board.empty:
        st.caption("No efforts recorded at this distance yet.")
        return
    board = board.copy()
    board["Time"] = board["time_sec"].apply(format_duration)
    board["Date"] = board["date"].dt.strftime("%b %d, %Y")
    board["Within"] = board.apply(lambda r: f"{r['run_type'] or 'Run'} · {r['distance_km']:.1f} km", axis=1)
    st.dataframe(board[["Time", "Within", "Date"]], width="stretch")


# --------------------------------------------------------------------------
# Recovery tab
# --------------------------------------------------------------------------

@section("hrv_rhr", "Recovery", "HRV & Resting Heart Rate")
def render_hrv_rhr(ctx):
    daily = ctx["daily"]
    st.subheader("HRV & Resting Heart Rate (last 8 weeks)")
    chart = crosshair_chart(melt_for_chart(daily.tail(56), {"hrv.weekly_avg": "HRV (weekly avg, ms)", "rhr.avg_7day": "Resting HR (7-day avg, bpm)"}))
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
    chart = crosshair_chart(melt_for_chart(ctx["daily"], {"sleep.duration_hours": "Nightly Duration (hrs)", "sleep.avg_7day_duration_hours": "7-Day Avg (hrs)"}), y_title="hours")
    if chart is not None:
        st.altair_chart(chart, width="stretch")
    else:
        st.caption("No sleep data yet.")


@section("stress", "Recovery", "Stress Trend")
def render_stress(ctx):
    st.subheader("Stress Trend")
    chart = crosshair_chart(melt_for_chart(ctx["daily"], {"stress.avg_stress": "Daily Stress", "stress.avg_7day": "7-Day Avg"}), y_title="stress score")
    if chart is not None:
        st.altair_chart(chart, width="stretch")
    else:
        st.caption("No stress data yet.")


@section("body_battery", "Recovery", "Body Battery")
def render_body_battery(ctx):
    st.subheader("Body Battery")
    chart = crosshair_chart(melt_for_chart(ctx["daily"], {"body_battery.end_level": "End-of-day level"}), y_title="level")
    if chart is not None:
        st.altair_chart(chart, width="stretch")
    else:
        st.caption("No body battery data yet.")


@section("polarization", "Recovery", "Training Polarization")
def render_polarization(ctx):
    st.subheader("Training Polarization")
    info_popover(
        "The 80/20 rule: roughly four-fifths of training time should be genuinely easy (Z1–2), with "
        "hard efforts (Z4–5) deliberately rare. Computed from time-in-HR-zone across your enriched runs."
    )
    pol = ctx["polarization"]
    if not pol:
        st.caption("No HR-zone data yet.")
        return
    cols = st.columns(3)
    cols[0].metric("Easy (Z1–2)", f"{pol['easy_pct']:.0f}%")
    cols[1].metric("Moderate (Z3)", f"{pol['moderate_pct']:.0f}%")
    cols[2].metric("Hard (Z4–5)", f"{pol['hard_pct']:.0f}%")
    if pol["easy_pct"] < 75:
        st.caption(f"You're at {pol['easy_pct']:.0f}% easy, below the ~80% target — a common sign of running easy days too hard.")


# --------------------------------------------------------------------------
# Runs tab
# --------------------------------------------------------------------------

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
        hide_index=True, column_config={"Date": st.column_config.DateColumn(format="YYYY-MM-DD")},
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
        alt.Chart(scatter_df).mark_circle(size=80).encode(
            x=alt.X("avg_pace_min_per_km:Q", title="Avg Pace (min/km)", scale=alt.Scale(reverse=True)),
            y=alt.Y("avg_hr:Q", title="Avg HR (bpm)"),
            color=alt.Color("run_type:N", title="Run Type", scale=alt.Scale(domain=run_type_domain, range=CATEGORICAL)),
            tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("distance_km:Q", title="Distance (km)"),
                     alt.Tooltip("avg_pace_min_per_km:Q", title="Pace (min/km)"), alt.Tooltip("avg_hr:Q", title="Avg HR"),
                     alt.Tooltip("run_type:N", title="Type")],
        ).interactive()
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
        alt.Chart(cal_df).mark_rect(cornerRadius=2).encode(
            x=alt.X("week:T", title=None, axis=alt.Axis(format="%b %d")),
            y=alt.Y("weekday:O", sort=weekday_order, title=None),
            color=alt.Color("distance_km:Q", title="km", scale=alt.Scale(range=SEQUENTIAL_BLUES)),
            tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("distance_km:Q", title="km", format=".1f")],
        )
    )
    st.altair_chart(heatmap, width="stretch")


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

st.sidebar.header("Running Dashboard")
config = load_config()

race_date_val = date.fromisoformat(config["race_date"]) if config.get("race_date") else None
new_race_date = st.sidebar.date_input("Race date", value=race_date_val)
race_date = new_race_date
if race_date != race_date_val:
    set_config_value(config, "race_date", race_date.isoformat() if race_date else None)

goal_input = st.sidebar.text_input("Goal time (H:MM:SS)", value=seconds_to_hms(config.get("goal_marathon_time_sec")), placeholder="3:30:00")
goal_marathon_sec = parse_hms(goal_input) if goal_input else None
if goal_input and goal_marathon_sec is None:
    st.sidebar.caption("⚠️ Couldn't parse that as H:MM:SS")
elif goal_marathon_sec != config.get("goal_marathon_time_sec"):
    set_config_value(config, "goal_marathon_time_sec", goal_marathon_sec)

with st.sidebar.expander("⚙️ Settings"):
    weekly_run_target = st.number_input("Weekly run target", min_value=1, max_value=14, value=config.get("weekly_run_target", DEFAULT_WEEKLY_RUN_TARGET))
    set_config_value(config, "weekly_run_target", weekly_run_target)

    phase_lookback_weeks = st.number_input(
        "Training block length (weeks)", min_value=4, max_value=52,
        value=config.get("phase_lookback_weeks", DEFAULT_PHASE_LOOKBACK_WEEKS),
        help="How far back the Base/Build/Peak/Taper rail starts, counted from race day.",
    )
    set_config_value(config, "phase_lookback_weeks", phase_lookback_weeks)

    st.caption("Advanced: run-type inference")
    long_run_factor = st.slider("Long-run distance factor", 1.0, 2.0, value=config.get("long_run_distance_factor", DEFAULT_LONG_RUN_DISTANCE_FACTOR), step=0.05)
    set_config_value(config, "long_run_distance_factor", long_run_factor)
    tempo_percentile = st.slider("Tempo/speed pace percentile", 0.05, 0.5, value=config.get("tempo_pace_percentile", DEFAULT_TEMPO_PACE_PERCENTILE), step=0.05)
    set_config_value(config, "tempo_pace_percentile", tempo_percentile)

    st.caption("Sections shown")
    enabled_sections = config.get("enabled_sections", {})
    for tab_name in TAB_ORDER:
        tab_sections = [s for s in SECTIONS if s["tab"] == tab_name]
        if not tab_sections:
            continue
        st.caption(f"**{tab_name}**")
        for s in tab_sections:
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

daily, acts = load_frames(raw)

if daily.empty and acts.empty:
    st.title("🏃 Running Dashboard")
    st.info("No data yet. Click **Sync now** in the sidebar to pull your first batch from Garmin Connect.")
    st.stop()

if not acts.empty:
    acts["run_type"] = infer_run_type(acts, long_run_factor, tempo_percentile)


# --------------------------------------------------------------------------
# Derived aggregates
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

daily_load = A.compute_daily_load(acts, daily) if not acts.empty else pd.Series(dtype=float)
pmc = A.compute_pmc(daily_load)
acwr = A.compute_acwr(daily_load)
heat = A.heat_adjusted_pace(acts) if not acts.empty else pd.DataFrame()
ef = A.efficiency_factor(acts) if not acts.empty else pd.Series(dtype=float)
polarization_stats = A.polarization(acts) if not acts.empty else {}

all_time_best_efforts = {}
if not acts.empty and "best_efforts" in acts.columns:
    for _, row in acts.iterrows():
        be = row.get("best_efforts")
        if isinstance(be, dict):
            for k, v in be.items():
                if k not in all_time_best_efforts or v < all_time_best_efforts[k]:
                    all_time_best_efforts[k] = v
own_predictions = A.predict_race_times(all_time_best_efforts)

ctx = {
    "daily": daily, "acts": acts, "weekly_periods": weekly_periods, "weekly_counts": weekly_counts,
    "monthly_periods": monthly_periods, "long_run_this_week": long_run_this_week, "weekly_run_target": weekly_run_target,
    "goal_marathon_sec": goal_marathon_sec, "pmc": pmc, "acwr": acwr, "heat": heat, "ef": ef,
    "polarization": polarization_stats, "own_predictions": own_predictions,
}


# --------------------------------------------------------------------------
# Race clock hero
# --------------------------------------------------------------------------

days_to_go = (race_date - date.today()).days if race_date else None
phase = training_phase(days_to_go)

if race_date:
    block_start = race_date - timedelta(weeks=phase_lookback_weeks)
    total_span = max((race_date - block_start).days, 1)
    elapsed = min(max((date.today() - block_start).days, 0), total_span)
    pct_now = elapsed / total_span * 100
    marks_days = [PHASE_BUILD_DAYS, PHASE_PEAK_DAYS, PHASE_TAPER_DAYS]
    marks_pct = [max(0, min(100, (total_span - d) / total_span * 100)) for d in marks_days if d < total_span]
    marks_html = "".join(f'<div class="rd-rail-mark" style="left:{p:.1f}%"></div>' for p in marks_pct)
    labels = ["Base", "Build", "Peak", "Taper"]
    labels_html = "".join(f'<span class="{"on" if labels[i] == phase else ""}">{labels[i]}</span>' for i in range(4))
    goal_line = f'<div class="goal">Goal {seconds_to_hms(goal_marathon_sec)} · {format_pace(goal_marathon_sec/60/(42195/1000))}</div>' if goal_marathon_sec else ""
    st.markdown(f"""
    <div class="rd-clock">
      <div class="rd-clock-top">
        <div><span class="rd-eyebrow">Time to start line</span><div class="rd-days">{days_to_go}<sup>days</sup></div></div>
        <div class="rd-target">
          <div class="name">Race day</div>
          <div class="sub">{race_date.strftime("%a %d %b %Y")}</div>
          {goal_line}
        </div>
      </div>
      <div class="rd-rail-track">
        <div class="rd-rail-fill" style="width:{pct_now:.1f}%"></div>
        {marks_html}
        <div class="rd-rail-now" style="left:{pct_now:.1f}%"></div>
      </div>
      <div class="rd-rail-labels">{labels_html}</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("Set your race date in the sidebar to see the countdown and training-phase rail.")

st.title("🏃 Running Dashboard")


# --------------------------------------------------------------------------
# Status chips
# --------------------------------------------------------------------------

chips_html = '<div class="rd-chips">'
latest_tsb = pmc["tsb"].iloc[-1] if not pmc.empty else None
tsb_sev = A.classify_tsb(latest_tsb)
tsb_status = {"good": "Fresh — absorbing load well", "warning": "Building fatigue", "critical": "High fatigue — recover"}.get(tsb_sev, "No data yet")
chips_html += chip("Form (TSB)", f"{latest_tsb:+.1f}" if latest_tsb is not None else "--", tsb_status, tsb_sev)

latest_acwr = acwr.iloc[-1] if not acwr.empty else None
acwr_sev = A.classify_acwr(latest_acwr)
acwr_status = {"good": "In the sweet spot", "warning": "Outside 0.8–1.3", "critical": "High injury-risk zone"}.get(acwr_sev, "No data yet")
chips_html += chip("Load Ratio (ACWR)", f"{latest_acwr:.2f}" if latest_acwr is not None else "--", acwr_status, acwr_sev)

hrv_status_val = daily["hrv.status"].dropna().iloc[-1] if "hrv.status" in daily.columns and daily["hrv.status"].notna().any() else None
rhr_val = daily["rhr.value"].dropna().iloc[-1] if "rhr.value" in daily.columns and daily["rhr.value"].notna().any() else None
hrv_sev = "good" if hrv_status_val == "BALANCED" else ("warning" if hrv_status_val else None)
chips_html += chip("HRV Status", hrv_status_val or "--", f"RHR {rhr_val:.0f} bpm" if rhr_val else "No data yet", hrv_sev)

this_week_km = weekly_periods.iloc[-1] if not weekly_periods.empty else None
runs_this_week = weekly_counts.iloc[-1] if not weekly_counts.empty else None
long_run_str = f"long run {long_run_this_week:.1f} km" if long_run_this_week is not None else "no runs yet"
chips_html += chip("This Week", f'{this_week_km:.1f}<small>km</small>' if this_week_km is not None else "--",
                    f"{int(runs_this_week) if runs_this_week is not None else 0} runs · {long_run_str}", "accent")
chips_html += "</div>"
st.markdown(chips_html, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------

# st.tabs() panels are hidden (height-collapsed) until clicked, and Vega /
# the dataframe grid measure their container once at mount time -- they never
# recover from mounting inside a zero-height panel. A segmented control that
# conditionally renders only the selected tab's Python code sidesteps that
# entirely: nothing ever mounts while hidden.
active_tab = st.segmented_control("Section", TAB_ORDER, default=TAB_ORDER[0], label_visibility="collapsed")
if not active_tab:
    active_tab = TAB_ORDER[0]

tab_sections = [s for s in SECTIONS if s["tab"] == active_tab]
visible = [s for s in tab_sections if enabled_sections.get(s["id"], s["default"])]
if not visible:
    st.caption("Nothing enabled in this tab — turn sections back on in Settings.")
else:
    for s in visible:
        s["render"](ctx)
        st.divider()
