"""Builds the static GitHub Pages demo at docs/index.html from the synthetic
sample dataset (docs/sample_data/garmin_history.json — see
generate_sample_data.py). Renders the same analytics (analytics.py, unchanged)
into a static page: charts as embedded Vega-Lite specs (vega-embed, CDN),
everything else as plain HTML computed with pandas.

This intentionally does NOT run dashboard.py or Streamlit — Streamlit has no
static-export mode. It re-implements dashboard.py's chart-construction logic
in a Streamlit-free form so it can run outside a Streamlit server, and reuses
analytics.py (already Streamlit-free) unchanged for all the actual math.

Run:
    python scripts/build_demo_site.py
Writes docs/index.html.
"""

import json
import sys
from datetime import date, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))  # so `import analytics` resolves regardless of cwd

import altair as alt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import analytics as A

SAMPLE_DATA_FILE = BASE_DIR / "docs" / "sample_data" / "garmin_history.json"
OG_IMAGE_FILE = BASE_DIR / "docs" / "og-image.png"
OUT_FILE = BASE_DIR / "docs" / "index.html"

DEMO_RACE_DATE = date(2026, 10, 25)
DEMO_GOAL_MARATHON_SEC = 14400  # 4:00:00
DEMO_WEEKLY_RUN_TARGET = 4
DEMO_LONG_RUN_FACTOR = 1.3
DEMO_TEMPO_PERCENTILE = 0.25
DEMO_PHASE_LOOKBACK_WEEKS = 20
PHASE_TAPER_DAYS, PHASE_PEAK_DAYS, PHASE_BUILD_DAYS = 14, 42, 98

REPO_URL = "https://github.com/jmp1909/garmin-data-tool"
SITE_TITLE = "Running Dashboard — Live Demo"
SITE_DESCRIPTION = (
    "A personal Garmin Connect training dashboard: fitness/fatigue/form, "
    "injury-risk load ratio, heat-adjusted pace, and a from-scratch race "
    "predictor — computed from scratch, not just a re-skin of Garmin Connect."
)

# Light-theme palette, matching dashboard.py's marathon_theme (light branch)
ACCENT = "#1e48c8"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a"]
STATUS_GOOD = "#157f3d"
STATUS_WARNING = "#b45309"
STATUS_CRITICAL = "#b42318"
INK = "#14181a"
INK_2 = "#5a6468"
INK_3 = "#8b9599"
SURFACE = "#ffffff"
GROUND = "#f6f7f6"
RULE = "#e2e6e5"
RULE_STRONG = "#cfd5d4"
MONO = 'ui-monospace, "SF Mono", "Cascadia Mono", "Roboto Mono", Menlo, Consolas, monospace'
SEV_COLOR = {"good": STATUS_GOOD, "warning": STATUS_WARNING, "critical": STATUS_CRITICAL, "accent": ACCENT, None: INK_3}

# Fixed (not "container") width: charts for inactive tabs mount inside a
# display:none panel, where a container-relative width would resolve to 0.
CHART_WIDTH = 900


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


# --------------------------------------------------------------------------
# Data loading / helpers (ported from dashboard.py, Streamlit calls removed)
# --------------------------------------------------------------------------

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


def infer_run_type(acts, long_run_factor, tempo_percentile):
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


def seconds_to_hms(total_seconds):
    if not total_seconds:
        return ""
    total_seconds = int(round(total_seconds))
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


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


def compute_streak(weekly_counts, target):
    complete_weeks = weekly_counts.iloc[:-1] if len(weekly_counts) > 0 else weekly_counts
    streak = 0
    for count in reversed(complete_weeks.tolist()):
        if count >= target:
            streak += 1
        else:
            break
    return streak


def melt_for_chart(df_wide, rename):
    present = [c for c in rename if c in df_wide.columns]
    if not present:
        return pd.DataFrame(columns=["date", "Series", "value"])
    value_vars = [rename[c] for c in present]
    long_df = df_wide[present].reset_index().rename(columns={df_wide.index.name or "index": "date", **rename})
    long_df = long_df.melt(id_vars="date", value_vars=value_vars, var_name="Series", value_name="value")
    return long_df.dropna(subset=["value"])


def crosshair_chart(long_df, y_title=None, colors=None, zero_line=False):
    if long_df is None or long_df.empty:
        return None
    series = sorted(long_df["Series"].unique().tolist())
    palette = colors or CATEGORICAL

    pivot_df = long_df.pivot_table(index="date", columns="Series", values="value", aggfunc="first").reset_index().sort_values("date")
    dense_dates = pd.DataFrame({"date": pd.date_range(pivot_df["date"].min(), pivot_df["date"].max(), freq="D")})
    hover_source = pivot_df.assign(actual_date=pivot_df["date"])
    hover_df = pd.merge_asof(dense_dates, hover_source, on="date", direction="nearest")
    hover_long = hover_df.melt(id_vars=["date", "actual_date"], value_vars=series, var_name="Series", value_name="value").dropna(subset=["value"])

    base = alt.Chart(long_df)
    hover_base = alt.Chart(hover_df)
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["date"], empty=False)

    color_enc = alt.Color("Series:N", title=None, scale=alt.Scale(domain=series, range=palette[:len(series)])) \
        if len(series) > 1 else alt.value(palette[0])

    layers = []
    if zero_line:
        layers.append(alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=RULE_STRONG, strokeDash=[3, 3]).encode(y="y:Q"))

    line = base.mark_line(point=False).encode(x=alt.X("date:T", title=None), y=alt.Y("value:Q", title=y_title), color=color_enc)
    points = alt.Chart(hover_long).mark_point(size=45).encode(
        x="date:T", y=alt.Y("value:Q"), color=color_enc,
        opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
    )
    tooltip = [alt.Tooltip(field="actual_date", type="temporal", title="Date")] + [
        alt.Tooltip(field=s, type="quantitative", title=s, format=".2f") for s in series
    ]
    rule_visible = hover_base.mark_rule(color=RULE_STRONG).encode(
        x="date:T", opacity=alt.condition(nearest, alt.value(0.5), alt.value(0))
    )
    selectors = hover_base.mark_rule(opacity=0, strokeWidth=24).encode(x="date:T", tooltip=tooltip).add_params(nearest)

    layers += [line, points, rule_visible, selectors]
    return alt.layer(*layers).properties(height=280, width=CHART_WIDTH)


def generate_og_image():
    """A simple branded 1200x630 PNG (standard OG/Twitter card size) for
    LinkedIn/social link previews -- no chart data, just title/subtitle, so
    it never needs to change when the sample data does."""
    W, H = 1200, 630
    img = Image.new("RGB", (W, H), GROUND)
    draw = ImageDraw.Draw(img)

    def font(size, bold=False):
        try:
            name = "segoeuib.ttf" if bold else "segoeui.ttf"
            return ImageFont.truetype(name, size)
        except OSError:
            return ImageFont.load_default(size=size)

    draw.rectangle([0, 0, 14, H], fill=ACCENT)
    draw.text((70, 210), "Running Dashboard", font=font(64, bold=True), fill=INK)
    draw.text((70, 300), "Fitness, fatigue, and race prediction —", font=font(30), fill=INK_2)
    draw.text((70, 344), "computed from your own training data.", font=font(30), fill=INK_2)
    draw.text((70, 420), "LIVE DEMO · SAMPLE DATA", font=font(22, bold=True), fill=ACCENT)

    img.save(OG_IMAGE_FILE)


# --------------------------------------------------------------------------
# Build everything
# --------------------------------------------------------------------------

def main():
    generate_og_image()
    raw = json.loads(SAMPLE_DATA_FILE.read_text())
    daily, acts = load_frames(raw)
    acts["run_type"] = infer_run_type(acts, DEMO_LONG_RUN_FACTOR, DEMO_TEMPO_PERCENTILE)

    weekly_periods = acts.groupby(acts["date"].dt.to_period("W"))["distance_km"].sum()
    weekly_counts = acts.groupby(acts["date"].dt.to_period("W"))["activity_id"].count()
    monthly_periods = acts.groupby(acts["date"].dt.to_period("M"))["distance_km"].sum()
    current_week_period = acts["date"].max().to_period("W")
    long_run_this_week = acts.loc[acts["date"].dt.to_period("W") == current_week_period, "distance_km"].max()

    daily_load = A.compute_daily_load(acts, daily)
    pmc = A.compute_pmc(daily_load)
    acwr = A.compute_acwr(daily_load)
    heat = A.heat_adjusted_pace(acts)
    ef = A.efficiency_factor(acts)
    weekly_pol = A.weekly_polarization(acts)

    all_time_best_efforts = {}
    for _, row in acts.iterrows():
        be = row.get("best_efforts")
        if isinstance(be, dict):
            for k, v in be.items():
                if k not in all_time_best_efforts or v < all_time_best_efforts[k]:
                    all_time_best_efforts[k] = v
    riegel = A.predict_race_times_riegel(all_time_best_efforts)
    cameron = A.predict_race_times_cameron(all_time_best_efforts)
    latest_vo2max = daily["vo2max.value"].dropna().iloc[-1] if daily["vo2max.value"].notna().any() else None
    daniels = A.predict_race_times_daniels(latest_vo2max)

    charts = {}

    # -- Load tab --
    pmc_long = pmc.reset_index().rename(columns={"index": "date"}).melt(
        id_vars="date", value_vars=["ctl", "atl", "tsb"], var_name="Series", value_name="value")
    pmc_long["Series"] = pmc_long["Series"].map({"ctl": "Fitness (CTL)", "atl": "Fatigue (ATL)", "tsb": "Form (TSB)"})
    charts["pmc"] = crosshair_chart(pmc_long, colors=[CATEGORICAL[0], INK_3, STATUS_GOOD], zero_line=True)

    acwr_long = acwr.reset_index().rename(columns={"index": "date", "acwr": "value"})
    acwr_long["Series"] = "ACWR"
    acwr_chart = crosshair_chart(acwr_long, colors=[ACCENT])
    if acwr_chart is not None:
        band = alt.Chart(pd.DataFrame({"lo": [0.8], "hi": [1.3]})).mark_rect(opacity=0.08, color=STATUS_GOOD).encode(y="lo:Q", y2="hi:Q")
        ceiling = alt.Chart(pd.DataFrame({"y": [1.3]})).mark_rule(strokeDash=[4, 3], color=STATUS_WARNING).encode(y="y:Q")
        charts["acwr"] = band + ceiling + acwr_chart
    else:
        charts["acwr"] = None

    weekly_recent = weekly_periods.tail(12)
    weekly_df = pd.DataFrame({
        "week_start": weekly_recent.index.start_time,
        "week_end": weekly_recent.index.end_time.normalize(),
        "distance_km": weekly_recent.values,
    })
    weekly_df["week_label"] = weekly_df["week_start"].dt.strftime("%b %d") + " - " + weekly_df["week_end"].dt.strftime("%b %d")
    charts["weekly_mileage"] = (
        alt.Chart(weekly_df).mark_bar(color=CATEGORICAL[0], cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(x=alt.X("week_start:T", title="Week"), y=alt.Y("distance_km:Q", title="km"),
                tooltip=[alt.Tooltip("week_label:N", title="Week"), alt.Tooltip("distance_km:Q", title="Distance (km)", format=".1f")])
        .properties(height=260, width=CHART_WIDTH)
    )

    monthly_df = monthly_periods.reset_index()
    monthly_df.columns = ["month_period", "distance_km"]
    monthly_df["month"] = monthly_df["month_period"].apply(lambda p: p.start_time)
    monthly_df["month_label"] = monthly_df["month_period"].astype(str)
    monthly_df = monthly_df.drop(columns=["month_period"])
    charts["monthly_mileage"] = (
        alt.Chart(monthly_df).mark_bar(color=CATEGORICAL[1], cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(x=alt.X("month:T", title="Month", axis=alt.Axis(format="%b %Y")),
                y=alt.Y("distance_km:Q", title="km"),
                tooltip=[alt.Tooltip("month_label:N", title="Month"), alt.Tooltip("distance_km:Q", title="km", format=".1f")])
        .properties(height=260, width=CHART_WIDTH)
    )

    counts_df = weekly_counts.tail(12).reset_index()
    counts_df.columns = ["week_period", "runs"]
    counts_df["week"] = counts_df["week_period"].apply(lambda p: p.start_time)
    counts_df = counts_df.drop(columns=["week_period"])
    bars = (
        alt.Chart(counts_df).mark_bar(color=CATEGORICAL[2], cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(x=alt.X("week:T", title="Week"), y=alt.Y("runs:Q", title="Runs", axis=alt.Axis(tickMinStep=1, format="d")),
                tooltip=[alt.Tooltip("week:T", title="Week"), alt.Tooltip("runs:Q", title="Runs")])
    )
    target_rule = alt.Chart(pd.DataFrame({"target": [DEMO_WEEKLY_RUN_TARGET]})).mark_rule(strokeDash=[4, 4], color=STATUS_WARNING).encode(y="target:Q")
    charts["weekly_run_count"] = (bars + target_rule).properties(height=260, width=CHART_WIDTH)

    # -- Fitness tab --
    charts["vo2max"] = crosshair_chart(melt_for_chart(daily, {"vo2max.value": "VO2max"}), colors=[CATEGORICAL[0]])

    cols_long = ["race_predictions.time_marathon_sec", "race_predictions.time_half_sec"]
    race_wide_long = daily[cols_long] / 60 if all(c in daily.columns for c in cols_long) else pd.DataFrame()
    long_race_df = melt_for_chart(race_wide_long, {"race_predictions.time_marathon_sec": "Marathon", "race_predictions.time_half_sec": "Half Marathon"})
    race_chart_long = crosshair_chart(long_race_df, y_title="Predicted time (min)")
    if race_chart_long is not None:
        goal_minutes = DEMO_GOAL_MARATHON_SEC / 60
        goal_rule = alt.Chart(pd.DataFrame({"minutes": [goal_minutes]})).mark_rule(strokeDash=[4, 4], color=STATUS_GOOD).encode(y="minutes:Q")
        charts["race_predictor_long"] = alt.layer(race_chart_long, goal_rule)
    else:
        charts["race_predictor_long"] = None

    cols_short = ["race_predictions.time_5k_sec", "race_predictions.time_10k_sec"]
    race_wide_short = daily[cols_short] / 60 if all(c in daily.columns for c in cols_short) else pd.DataFrame()
    charts["race_predictor_short"] = crosshair_chart(
        melt_for_chart(race_wide_short, {"race_predictions.time_5k_sec": "5K", "race_predictions.time_10k_sec": "10K"}),
        y_title="Predicted time (min)")

    charts["heat_adjusted"] = None
    if not heat.empty:
        heat_long = heat.reset_index().rename(columns={"index": "date"}).melt(
            id_vars=["date", "temp_c"], value_vars=["raw_pace", "adjusted_pace"], var_name="Series", value_name="value")
        heat_long["Series"] = heat_long["Series"].map({"raw_pace": "Raw pace", "adjusted_pace": "Heat-adjusted"})
        charts["heat_adjusted"] = crosshair_chart(heat_long, y_title="min/km", colors=[INK_3, ACCENT])

    charts["efficiency_factor"] = None
    if not ef.empty:
        ef_long = ef.reset_index()
        ef_long.columns = ["date", "value"]
        ef_long["Series"] = "EF"
        charts["efficiency_factor"] = crosshair_chart(ef_long, colors=[ACCENT])

    # -- Runs tab --
    run_type_domain = ["Easy/Recovery", "Long Run", "Tempo/Speed"]
    scatter_df = acts.dropna(subset=["avg_pace_min_per_km", "avg_hr"])
    charts["pace_hr_scatter"] = (
        alt.Chart(scatter_df).mark_circle(size=80).encode(
            x=alt.X("avg_pace_min_per_km:Q", title="Avg Pace (min/km)", scale=alt.Scale(reverse=True)),
            y=alt.Y("avg_hr:Q", title="Avg HR (bpm)"),
            color=alt.Color("run_type:N", title="Run Type", scale=alt.Scale(domain=run_type_domain, range=CATEGORICAL)),
            tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("distance_km:Q", title="Distance (km)"),
                     alt.Tooltip("avg_pace_min_per_km:Q", title="Pace (min/km)"), alt.Tooltip("avg_hr:Q", title="Avg HR"),
                     alt.Tooltip("run_type:N", title="Type")],
        ).properties(height=320, width=CHART_WIDTH)
    )

    # -- Recovery tab --
    charts["hrv_rhr"] = crosshair_chart(melt_for_chart(daily.tail(56), {"hrv.weekly_avg": "HRV (weekly avg, ms)", "rhr.avg_7day": "Resting HR (7-day avg, bpm)"}))
    charts["sleep"] = crosshair_chart(melt_for_chart(daily, {"sleep.duration_hours": "Nightly Duration (hrs)", "sleep.avg_7day_duration_hours": "7-Day Avg (hrs)"}), y_title="hours")
    charts["stress"] = crosshair_chart(melt_for_chart(daily, {"stress.avg_stress": "Daily Stress", "stress.avg_7day": "7-Day Avg"}), y_title="stress score")

    charts["polarization"] = None
    if not weekly_pol.empty:
        pol_long = weekly_pol.melt(id_vars="week", value_vars=["easy_pct", "moderate_pct", "hard_pct"], var_name="Zone", value_name="pct")
        zone_labels = {"easy_pct": "Easy (Z1–2)", "moderate_pct": "Moderate (Z3)", "hard_pct": "Hard (Z4–5)"}
        pol_long["Zone"] = pol_long["Zone"].map(zone_labels)
        zone_order = ["Easy (Z1–2)", "Moderate (Z3)", "Hard (Z4–5)"]
        zone_colors = [CATEGORICAL[0], STATUS_WARNING, STATUS_CRITICAL]
        pol_bars = alt.Chart(pol_long).mark_bar().encode(
            x=alt.X("week:T", title="Week"), y=alt.Y("pct:Q", title="% of time", stack="zero"),
            color=alt.Color("Zone:N", title=None, scale=alt.Scale(domain=zone_order, range=zone_colors)),
            tooltip=[alt.Tooltip("week:T", title="Week"), alt.Tooltip("Zone:N"), alt.Tooltip("pct:Q", title="%", format=".0f")],
        )
        pol_target = alt.Chart(pd.DataFrame({"y": [80]})).mark_rule(strokeDash=[4, 3], color=INK).encode(y="y:Q")
        charts["polarization"] = (pol_bars + pol_target).properties(height=260, width=CHART_WIDTH)

    # -- Tables / stats --
    fastest_idx = acts["avg_pace_min_per_km"].idxmin()
    longest_idx = acts["distance_km"].idxmax()
    vo2_idx = daily["vo2max.value"].idxmax() if daily["vo2max.value"].notna().any() else None
    personal_records = {
        "fastest_pace": format_pace(acts.loc[fastest_idx, "avg_pace_min_per_km"]),
        "fastest_date": acts.loc[fastest_idx, "date"].strftime("%b %d, %Y"),
        "longest_km": f"{acts.loc[longest_idx, 'distance_km']:.1f} km",
        "longest_date": acts.loc[longest_idx, "date"].strftime("%b %d, %Y"),
        "highest_vo2max": f"{daily.loc[vo2_idx, 'vo2max.value']:.1f}" if vo2_idx is not None else "--",
        "vo2max_date": vo2_idx.strftime("%b %d, %Y") if vo2_idx is not None else "",
    }

    distance_order = ["5k", "10k", "half", "marathon"]
    distance_names = {"5k": "5 km", "10k": "10 km", "half": "Half Marathon", "marathon": "Marathon"}
    garmin_cols = {"5k": "race_predictions.time_5k_sec", "10k": "race_predictions.time_10k_sec",
                   "half": "race_predictions.time_half_sec", "marathon": "race_predictions.time_marathon_sec"}
    latest = daily.iloc[-1]
    predictor_rows = []
    for label in distance_order:
        predictor_rows.append({
            "Distance": distance_names[label],
            "Garmin": format_duration(latest.get(garmin_cols[label])),
            "Riegel": format_duration(riegel.get(label)),
            "Cameron": format_duration(cameron.get(label)),
            "Daniels/VDOT": format_duration(daniels.get(label)),
        })

    default_dist = "5k"
    this_week_be, all_time_be = A.this_week_vs_alltime_best(acts, default_dist)
    board = A.best_efforts_leaderboard(acts, default_dist, top_n=8).copy()
    board["Time"] = board["time_sec"].apply(format_duration)
    board["Date"] = board["date"].dt.strftime("%b %d, %Y")
    board["Within"] = board.apply(lambda r: f"{r['run_type'] or 'Run'} · {r['distance_km']:.1f} km", axis=1)

    recent_runs = acts.sort_values("date", ascending=False).head(20).copy()
    recent_runs["Pace"] = recent_runs["avg_pace_min_per_km"].apply(format_pace)
    recent_runs["Date"] = recent_runs["date"].dt.strftime("%Y-%m-%d")

    # -- Status chips (Load tab, shown by default) --
    latest_tsb = pmc["tsb"].iloc[-1] if not pmc.empty else None
    latest_acwr = acwr.iloc[-1] if not acwr.empty else None
    this_week_km = weekly_periods.iloc[-1] if not weekly_periods.empty else None
    runs_this_week = weekly_counts.iloc[-1] if not weekly_counts.empty else None
    streak = compute_streak(weekly_counts, DEMO_WEEKLY_RUN_TARGET)

    tsb_sev = A.classify_tsb(latest_tsb)
    acwr_sev = A.classify_acwr(latest_acwr)
    chips = [
        chip("Form (TSB)", f"{latest_tsb:+.1f}", {"good": "Fresh — absorbing load well", "warning": "Building fatigue", "critical": "High fatigue — recover"}.get(tsb_sev, "--"), tsb_sev),
        chip("Load Ratio (ACWR)", f"{latest_acwr:.2f}", {"good": "In the sweet spot", "warning": "Outside 0.8–1.3", "critical": "High injury-risk zone"}.get(acwr_sev, "--"), acwr_sev),
        chip("This Week", f'{this_week_km:.1f}<small>km</small>', f"{int(runs_this_week)} runs · long run {long_run_this_week:.1f} km", "accent"),
        chip("Streak", f"{streak}<small>wk</small>", f"weeks hitting {DEMO_WEEKLY_RUN_TARGET}+ runs", "accent" if streak else None),
    ]

    # -- Race clock hero --
    days_to_go = (DEMO_RACE_DATE - acts["date"].max().date()).days
    phase = training_phase(days_to_go)
    block_start = DEMO_RACE_DATE - timedelta(weeks=DEMO_PHASE_LOOKBACK_WEEKS)
    total_span = max((DEMO_RACE_DATE - block_start).days, 1)
    elapsed = min(max((acts["date"].max().date() - block_start).days, 0), total_span)
    pct_now = elapsed / total_span * 100
    marks_days = [PHASE_BUILD_DAYS, PHASE_PEAK_DAYS, PHASE_TAPER_DAYS]
    marks_pct = [max(0, min(100, (total_span - d) / total_span * 100)) for d in marks_days if d < total_span]
    marks_html = "".join(f'<div class="rd-rail-mark" style="left:{p:.1f}%"></div>' for p in marks_pct)
    labels = ["Base", "Build", "Peak", "Taper"]
    labels_html = "".join(f'<span class="{"on" if labels[i] == phase else ""}">{labels[i]}</span>' for i in range(4))
    goal_pace = format_pace(DEMO_GOAL_MARATHON_SEC / 60 / (42195 / 1000))
    race_clock_html = f"""
    <div class="rd-clock">
      <div class="rd-clock-top">
        <div><span class="rd-eyebrow">Time to start line</span><div class="rd-days">{days_to_go}<sup>days</sup></div></div>
        <div class="rd-target">
          <div class="name">Race day</div>
          <div class="sub">{DEMO_RACE_DATE.strftime("%a %d %b %Y")}</div>
          <div class="goal">Goal {seconds_to_hms(DEMO_GOAL_MARATHON_SEC)} · {goal_pace}</div>
        </div>
      </div>
      <div class="rd-rail-track">
        <div class="rd-rail-fill" style="width:{pct_now:.1f}%"></div>
        {marks_html}
        <div class="rd-rail-now" style="left:{pct_now:.1f}%"></div>
      </div>
      <div class="rd-rail-labels">{labels_html}</div>
    </div>
    """

    html = render_html(
        charts=charts, chips=chips, race_clock_html=race_clock_html,
        personal_records=personal_records, predictor_rows=predictor_rows,
        this_week_be=this_week_be, all_time_be=all_time_be, board=board,
        recent_runs=recent_runs,
    )
    OUT_FILE.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT_FILE}")


def chip(key, value_html, status_html, sev=None):
    color = SEV_COLOR.get(sev, INK_3)
    return f"""<div class="rd-chip" style="--sev:{color}">
      <span class="k">{key}</span>
      <span class="v">{value_html}</span>
      <span class="s"><span class="rd-dot"></span>{status_html}</span>
    </div>"""


def chart_div(chart_id, chart):
    if chart is None:
        return f'<p class="rd-empty">No data for this metric.</p>'
    spec_json = chart.to_json()
    return f"""<div id="{chart_id}" class="rd-vega"></div>
    <script>
      vegaEmbed("#{chart_id}", {spec_json}, {{actions: false, renderer: "svg"}});
    </script>"""


def table_html(df: pd.DataFrame, columns: list[str]):
    if df.empty:
        return '<p class="rd-empty">No data.</p>'
    head = "".join(f"<th>{c}</th>" for c in columns)
    rows = ""
    for _, row in df.iterrows():
        rows += "<tr>" + "".join(f"<td>{row[c]}</td>" for c in columns) + "</tr>"
    return f'<table class="rd-table"><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>'


def render_html(charts, chips, race_clock_html, personal_records, predictor_rows,
                 this_week_be, all_time_be, board, recent_runs):
    predictor_table = table_html(pd.DataFrame(predictor_rows), ["Distance", "Garmin", "Riegel", "Cameron", "Daniels/VDOT"])
    leaderboard_table = table_html(board, ["Time", "Within", "Date"])
    recent_runs_table = table_html(
        recent_runs.rename(columns={"run_type": "Type", "distance_km": "Distance (km)", "avg_hr": "Avg HR"}),
        ["Date", "Type", "Distance (km)", "Pace", "Avg HR"],
    )

    def fmt_sec(v):
        return format_duration(v) if v else "no effort"

    tabs = [
        ("Load", [
            ("Fitness, Fatigue & Form", "CTL (fitness), ATL (fatigue), TSB (form) — the standard Performance Manager Chart model.", chart_div("pmc", charts["pmc"])),
            ("Load Ratio (ACWR)", "Acute:chronic workload ratio — 0.8–1.3 is the published injury-risk sweet spot.", chart_div("acwr", charts["acwr"])),
            ("Weekly Mileage", "", chart_div("weekly_mileage", charts["weekly_mileage"])),
            ("Monthly Mileage", "", chart_div("monthly_mileage", charts["monthly_mileage"])),
            ("Weekly Run Count", "", chart_div("weekly_run_count", charts["weekly_run_count"])),
        ]),
        ("Fitness", [
            ("VO2max Trend", "", chart_div("vo2max", charts["vo2max"])),
            ("Race Predictor: Marathon & Half Marathon", "", chart_div("race_predictor_long", charts["race_predictor_long"])),
            ("Race Predictor: 5K & 10K", "", chart_div("race_predictor_short", charts["race_predictor_short"])),
            ("Race Predictor Comparison", "Riegel and Cameron extrapolate from your own best effort; Daniels/VDOT uses Garmin's measured VO2max.", predictor_table),
            ("Heat-Adjusted Pace", "A directional correction (~0.5%/°C above 15°C), not a validated physiological model.", chart_div("heat_adjusted", charts["heat_adjusted"])),
            ("Efficiency Factor", "Speed (m/s) per heartbeat — rising means more speed for the same effort.", chart_div("efficiency_factor", charts["efficiency_factor"])),
            ("Personal Records", "", f"""
                <div class="rd-pr-grid">
                  <div><span class="k">Fastest Pace</span><span class="v">{personal_records['fastest_pace']}</span><span class="s">{personal_records['fastest_date']}</span></div>
                  <div><span class="k">Longest Run</span><span class="v">{personal_records['longest_km']}</span><span class="s">{personal_records['longest_date']}</span></div>
                  <div><span class="k">Highest VO2max</span><span class="v">{personal_records['highest_vo2max']}</span><span class="s">{personal_records['vo2max_date']}</span></div>
                </div>"""),
        ]),
        ("Best Efforts", [
            ("Best Efforts — 5 km", "Fastest continuous segment found anywhere inside any run, not Garmin's own lap markers.", f"""
                <div class="rd-pr-grid">
                  <div><span class="k">This week's best 5 km</span><span class="v">{fmt_sec(this_week_be)}</span></div>
                  <div><span class="k">All-time best 5 km</span><span class="v">{fmt_sec(all_time_be)}</span></div>
                </div>
                <p class="rd-hint" style="margin-top:14px">Leaderboard</p>
                {leaderboard_table}"""),
        ]),
        ("Recovery", [
            ("HRV & Resting Heart Rate (last 8 weeks)", "", chart_div("hrv_rhr", charts["hrv_rhr"])),
            ("Sleep Duration vs 7-Day Rolling Average", "", chart_div("sleep", charts["sleep"])),
            ("Stress Trend", "", chart_div("stress", charts["stress"])),
            ("Training Polarization", "The 80/20 rule: most training time should be genuinely easy (Z1–2).", chart_div("polarization", charts["polarization"])),
        ]),
        ("Runs", [
            ("Recent Runs", "", recent_runs_table),
            ("Pace vs Heart Rate (Aerobic Decoupling)", "Same pace at a lower heart rate over time suggests improving aerobic fitness.", chart_div("pace_hr_scatter", charts["pace_hr_scatter"])),
        ]),
    ]

    tab_buttons = "".join(
        f'<button class="rd-tabbtn{" active" if i == 0 else ""}" data-tab="tab-{i}">{name}</button>'
        for i, (name, _) in enumerate(tabs)
    )
    tab_panels = ""
    for i, (name, sections) in enumerate(tabs):
        sections_html = ""
        for title, desc, body in sections:
            desc_html = f'<p class="rd-hint">{desc}</p>' if desc else ""
            sections_html += f'<div class="rd-section"><h3>{title}</h3>{desc_html}{body}</div>'
        tab_panels += f'<div class="rd-tabpanel{" active" if i == 0 else ""}" id="tab-{i}">{sections_html}</div>'

    og_image = f"{REPO_URL.replace('github.com', 'raw.githubusercontent.com')}/main/docs/og-image.png"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{SITE_TITLE}</title>
<meta name="description" content="{SITE_DESCRIPTION}">

<meta property="og:type" content="website">
<meta property="og:title" content="{SITE_TITLE}">
<meta property="og:description" content="{SITE_DESCRIPTION}">
<meta property="og:url" content="{REPO_URL}">
<meta property="og:image" content="{og_image}">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{SITE_TITLE}">
<meta name="twitter:description" content="{SITE_DESCRIPTION}">
<meta name="twitter:image" content="{og_image}">

<script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>

<style>
  :root {{ color-scheme: light; }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: {GROUND}; color: {INK};
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }}
  .rd-banner {{
    background: {ACCENT}; color: #fff; text-align: center; padding: 10px 16px;
    font-size: 13px; font-weight: 600; letter-spacing: .01em;
  }}
  .rd-wrap {{ max-width: 1080px; margin: 0 auto; padding: 28px 20px 80px; }}
  h1 {{ font-size: 26px; margin: 4px 0 20px; }}
  .rd-num, [class*="rd-"] {{ font-variant-numeric: tabular-nums; }}
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
  .rd-chips {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px,1fr)); gap: 10px; margin-bottom: 24px; }}
  .rd-chip {{ background: {SURFACE}; border: 1px solid {RULE}; border-radius: 4px; padding: 12px 14px; position: relative; overflow: hidden; }}
  .rd-chip::before {{ content: ""; position: absolute; inset: 0 auto 0 0; width: 2px; background: var(--sev); }}
  .rd-chip .k {{ font-family: {MONO}; font-size: 10px; letter-spacing: .12em; text-transform: uppercase; color: {INK_3}; display: block; }}
  .rd-chip .v {{ font-family: {MONO}; font-size: 22px; font-weight: 600; color: {INK}; }}
  .rd-chip .v small {{ font-size: .5em; font-weight: 400; color: {INK_3}; margin-left: 3px; }}
  .rd-chip .s {{ font-size: 11.5px; color: {INK_2}; display: flex; align-items: center; gap: 5px; margin-top: 2px; }}
  .rd-dot {{ width: 6px; height: 6px; border-radius: 50%; background: var(--sev); flex: none; }}
  .rd-tabs {{ display: flex; gap: 4px; margin-bottom: 20px; flex-wrap: wrap; border-bottom: 1px solid {RULE}; }}
  .rd-tabbtn {{
    font-family: inherit; font-size: 13px; font-weight: 600; color: {INK_2}; background: none;
    border: none; border-bottom: 2px solid transparent; padding: 10px 14px; cursor: pointer;
  }}
  .rd-tabbtn.active {{ color: {ACCENT}; border-bottom-color: {ACCENT}; }}
  .rd-tabpanel {{ display: none; }}
  .rd-tabpanel.active {{ display: block; }}
  .rd-section {{ background: {SURFACE}; border: 1px solid {RULE}; border-radius: 4px; padding: 18px 20px; margin-bottom: 16px; }}
  .rd-section h3 {{ margin: 0 0 6px; font-size: 15px; }}
  .rd-hint {{ font-size: 12px; color: {INK_2}; max-width: 72ch; margin: 0 0 12px; }}
  .rd-vega {{ width: 100%; overflow-x: auto; }}
  .rd-empty {{ font-size: 13px; color: {INK_3}; }}
  .rd-pr-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px,1fr)); gap: 14px; }}
  .rd-pr-grid .k {{ font-family: {MONO}; font-size: 10px; letter-spacing: .1em; text-transform: uppercase; color: {INK_3}; display: block; }}
  .rd-pr-grid .v {{ font-family: {MONO}; font-size: 20px; font-weight: 600; display: block; margin: 2px 0; }}
  .rd-pr-grid .s {{ font-size: 11.5px; color: {INK_2}; }}
  table.rd-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  table.rd-table th {{ text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: {INK_3}; padding: 6px 8px; border-bottom: 1px solid {RULE_STRONG}; }}
  table.rd-table td {{ padding: 7px 8px; border-bottom: 1px solid {RULE}; font-family: {MONO}; }}
  footer {{ text-align: center; font-size: 12px; color: {INK_3}; padding: 30px 20px; }}
  footer a {{ color: {ACCENT}; }}
</style>
</head>
<body>

<div class="rd-banner">Live demo — sample data, not a real athlete's activity. <a href="{REPO_URL}" style="color:#fff">View source →</a></div>

<div class="rd-wrap">
  <h1>Running Dashboard</h1>
  {race_clock_html}
  <div class="rd-chips">{"".join(chips)}</div>
  <div class="rd-tabs">{tab_buttons}</div>
  {tab_panels}
</div>

<footer>
  Built with Streamlit, Altair, and the (unofficial) Garmin Connect API — <a href="{REPO_URL}">source on GitHub</a>.
  This page is a static export of a locally-run Streamlit app, generated from synthetic sample data.
</footer>

<script>
  document.querySelectorAll(".rd-tabbtn").forEach(btn => {{
    btn.addEventListener("click", () => {{
      document.querySelectorAll(".rd-tabbtn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".rd-tabpanel").forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(btn.dataset.tab).classList.add("active");
    }});
  }});
</script>

</body>
</html>
"""


if __name__ == "__main__":
    main()
