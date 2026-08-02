"""Builds docs/slide_deck.html: a 16:9 slide deck (one section per slide)
for capture as a PDF via headless Chrome, for LinkedIn. Reuses the exact
same chart/table data as the web demo (build_demo_site.build_page_data())
so the two outputs can't drift apart.

Unlike the web demo, this has no tabs/JS toggles -- everything renders in a
fixed reading order, one slide per <section>, since nobody can click a tab
inside a PDF. Print sizing follows the standard 16:9 @ 96dpi recipe:
@page { size: 13.333in 7.5in; margin: 0 } == 1280x720px slides.

Run:
    python scripts/build_slide_deck.py
Writes docs/slide_deck.html. Capture the PDF with capture_slide_deck_pdf.py.
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "scripts"))

import build_demo_site as site  # noqa: E402

OUT_FILE = BASE_DIR / "docs" / "slide_deck.html"
REPO_URL = site.REPO_URL


def slide(content_html, slide_id=None):
    id_attr = f' id="{slide_id}"' if slide_id else ""
    return f'<section class="slide"{id_attr}><div class="slide-inner">{content_html}</div></section>'


TWO_COL_CHART_WIDTH = 520  # (1180 slide-inner - 30 gap) / 2 columns, minus margin


def chart_slide_div(chart_id, chart, width=None):
    """Same idea as build_demo_site.chart_div but tracks a ready-promise
    instead of firing vegaEmbed immediately -- collected and awaited before
    the deck signals it's done rendering (see render_deck's READY script).

    `width` overrides the chart's baked-in width (build_page_data() built
    every chart at the demo site's full CHART_WIDTH=900) -- charts placed in
    a two-column slide need a narrower spec or they overflow the column and
    get clipped by the page boundary instead of just looking cramped."""
    if chart is None:
        return '<p class="rd-empty">No data for this metric.</p>', None
    spec = json.loads(chart.to_json())
    if width is not None:
        spec["width"] = width
    script = f'PENDING.push(vegaEmbed("#{chart_id}", {json.dumps(spec)}, {{actions: false, renderer: "svg"}}));'
    return f'<div id="{chart_id}" class="rd-vega"></div>', script


def build():
    data = site.build_page_data()
    charts = data["charts"]

    pending_scripts = []

    def cdiv(chart_id, chart, width=None):
        html, script = chart_slide_div(chart_id, chart, width=width)
        if script:
            pending_scripts.append(script)
        return html

    predictor_table = site.table_html(
        __import__("pandas").DataFrame(data["predictor_rows"]),
        ["Distance", "Garmin", "Riegel", "Cameron", "Daniels/VDOT"],
    )
    leaderboard_table = site.table_html(data["board"], ["Time", "Within", "Date"])

    def fmt_sec(v):
        return site.format_duration(v) if v else "no effort"

    slides_html = []

    # 1. Title slide
    slides_html.append(slide(f"""
        <div class="title-slide">
          <h1>Running Dashboard</h1>
          <p class="subtitle">Fitness, fatigue, and race prediction — computed from your own training data, not just a re-skin of Garmin Connect.</p>
          {data['race_clock_html']}
          <div class="rd-chips">{"".join(data['chips'])}</div>
        </div>
    """))

    # 2. Fitness/Fatigue/Form
    slides_html.append(slide(f"""
        <h2>Fitness, Fatigue &amp; Form</h2>
        <p class="rd-hint">CTL (fitness), ATL (fatigue), TSB (form) — the standard Performance Manager Chart model.</p>
        {cdiv("pmc", charts["pmc"])}
    """))

    # 3. Load ratio
    slides_html.append(slide(f"""
        <h2>Load Ratio (ACWR)</h2>
        <p class="rd-hint">Acute:chronic workload ratio — 0.8–1.3 is the published injury-risk sweet spot.</p>
        {cdiv("acwr", charts["acwr"])}
    """))

    # 4. Mileage (two charts, one slide)
    slides_html.append(slide(f"""
        <h2>Mileage</h2>
        <div class="two-col">
          <div><h3>Weekly</h3>{cdiv("weekly_mileage", charts["weekly_mileage"], width=TWO_COL_CHART_WIDTH)}</div>
          <div><h3>Monthly</h3>{cdiv("monthly_mileage", charts["monthly_mileage"], width=TWO_COL_CHART_WIDTH)}</div>
        </div>
    """))

    # 5. VO2max
    slides_html.append(slide(f"""
        <h2>VO2max Trend</h2>
        {cdiv("vo2max", charts["vo2max"])}
    """))

    # 6. Race predictor charts
    slides_html.append(slide(f"""
        <h2>Race Predictor Trend</h2>
        <div class="two-col">
          <div><h3>Marathon &amp; Half</h3>{cdiv("race_predictor_long", charts["race_predictor_long"], width=TWO_COL_CHART_WIDTH)}</div>
          <div><h3>5K &amp; 10K</h3>{cdiv("race_predictor_short", charts["race_predictor_short"], width=TWO_COL_CHART_WIDTH)}</div>
        </div>
    """))

    # 7. Race predictor comparison table
    slides_html.append(slide(f"""
        <h2>Race Predictor Comparison</h2>
        <p class="rd-hint">Riegel and Cameron extrapolate from your own best effort; Daniels/VDOT uses Garmin's measured VO2max.</p>
        {predictor_table}
    """))

    # 8. Best efforts
    this_week_be, all_time_be = data["this_week_be"], data["all_time_be"]
    slides_html.append(slide(f"""
        <h2>Best Efforts — 5 km</h2>
        <p class="rd-hint">Fastest continuous segment found anywhere inside any run, not Garmin's own lap markers.</p>
        <div class="rd-pr-grid">
          <div><span class="k">This week's best</span><span class="v">{fmt_sec(this_week_be)}</span></div>
          <div><span class="k">All-time best</span><span class="v">{fmt_sec(all_time_be)}</span></div>
        </div>
        <p class="rd-hint" style="margin-top:14px">Leaderboard</p>
        {leaderboard_table}
    """))

    # 9. Recovery: HRV/RHR + Sleep
    slides_html.append(slide(f"""
        <h2>Recovery</h2>
        <div class="two-col">
          <div><h3>HRV &amp; Resting HR</h3>{cdiv("hrv_rhr", charts["hrv_rhr"], width=TWO_COL_CHART_WIDTH)}</div>
          <div><h3>Sleep Duration</h3>{cdiv("sleep", charts["sleep"], width=TWO_COL_CHART_WIDTH)}</div>
        </div>
    """))

    # 10. Training polarization
    slides_html.append(slide(f"""
        <h2>Training Polarization</h2>
        <p class="rd-hint">The 80/20 rule: most training time should be genuinely easy (Z1–2).</p>
        {cdiv("polarization", charts["polarization"])}
    """))

    # 11. Pace vs HR
    slides_html.append(slide(f"""
        <h2>Pace vs Heart Rate (Aerobic Decoupling)</h2>
        <p class="rd-hint">Same pace at a lower heart rate over time suggests improving aerobic fitness.</p>
        {cdiv("pace_hr_scatter", charts["pace_hr_scatter"])}
    """))

    # 12. Closing slide
    slides_html.append(slide(f"""
        <div class="closing-slide">
          <h1>Running Dashboard</h1>
          <p class="subtitle">Open source, self-hosted, runs on your own Garmin data.</p>
          <p class="repo-url">{REPO_URL}</p>
          <p class="disclaimer">All figures in this deck are synthetic sample data, not a real athlete's activity.</p>
        </div>
    """, slide_id="closing"))

    n_charts = len(pending_scripts)
    html = render(slides_html, pending_scripts, n_charts)
    OUT_FILE.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT_FILE} ({len(slides_html)} slides, {n_charts} charts)")


def render(slides_html, pending_scripts, n_charts):
    s = site
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Running Dashboard — Slide Deck</title>
<script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
<style>
  @page {{ size: 13.333in 7.5in; margin: 0; }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; background: {s.GROUND}; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: {s.INK};
  }}
  .slide {{
    width: 1280px; height: 720px; overflow: hidden;
    display: flex; align-items: center; justify-content: center;
    page-break-after: always; position: relative;
    background: {s.GROUND};
  }}
  .slide:last-child {{ page-break-after: auto; }}
  .slide-inner {{
    width: 1180px; max-height: 660px; overflow: hidden;
    transform-origin: center center;
  }}
  h1 {{ font-size: 44px; margin: 0 0 10px; }}
  h2 {{ font-size: 28px; margin: 0 0 6px; }}
  h3 {{ font-size: 16px; margin: 0 0 6px; color: {s.INK_2}; }}
  .subtitle {{ font-size: 17px; color: {s.INK_2}; max-width: 760px; margin: 0 0 22px; }}
  .rd-hint {{ font-size: 13px; color: {s.INK_2}; max-width: 80ch; margin: 0 0 14px; }}
  .rd-vega {{ width: 100%; }}
  .rd-empty {{ font-size: 13px; color: {s.INK_3}; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 30px; align-items: start; }}
  .title-slide, .closing-slide {{ text-align: left; }}
  .closing-slide {{ text-align: center; }}
  .closing-slide .repo-url {{ font-family: {s.MONO}; font-size: 20px; color: {s.ACCENT}; margin-top: 30px; }}
  .closing-slide .disclaimer {{ font-size: 13px; color: {s.INK_3}; margin-top: 14px; }}
  .rd-clock {{ background: {s.SURFACE}; border: 1px solid {s.RULE}; border-radius: 4px; padding: 18px 20px 14px; margin-bottom: 16px; }}
  .rd-clock-top {{ display: flex; align-items: flex-end; justify-content: space-between; gap: 20px; }}
  .rd-eyebrow {{ font-family: {s.MONO}; font-size: 10px; letter-spacing: .14em; text-transform: uppercase; color: {s.INK_3}; display: block; margin-bottom: 4px; }}
  .rd-days {{ font-family: {s.MONO}; font-size: 46px; font-weight: 600; line-height: .9; letter-spacing: -.02em; color: {s.INK}; }}
  .rd-days sup {{ font-size: .28em; font-weight: 500; letter-spacing: .12em; text-transform: uppercase; color: {s.INK_3}; margin-left: 8px; }}
  .rd-target {{ text-align: right; }}
  .rd-target .name {{ font-size: 13px; font-weight: 600; color: {s.INK}; }}
  .rd-target .sub {{ font-family: {s.MONO}; font-size: 11px; color: {s.INK_2}; }}
  .rd-target .goal {{ font-family: {s.MONO}; font-size: 11px; color: {s.ACCENT}; }}
  .rd-rail-track {{ position: relative; height: 3px; background: {s.RULE}; border-radius: 2px; margin-top: 14px; }}
  .rd-rail-fill {{ position: absolute; inset: 0 auto 0 0; background: {s.ACCENT}; border-radius: 2px; }}
  .rd-rail-now {{ position: absolute; top: 50%; width: 9px; height: 9px; border-radius: 50%; background: {s.ACCENT}; border: 2px solid {s.SURFACE}; transform: translate(-50%,-50%); }}
  .rd-rail-mark {{ position: absolute; top: -3px; width: 1px; height: 9px; background: {s.RULE_STRONG}; }}
  .rd-rail-labels {{ display: flex; justify-content: space-between; font-family: {s.MONO}; font-size: 9px; letter-spacing: .1em; text-transform: uppercase; color: {s.INK_3}; margin-top: 6px; }}
  .rd-rail-labels span.on {{ color: {s.ACCENT}; font-weight: 600; }}
  .rd-chips {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }}
  .rd-chip {{ background: {s.SURFACE}; border: 1px solid {s.RULE}; border-radius: 4px; padding: 10px 12px; position: relative; overflow: hidden; }}
  .rd-chip::before {{ content: ""; position: absolute; inset: 0 auto 0 0; width: 2px; background: var(--sev); }}
  .rd-chip .k {{ font-family: {s.MONO}; font-size: 9px; letter-spacing: .12em; text-transform: uppercase; color: {s.INK_3}; display: block; }}
  .rd-chip .v {{ font-family: {s.MONO}; font-size: 19px; font-weight: 600; color: {s.INK}; }}
  .rd-chip .v small {{ font-size: .5em; font-weight: 400; color: {s.INK_3}; margin-left: 3px; }}
  .rd-chip .s {{ font-size: 10.5px; color: {s.INK_2}; display: flex; align-items: center; gap: 5px; margin-top: 2px; }}
  .rd-dot {{ width: 6px; height: 6px; border-radius: 50%; background: var(--sev); flex: none; }}
  .rd-pr-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; margin-bottom: 6px; }}
  .rd-pr-grid .k {{ font-family: {s.MONO}; font-size: 10px; letter-spacing: .1em; text-transform: uppercase; color: {s.INK_3}; display: block; }}
  .rd-pr-grid .v {{ font-family: {s.MONO}; font-size: 22px; font-weight: 600; display: block; }}
  table.rd-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  table.rd-table th {{ text-align: left; font-size: 10px; text-transform: uppercase; letter-spacing: .06em; color: {s.INK_3}; padding: 5px 8px; border-bottom: 1px solid {s.RULE_STRONG}; }}
  table.rd-table td {{ padding: 6px 8px; border-bottom: 1px solid {s.RULE}; font-family: {s.MONO}; }}
</style>
</head>
<body>

{"".join(slides_html)}

<script>
  const PENDING = [];
  {"".join(pending_scripts)}
  Promise.all(PENDING).then(() => {{
    document.title = "SLIDES_READY_{n_charts}";
  }}).catch((err) => {{
    document.title = "SLIDES_ERROR_" + err;
  }});
</script>

</body>
</html>
"""


if __name__ == "__main__":
    build()
