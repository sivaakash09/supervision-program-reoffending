"""Part C: turn the results into things people can read.

  reports/white_paper.pdf    the white paper (from white_paper.md), for a policy audience
  reports/dashboard.html     interactive dashboard (open in any browser)
  reports/for_bi/*.csv       tidy tables for Power BI or Tableau

PDF needs pandoc and Playwright's Chromium (`pip install playwright && playwright install chromium`).
If they are missing, the script still builds the dashboard and tells you how to make the PDF.

Run:  python src/build_report.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"

CSS = """
body { font-family: Georgia, 'Times New Roman', serif; max-width: 760px; margin: 0 auto; color: #1a1a1a;
       line-height: 1.5; font-size: 11pt; }
h1 { font-size: 22pt; margin-bottom: 0; } .subtitle { font-size: 13pt; color: #444; margin-top: 4px; }
.author { color: #666; margin-bottom: 18px; }
h2 { font-size: 14pt; border-bottom: 1px solid #ccc; padding-bottom: 3px; margin-top: 22px; }
table { border-collapse: collapse; margin: 10px 0; font-size: 9.5pt; width: 100%; }
th, td { border-bottom: 1px solid #ddd; padding: 4px 8px; } th { background: #f3f3f3; text-align: left; }
img { width: 100%; margin: 8px 0; } em { color: #444; } code { font-size: 9pt; }
"""


def build_pdf() -> None:
    md, html = REP / "white_paper.md", REP / "white_paper.html"
    if not shutil.which("pandoc"):
        print("pandoc not found: skipping PDF (install pandoc to build reports/white_paper.pdf)")
        return
    css = REP / "_paper.css"
    css.write_text(CSS)
    subprocess.run(["pandoc", str(md), "-s", "--embed-resources", "--css", str(css), "-o", str(html),
                    "--resource-path", str(REP), "--metadata", "lang=en"], check=True)
    css.unlink()
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch()
            page = b.new_page()
            page.goto(html.resolve().as_uri())
            page.pdf(path=str(REP / "white_paper.pdf"), format="Letter",
                     margin={"top": "0.7in", "bottom": "0.7in", "left": "0.7in", "right": "0.7in"})
            b.close()
        html.unlink()
        print("saved reports/white_paper.pdf")
    except Exception as exc:
        print(f"could not print PDF ({exc}); open reports/white_paper.html in a browser and print to PDF instead")


def build_dashboard() -> None:
    pred = json.loads((REP / "prediction_results.json").read_text())
    eff = json.loads((REP / "program_effect_results.json").read_text())
    audit = pd.read_csv(REP / "fairness_audit.csv")
    tp = pd.read_csv(REP / "for_bi" / "test_predictions.csv")

    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.16, horizontal_spacing=0.1, subplot_titles=(
        "How well each model ranks risk (AUC, 95% CI)", "Program assigned vs similar people not assigned",
        "Error rates by group (logistic model)", "Predicted vs actual re-arrest, by race"))

    models = ["State risk score only", "Logistic regression", "XGBoost"]
    fig.add_trace(go.Bar(x=models, y=[pred[m]["auc"] for m in models], marker_color=["#bbbbbb", "#2166ac", "#67a9cf"],
                         error_y=dict(type="data", symmetric=False,
                                      array=[pred[m]["auc_95ci"][1] - pred[m]["auc"] for m in models],
                                      arrayminus=[pred[m]["auc"] - pred[m]["auc_95ci"][0] for m in models]),
                         showlegend=False), 1, 1)
    fig.update_yaxes(range=[0.5, 0.75], row=1, col=1)

    names = {"Condition_Cog_Ed": "Cognitive / education", "Condition_MH_SA": "Mental health / substance"}
    for k, r in eff.items():
        fig.add_trace(go.Bar(name=f"{names[k]}: raw", x=[names[k]], y=[r["raw_difference_pts"]],
                             marker_color="#f4a582", showlegend=False), 1, 2)
        fig.add_trace(go.Bar(name=f"{names[k]}: matched", x=[names[k]], y=[r["att_matching_pts"]],
                             marker_color="#b2182b", showlegend=False,
                             error_y=dict(type="data", symmetric=False,
                                          array=[r["att_matching_95ci"][1] - r["att_matching_pts"]],
                                          arrayminus=[r["att_matching_pts"] - r["att_matching_95ci"][0]])), 1, 2)
    fig.update_yaxes(title_text="Difference (pct. points)<br>light = raw, dark = matched", row=1, col=2)

    fig.add_trace(go.Bar(x=audit["group"], y=audit["false_positive_rate"], name="Flagged, not re-arrested",
                         marker_color="#4393c3"), 2, 1)
    fig.add_trace(go.Bar(x=audit["group"], y=audit["false_negative_rate"], name="Re-arrested, not flagged",
                         marker_color="#d6604d"), 2, 1)

    tp["bin"] = pd.qcut(tp["predicted_risk"], 8, labels=False)
    for race, color in [("Black", "#1b7837"), ("White", "#762a83")]:
        g = tp[tp.race == race].groupby("bin").agg(pred=("predicted_risk", "mean"), act=("rearrested_year1", "mean"))
        fig.add_trace(go.Scatter(x=g["pred"], y=g["act"], mode="lines+markers", name=race, line=dict(color=color)), 2, 2)
    fig.add_trace(go.Scatter(x=[0, 0.7], y=[0, 0.7], mode="lines", line=dict(dash="dash", color="grey"),
                             name="Perfect calibration"), 2, 2)
    fig.update_xaxes(title_text="Predicted chance", row=2, col=2)
    fig.update_yaxes(title_text="Actual share re-arrested", row=2, col=2)
    fig.update_layout(title="Supervision Programs & Re-Arrest: NIJ Georgia data (25,835 people)",
                      template="plotly_white", height=820, barmode="group")
    fig.write_html(str(REP / "dashboard.html"), include_plotlyjs=True)
    print("saved reports/dashboard.html")

    # tidy tables for BI tools
    pd.DataFrame([{"model": m, **{k: v for k, v in pred[m].items() if k != "auc_95ci"},
                   "auc_low": pred[m]["auc_95ci"][0], "auc_high": pred[m]["auc_95ci"][1]} for m in models]) \
        .to_csv(REP / "for_bi" / "model_comparison.csv", index=False)
    audit.to_csv(REP / "for_bi" / "fairness_audit.csv", index=False)
    pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, list)} for r in eff.values()]) \
        .to_csv(REP / "for_bi" / "program_effects.csv", index=False)


def main() -> None:
    build_dashboard()
    build_pdf()


if __name__ == "__main__":
    main()
