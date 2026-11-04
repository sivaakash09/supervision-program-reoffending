"""Part A: predict re-arrest within one year of release, then check the model for unequal errors.

Models (trained on NIJ's official training split, scored on its held-out test split):
  * the state's own risk score alone (the existing tool, as a reference point)
  * logistic regression: simple and explainable; every input has one readable weight
  * XGBoost: gradient-boosted trees, more flexible

Race and gender are NOT model inputs. They are used only afterwards, to audit whether the model makes
more mistakes for some groups than others.

Run:  python src/predict_and_audit.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from features import OUTCOME, build_features, load_raw

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
FIG = REP / "figures"
N_BOOT = 1000
rng = np.random.default_rng(7)


def boot_ci(y: np.ndarray, p: np.ndarray, fn) -> tuple[float, float]:
    vals = []
    n = len(y)
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        vals.append(fn(y[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def group_errors(y: np.ndarray, p: np.ndarray, flag: np.ndarray, groups: pd.Series) -> pd.DataFrame:
    rows = []
    for g in sorted(groups.unique()):
        m = (groups == g).values
        yt, ft, pt = y[m], flag[m], p[m]
        rows.append({
            "group": g, "people": int(m.sum()),
            "actual_rearrest_rate": round(yt.mean() * 100, 1),
            "avg_predicted_risk": round(pt.mean() * 100, 1),
            "flagged_high_risk_pct": round(ft.mean() * 100, 1),
            # false positive rate: flagged high risk but NOT re-arrested
            "false_positive_rate": round(ft[yt == 0].mean() * 100, 1),
            # false negative rate: NOT flagged but re-arrested
            "false_negative_rate": round((1 - ft[yt == 1]).mean() * 100, 1),
            # of those flagged, how many were re-arrested
            "precision_of_flag": round(yt[ft == 1].mean() * 100, 1),
        })
    return pd.DataFrame(rows)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_raw()
    X = build_features(df, include_demographics=False)
    y = df[OUTCOME].astype(int).values
    train = (df["Training_Sample"] == 1).values
    test = ~train
    print(f"train: {train.sum():,} people   test: {test.sum():,} people   base rate: {y.mean():.1%}")

    logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    logit.fit(X[train], y[train])
    xgb = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8,
                        colsample_bytree=0.8, min_child_weight=5, eval_metric="logloss", random_state=7)
    xgb.fit(X[train], y[train])

    preds = {
        "State risk score only": X.loc[test, "state_risk_score"].values / 10,  # ranking only (AUC)
        "Logistic regression": logit.predict_proba(X[test])[:, 1],
        "XGBoost": xgb.predict_proba(X[test])[:, 1],
    }
    yt = y[test]
    results = {}
    for name, p in preds.items():
        auc = roc_auc_score(yt, p)
        lo, hi = boot_ci(yt, p, roc_auc_score)
        r = {"auc": round(auc, 3), "auc_95ci": [round(lo, 3), round(hi, 3)]}
        if name != "State risk score only":
            r["brier"] = round(brier_score_loss(yt, p), 4)
        results[name] = r
        print(f"  {name:22s} AUC {auc:.3f} [{lo:.3f}, {hi:.3f}]" + (f"  Brier {r['brier']:.4f}" if "brier" in r else ""))
    results["base_rate_test"] = round(float(yt.mean()), 4)
    results["brier_if_everyone_gets_base_rate"] = round(brier_score_loss(yt, np.full_like(yt, yt.mean(), dtype=float)), 4)

    # ---- top drivers in the logistic model (standardized weights)
    coefs = pd.Series(logit[-1].coef_[0], index=X.columns).sort_values(key=np.abs, ascending=False)
    coefs.head(12).round(3).to_csv(REP / "logistic_top_weights.csv", header=["standardized_weight"])

    # ---- fairness audit on the simpler model (the one a public agency could explain)
    p = preds["Logistic regression"]
    cut = np.quantile(p, 1 - yt.mean())  # flag the same share as actually re-offend (~30%)
    flag = (p >= cut).astype(int)
    race = df.loc[test, "Race"].str.title().reset_index(drop=True)
    sex = df.loc[test, "Gender"].map({"M": "Men", "F": "Women"}).reset_index(drop=True)
    audit = pd.concat([group_errors(yt, p, flag, race).assign(attribute="race"),
                       group_errors(yt, p, flag, sex).assign(attribute="sex")])
    audit.to_csv(REP / "fairness_audit.csv", index=False)
    print(audit.to_string(index=False))
    results["flag_threshold"] = round(float(cut), 3)
    (REP / "prediction_results.json").write_text(json.dumps(results, indent=2))

    # ---- figures: calibration by race
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for g in ["Black", "White"]:
        m = (race == g).values
        frac, mean_p = calibration_curve(yt[m], p[m], n_bins=8, strategy="quantile")
        axes[0].plot(mean_p, frac, marker="o", label=g)
    axes[0].plot([0, 0.8], [0, 0.8], color="grey", ls="--", lw=0.8)
    axes[0].set_xlabel("Predicted chance of re-arrest"); axes[0].set_ylabel("Actual share re-arrested")
    axes[0].set_title("Calibration: does 30% risk mean 30%?"); axes[0].legend()
    a = audit[audit.attribute == "race"]
    x = np.arange(len(a)); w = 0.35
    axes[1].bar(x - w / 2, a["false_positive_rate"], w, label="Flagged, but not re-arrested")
    axes[1].bar(x + w / 2, a["false_negative_rate"], w, label="Not flagged, but re-arrested")
    axes[1].set_xticks(x, a["group"]); axes[1].set_ylabel("% of group"); axes[1].set_ylim(0, 70)
    axes[1].legend(fontsize=8, loc="upper center", ncol=2)
    axes[1].set_title("Error rates by race (logistic model)")
    fig.tight_layout(); fig.savefig(FIG / "fairness_audit.png", dpi=150); plt.close(fig)

    # tidy table for dashboards / BI
    pd.DataFrame({"race": race, "sex": sex, "predicted_risk": p.round(4), "flagged": flag,
                  "rearrested_year1": yt}).to_csv(REP / "for_bi" / "test_predictions.csv", index=False)


if __name__ == "__main__":
    main()
