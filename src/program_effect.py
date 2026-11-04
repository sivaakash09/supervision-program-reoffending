"""Part B: did people assigned to cognitive-skills / education programming get re-arrested less?

"Treatment": Condition_Cog_Ed, a parole condition that requires cognitive-skills or education programming.
Outcome:     re-arrest within one year of release.

The problem: the condition is not assigned at random. Officers give it to people with more needs, so a
raw comparison mixes the program's effect with who was sent to it. We use two standard ways to compare
like with like, and check that they agree:

  1. Propensity score matching: estimate each person's chance of being assigned the condition from what was
     known at release, then pair every assigned person with the most similar unassigned person
     (1-to-1 nearest neighbour, with replacement, caliper 0.2 SD of the logit). Compare re-arrest rates
     within pairs. Effect = average effect on those who were assigned (ATT).
  2. Regression adjustment: model re-arrest from treatment + all covariates, then compare predicted rates
     with the condition switched on and off for the assigned group.
95% confidence intervals come from 1,000 bootstrap resamples.

Same analysis is repeated for Condition_MH_SA (mental health / substance abuse programming) as a comparison.

Run:  python src/program_effect.py   (about 1 minute; prints progress)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from features import CONDITIONS, OUTCOME, build_features, load_raw

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
FIG = REP / "figures"
N_BOOT = 1000
CALIPER_SD = 0.2


def smd(x: np.ndarray, t: np.ndarray, w_c: np.ndarray | None = None) -> float:
    """Standardized mean difference between treated and (weighted) controls."""
    xt, xc = x[t == 1], x[t == 0]
    wc = np.ones_like(xc) if w_c is None else w_c
    mt, mc = xt.mean(), np.average(xc, weights=wc)
    sd = np.sqrt((xt.var() + x[t == 0].var()) / 2)
    return 0.0 if sd == 0 else (mt - mc) / sd


def match(logit_ps: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (treated indices kept, matched control index for each)."""
    caliper = CALIPER_SD * logit_ps.std()
    ti, ci = np.where(t == 1)[0], np.where(t == 0)[0]
    nn = NearestNeighbors(n_neighbors=1).fit(logit_ps[ci].reshape(-1, 1))
    dist, idx = nn.kneighbors(logit_ps[ti].reshape(-1, 1))
    keep = dist[:, 0] <= caliper
    return ti[keep], ci[idx[keep, 0]]


def analyse(df: pd.DataFrame, treatment: str, rng: np.random.Generator) -> dict:
    t = df[treatment].astype(str).str.lower().eq("true").astype(int).values
    y = df[OUTCOME].astype(int).values
    others = [c.lower() for c in CONDITIONS if c != treatment]
    X = build_features(df, include_demographics=True, include_conditions=False)
    for c in others:
        X[c] = df[[k for k in CONDITIONS if k.lower() == c][0]].astype(str).str.lower().eq("true").astype(float)
    Xs = StandardScaler().fit_transform(X)

    ps_model = LogisticRegression(max_iter=3000, C=1.0).fit(Xs, t)
    ps = ps_model.predict_proba(Xs)[:, 1]
    lps = np.log(ps / (1 - ps))

    raw_diff = y[t == 1].mean() - y[t == 0].mean()
    tr, co = match(lps, t)
    att_match = (y[tr] - y[co]).mean()

    # balance before / after matching
    w = np.bincount(co, minlength=len(t))[t == 0].astype(float)
    bal = pd.DataFrame({
        "covariate": X.columns,
        "smd_before": [smd(X[c].values, t) for c in X.columns],
        "smd_after": [smd(X[c].values, t, w + 1e-12) for c in X.columns],
    })
    max_before, max_after = bal["smd_before"].abs().max(), bal["smd_after"].abs().max()

    # regression adjustment (g-computation) on the treated
    out = LogisticRegression(max_iter=3000, C=1.0).fit(np.column_stack([Xs, t]), y)
    Xt = Xs[t == 1]
    p1 = out.predict_proba(np.column_stack([Xt, np.ones(len(Xt))]))[:, 1].mean()
    p0 = out.predict_proba(np.column_stack([Xt, np.zeros(len(Xt))]))[:, 1].mean()
    att_reg = p1 - p0

    # bootstrap: resample matched pairs for matching; resample people for regression
    boot_m, boot_r = [], []
    pairs = np.arange(len(tr))
    for b in range(N_BOOT):
        i = rng.choice(pairs, len(pairs))
        boot_m.append((y[tr[i]] - y[co[i]]).mean())
        if b < 300:  # regression bootstrap is slower; 300 resamples is plenty for a CI check
            j = rng.integers(0, len(y), len(y))
            m = LogisticRegression(max_iter=3000, C=1.0).fit(np.column_stack([Xs[j], t[j]]), y[j])
            Xtj = Xs[j][t[j] == 1]
            boot_r.append(m.predict_proba(np.column_stack([Xtj, np.ones(len(Xtj))]))[:, 1].mean()
                          - m.predict_proba(np.column_stack([Xtj, np.zeros(len(Xtj))]))[:, 1].mean())
        if (b + 1) % 250 == 0:
            print(f"    {treatment}: bootstrap {b + 1}/{N_BOOT}", flush=True)

    pct = lambda v: round(100 * float(v), 1)
    res = {
        "treatment": treatment,
        "assigned": int(t.sum()), "not_assigned": int((1 - t).sum()),
        "rearrest_rate_assigned": pct(y[t == 1].mean()), "rearrest_rate_not_assigned": pct(y[t == 0].mean()),
        "raw_difference_pts": pct(raw_diff),
        "matched_pairs": int(len(tr)), "assigned_dropped_by_caliper": int(t.sum() - len(tr)),
        "matched_rate_assigned": pct(y[tr].mean()), "matched_rate_comparison": pct(y[co].mean()),
        "att_matching_pts": pct(att_match),
        "att_matching_95ci": [pct(np.percentile(boot_m, 2.5)), pct(np.percentile(boot_m, 97.5))],
        "att_regression_pts": pct(att_reg),
        "att_regression_95ci": [pct(np.percentile(boot_r, 2.5)), pct(np.percentile(boot_r, 97.5))],
        "max_abs_smd_before": round(float(max_before), 3), "max_abs_smd_after": round(float(max_after), 3),
    }
    bal.round(3).to_csv(REP / f"balance_{treatment.lower()}.csv", index=False)
    return res | {"_bal": bal, "_ps": ps, "_t": t}


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_raw()
    rng = np.random.default_rng(11)
    results = {}
    for tr in ["Condition_Cog_Ed", "Condition_MH_SA"]:
        print(f"analysing {tr} ...", flush=True)
        results[tr] = analyse(df, tr, rng)
        r = results[tr]
        print(f"  raw difference {r['raw_difference_pts']:+.1f} pts | matched {r['att_matching_pts']:+.1f} "
              f"{r['att_matching_95ci']} | regression {r['att_regression_pts']:+.1f} {r['att_regression_95ci']} | "
              f"max SMD {r['max_abs_smd_before']} -> {r['max_abs_smd_after']}")

    clean = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in results.items()}
    (REP / "program_effect_results.json").write_text(json.dumps(clean, indent=2))

    # ---- figures
    main_r = results["Condition_Cog_Ed"]
    bal = main_r["_bal"].assign(abs_before=lambda d: d.smd_before.abs()).sort_values("abs_before", ascending=False).head(15)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    yy = np.arange(len(bal))[::-1]
    axes[0].scatter(bal["smd_before"].abs(), yy, label="Before matching", color="#d6604d")
    axes[0].scatter(bal["smd_after"].abs(), yy, label="After matching", color="#2166ac")
    axes[0].axvline(0.1, color="grey", ls="--", lw=0.8)
    axes[0].set_yticks(yy, bal["covariate"].str.replace("_", " "), fontsize=7)
    axes[0].set_xlabel("Difference between groups (standardized)")
    axes[0].set_title("Matching made the groups comparable"); axes[0].legend(fontsize=8)

    labels, est, lo, hi = [], [], [], []
    names = {"Condition_Cog_Ed": "Cognitive / education", "Condition_MH_SA": "Mental health / substance"}
    for k, r in results.items():
        for method, key in [("raw", None), ("matched", "att_matching"), ("regression", "att_regression")]:
            labels.append(f"{names[k]}\n{method}")
            if key is None:
                est.append(r["raw_difference_pts"]); lo.append(np.nan); hi.append(np.nan)
            else:
                est.append(r[f"{key}_pts"]); lo.append(r[f"{key}_95ci"][0]); hi.append(r[f"{key}_95ci"][1])
    est, lo, hi = map(np.array, (est, lo, hi))
    yy = np.arange(len(labels))[::-1]
    err = np.vstack([est - lo, hi - est])
    axes[1].errorbar(est, yy, xerr=np.nan_to_num(err), fmt="o", color="black", capsize=3)
    axes[1].axvline(0, color="grey", lw=0.8)
    axes[1].set_yticks(yy, labels, fontsize=8)
    axes[1].set_xlabel("Difference in 1-year re-arrest rate (percentage points)")
    axes[1].set_title("Assigned vs similar people not assigned")
    fig.tight_layout(); fig.savefig(FIG / "program_effect.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
