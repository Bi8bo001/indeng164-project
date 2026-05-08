"""
Tight-constraint ablations: at oracle-saturating budgets, slacks/storage rarely
bind. We rerun fairness, storage, and quality-target experiments at *tight*
budget B = 5e-5 (well below the per-prompt cheapest flagship pricing) so the
constraints actively shape the solution. Also a per-prompt fairness scan with
hard tau (no slack) to demonstrate infeasibility regions.
"""

from __future__ import annotations

import time
import pandas as pd
import numpy as np

import solution as sol

OUT = sol.OUT_DIR / "tight"
OUT.mkdir(parents=True, exist_ok=True)
TBL = OUT / "tables"
TBL.mkdir(exist_ok=True)


def main():
    print("=" * 72)
    print("TIGHT-CONSTRAINT ABLATIONS")
    print("=" * 72)
    df = sol.restrict_to_full_coverage(sol.load_data())
    weights = sol.make_uniform_weights(sorted(df["prompt_id"].unique()))

    # ---- Fairness slack at TIGHT budget ----
    print("\n[Tight-1] Fairness slack at B = 5e-5 (very tight)")
    rows = []
    for tau in [None, 0.0, 0.3, 0.5, 0.7, 1.0]:
        for lam in [0.5, 2.0, 10.0]:
            r = sol.solve_bc2sp(df, weights, B=5e-5, K=8,
                                fairness_tau=tau, lambda_slack=lam)
            rows.append({
                "tau": tau if tau is not None else "off",
                "lambda": lam,
                "avg_score": r["avg_score"], "avg_cost": r["avg_cost"],
                "pool_size": r["pool_size"],
                "pool": "|".join(r["selected_pool"]),
            })
        print(f"  tau={tau}: scores={[round(x['avg_score'],4) for x in rows[-3:]]}")
    df_F = pd.DataFrame(rows)
    df_F.to_csv(TBL / "tight_fairness.csv", index=False)
    print(df_F.to_string())

    # ---- Storage at TIGHT budget ----
    print("\n[Tight-2] Storage budget at B = 5e-5 (very tight)")
    rows = []
    for S_max in [None, 100, 50, 20, 0.0]:
        r = sol.solve_bc2sp(df, weights, B=5e-5, K=8, S_max=S_max)
        rows.append({
            "S_max_GB": S_max if S_max is not None else "inf",
            "avg_score": r["avg_score"], "avg_cost": r["avg_cost"],
            "pool_size": r["pool_size"], "pool": "|".join(r["selected_pool"])
        })
    df_S = pd.DataFrame(rows)
    df_S.to_csv(TBL / "tight_storage.csv", index=False)
    print(df_S.to_string())

    # ---- Combined: tight budget + storage + fairness ----
    print("\n[Tight-3] Combined tight (B=2e-4, S_max=50, tau=0.5, sweep K)")
    rows = []
    for K in range(1, 11):
        r = sol.solve_bc2sp(df, weights, B=2e-4, K=K, S_max=50,
                            fairness_tau=0.5, lambda_slack=2.0)
        rows.append({
            "K": K, "avg_score": r["avg_score"], "avg_cost": r["avg_cost"],
            "pool_size": r["pool_size"], "pool": "|".join(r["selected_pool"]),
        })
    df_C = pd.DataFrame(rows)
    df_C.to_csv(TBL / "tight_combined_K.csv", index=False)
    print(df_C.to_string())

    # ---- Show how slack rises as B falls (single dataset target=0.5) ----
    print("\n[Tight-4] Slack sum vs B (tau=0.5, lambda=2.0)")
    rows = []
    for B in [1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]:
        r = sol.solve_bc2sp(df, weights, B=B, K=8, fairness_tau=0.5, lambda_slack=2.0)
        # Recompute slack manually from assignments
        a = r["assignments"]
        a["weighted_score"] = a["weight"] * a["x"] * a["score"]
        per_prompt_score = a.groupby("prompt_id")["weighted_score"].sum() / a.groupby("prompt_id")["weight"].first()
        slack = (0.5 - per_prompt_score).clip(lower=0)
        rows.append({
            "B": B, "avg_score": r["avg_score"], "avg_cost": r["avg_cost"],
            "n_violating_prompts": int((slack > 0).sum()),
            "total_slack": float(slack.sum()),
        })
    df_slack = pd.DataFrame(rows)
    df_slack.to_csv(TBL / "tight_slack_vs_B.csv", index=False)
    print(df_slack.to_string())

    print("\nDONE")


if __name__ == "__main__":
    main()
