"""
Additional / alternative experiments for the project.

Includes:
  (A) Cascading routing: try-cheap-first, escalate-on-failure.
      We model each prompt p as a tuple (m1, m2). Expected cost / score under
      a perfect verifier are computable in closed form per (p, m1, m2). The
      MILP picks one (m1, m2) per prompt + a pool y_m of size <= K, with
      cost-budget. The verifier is treated as zero-cost.

  (B) High-resolution Pareto and K-curve to make the report frontier smooth.

  (C) Score-uncertainty robust BC-2SP via bootstrapped Bernoulli noise.
      Repeat: redraw scores ~ Bernoulli(Q_pm) S times, solve BC-2SP with the
      averaged Q, compare to nominal. Reports policy stability.

  (D) Provider exclusivity matrix (no-OpenAI / no-Anthropic / no-Google /
      must-OpenAI / must-Claude).
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo

import solution as sol  # reuse helpers

OUT = sol.OUT_DIR / "extra"
OUT.mkdir(exist_ok=True, parents=True)
TBL = OUT / "tables"
TBL.mkdir(exist_ok=True)


# --------------------------------------------------------------
# A. Cascade routing
# --------------------------------------------------------------
def solve_cascade_2sp(df, weights, *, B, K=8, allow_no_cascade=True):
    """
    Cascading routing with a perfect zero-cost verifier.
        Stage 1: call m1; verifier observes correctness.
        Stage 2: if m1 was wrong, escalate to m2.

    Expected cost     E[cost | (p, m1, m2)] = C_{p,m1} + (1 - Q_{p,m1}) * C_{p,m2}
    Expected score    E[score | (p, m1, m2)] = 1 - (1 - Q_{p,m1})(1 - Q_{p,m2})

    With binary scores Q in {0,1}, this becomes:
        cost = C_{p,m1} if Q_{p,m1}=1 else C_{p,m1}+C_{p,m2}
        score = 1 if (Q_{p,m1}=1 or Q_{p,m2}=1) else 0
    """
    prompts, models, datasets, Q, C, dom = sol.make_dicts(df)
    storage = {m: sol.estimate_storage_gb(m) for m in models}

    # Pre-compute (p, m1, m2) costs and scores, including no-cascade option (m2 = m1).
    pmm = []  # list of (p, m1, m2)
    score = {}
    cost = {}
    for p in prompts:
        for m1 in models:
            for m2 in models:
                if not allow_no_cascade and m2 == m1:
                    continue
                pmm.append((p, m1, m2))
                q1 = Q[(p, m1)]
                q2 = Q[(p, m2)]
                # Expected cost / score (binary case)
                if q1 >= 0.5:
                    cost[(p, m1, m2)] = C[(p, m1)]
                    score[(p, m1, m2)] = 1.0
                else:
                    cost[(p, m1, m2)] = C[(p, m1)] + C[(p, m2)]
                    score[(p, m1, m2)] = 1.0 if q2 >= 0.5 else 0.0

    mdl = pyo.ConcreteModel()
    mdl.P = pyo.Set(initialize=prompts)
    mdl.M = pyo.Set(initialize=models)
    mdl.PMM = pyo.Set(initialize=pmm, dimen=3)
    mdl.x = pyo.Var(mdl.PMM, within=pyo.Binary)
    mdl.y = pyo.Var(mdl.M, within=pyo.Binary)

    # exactly one (m1, m2) per prompt
    def assign_rule(m, p):
        return sum(m.x[p, a, b] for (q, a, b) in pmm if q == p) == 1
    mdl.assign = pyo.Constraint(mdl.P, rule=assign_rule)

    # x_{p,m1,m2} <= y_{m1}, x_{p,m1,m2} <= y_{m2}
    def link_m1(m, p, a, b):
        return m.x[p, a, b] <= m.y[a]
    def link_m2(m, p, a, b):
        return m.x[p, a, b] <= m.y[b]
    mdl.link1 = pyo.Constraint(mdl.PMM, rule=link_m1)
    mdl.link2 = pyo.Constraint(mdl.PMM, rule=link_m2)

    # pool size cap
    if K is not None:
        mdl.pool_cap = pyo.Constraint(rule=lambda m: sum(m.y[k] for k in m.M) <= K)

    # budget
    if B is not None:
        mdl.budget = pyo.Constraint(
            rule=lambda m: sum(weights[p] * cost[(p, a, b)] * m.x[p, a, b]
                               for (p, a, b) in pmm) <= B
        )

    mdl.obj = pyo.Objective(
        expr=sum(weights[p] * score[(p, a, b)] * mdl.x[p, a, b] for (p, a, b) in pmm),
        sense=pyo.maximize,
    )
    res = sol.get_solver().solve(mdl, tee=False)

    pool = [k for k in models if pyo.value(mdl.y[k]) > 0.5]
    rows = []
    for (p, a, b) in pmm:
        if pyo.value(mdl.x[p, a, b]) > 0.5:
            rows.append({"prompt_id": p, "m1": a, "m2": b, "dataset": dom[p],
                         "weight": weights[p],
                         "exp_score": score[(p, a, b)], "exp_cost": cost[(p, a, b)]})
    a_df = pd.DataFrame(rows)
    avg_cost = float((a_df["weight"] * a_df["exp_cost"]).sum())
    avg_score = float((a_df["weight"] * a_df["exp_score"]).sum())
    pct_escalated = float((a_df["m1"] != a_df["m2"]).mean())
    cascade_used = float(((a_df["m1"] != a_df["m2"]) &
                           (a_df["exp_cost"] > a_df.apply(lambda r: a_df.loc[a_df.index[0], "weight"], axis=1) * 0)).mean())
    return {
        "policy": "Cascade-2SP",
        "B": B, "K": K,
        "selected_pool": pool, "pool_size": len(pool),
        "avg_cost": avg_cost, "avg_score": avg_score,
        "pct_distinct_m1m2": pct_escalated,
        "assignments": a_df,
        "obj_value": pyo.value(mdl.obj),
    }


# --------------------------------------------------------------
# B. High-resolution sweeps
# --------------------------------------------------------------
def hires_pareto(df, weights):
    B_grid = np.geomspace(5e-6, 1e-1, 25).tolist()
    rows = []
    for B in B_grid:
        r = sol.solve_bc2sp(df, weights, B=B, K=8)
        rows.append({"B": B, "avg_cost": r["avg_cost"], "avg_score": r["avg_score"],
                     "pool_size": r["pool_size"], "pool": "|".join(r["selected_pool"])})
    return pd.DataFrame(rows)


def hires_K_curve(df, weights, B=5e-3):
    rows = []
    for K in range(1, 16):
        r = sol.solve_bc2sp(df, weights, B=B, K=K)
        rows.append({"K": K, "avg_cost": r["avg_cost"], "avg_score": r["avg_score"],
                     "pool_size": r["pool_size"], "pool": "|".join(r["selected_pool"])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------
# C. Score-uncertainty robust BC-2SP via bootstrap
# --------------------------------------------------------------
def bootstrap_score_uncertainty(df, weights, B=5e-3, K=8, n_seeds=20, sigma=0.10, seed=42):
    """
    Each Q_pm is the empirical *probability* of correctness; we resample a noisy
    point estimate Q'_pm ~ clip(Q_pm + N(0, sigma), [0,1]) for n_seeds and solve
    BC-2SP on the perturbed data. Compare the chosen pool's frequency (stability).
    """
    rng = np.random.default_rng(seed)
    prompts, models, _, Q, C, dom = sol.make_dicts(df)
    pool_counter = {}
    rows = []
    for s in range(n_seeds):
        df_perturbed = df.copy()
        noise = rng.normal(0, sigma, size=len(df_perturbed))
        df_perturbed["score"] = np.clip(df_perturbed["score"] + noise, 0.0, 1.0)
        r = sol.solve_bc2sp(df_perturbed, weights, B=B, K=K)
        rows.append({
            "seed": s, "avg_score_train": r["avg_score"], "avg_cost_train": r["avg_cost"],
            "pool": "|".join(r["selected_pool"]),
        })
        for m in r["selected_pool"]:
            pool_counter[m] = pool_counter.get(m, 0) + 1

        # Evaluate perturbed-trained pool on nominal data
        # by re-solving routing only with y fixed
        r_eval = sol.solve_bc2sp(df, weights, B=B, K=K,
                                 must_include=r["selected_pool"], must_exclude=[
                                     m for m in models if m not in r["selected_pool"]
                                 ])
        rows[-1]["avg_score_nominal"] = r_eval["avg_score"]
        rows[-1]["avg_cost_nominal"] = r_eval["avg_cost"]

    df_out = pd.DataFrame(rows)
    pool_freq = (pd.Series(pool_counter, name="freq") / n_seeds).sort_values(ascending=False)
    return df_out, pool_freq


# --------------------------------------------------------------
# D. Provider exclusivity matrix
# --------------------------------------------------------------
def provider_matrix(df, weights, B=5e-3, K=8):
    flagship = {
        "OpenAI": ["gpt-5", "gpt-5-chat"],
        "Anthropic": ["claude-sonnet-4"],
        "Google": ["gemini-2.5-pro", "gemini-2.5-flash", "gemma-2-9b-it"],
    }
    rows = []
    for setting in ["free", "no-OpenAI", "no-Anthropic", "no-Google",
                    "must-OpenAI", "must-Claude", "open-source-only"]:
        kw = {}
        if setting == "no-OpenAI":
            kw["must_exclude"] = flagship["OpenAI"]
        elif setting == "no-Anthropic":
            kw["must_exclude"] = flagship["Anthropic"]
        elif setting == "no-Google":
            kw["must_exclude"] = flagship["Google"]
        elif setting == "must-OpenAI":
            kw["must_include"] = ["gpt-5"]
        elif setting == "must-Claude":
            kw["must_include"] = ["claude-sonnet-4"]
        elif setting == "open-source-only":
            # exclude everything with cost > $0.001/prompt average
            costly = (df.groupby("model")["cost"].mean() > 0.0005).pipe(lambda s: s[s].index.tolist())
            kw["must_exclude"] = costly
        try:
            r = sol.solve_bc2sp(df, weights, B=B, K=K, **kw)
            rows.append({"setting": setting, "avg_score": r["avg_score"], "avg_cost": r["avg_cost"],
                         "pool_size": r["pool_size"], "pool": "|".join(r["selected_pool"])})
        except Exception as e:
            rows.append({"setting": setting, "error": str(e)[:80]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------
# Driver
# --------------------------------------------------------------
def main():
    print("=" * 72)
    print("EXTRA EXPERIMENTS")
    print("=" * 72)
    df_raw = sol.load_data()
    df = sol.restrict_to_full_coverage(df_raw)
    weights = sol.make_uniform_weights(sorted(df["prompt_id"].unique()))

    # B. High-res sweeps
    print("\n[ExtraB] high-res Pareto frontier")
    t0 = time.time()
    df_pareto = hires_pareto(df, weights)
    print(df_pareto.to_string()); df_pareto.to_csv(TBL / "hires_frontier_B.csv", index=False)
    print(f"  ({time.time()-t0:.1f}s)")

    print("\n[ExtraB] high-res K curve")
    df_K = hires_K_curve(df, weights, B=5e-3)
    print(df_K.to_string()); df_K.to_csv(TBL / "hires_K_curve.csv", index=False)

    # A. Cascade
    print("\n[ExtraA] Cascade-2SP (sweep B)")
    rows = []
    for B in [5e-5, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]:
        r = solve_cascade_2sp(df, weights, B=B, K=8)
        rows.append({"B": B, "avg_cost": r["avg_cost"], "avg_score": r["avg_score"],
                     "pool_size": r["pool_size"], "pool": "|".join(r["selected_pool"]),
                     "pct_distinct_m1m2": r["pct_distinct_m1m2"]})
        print(f"  B={B:.5g} -> score={r['avg_score']:.4f}, cost={r['avg_cost']:.6f}, pool={len(r['selected_pool'])}, esc%={r['pct_distinct_m1m2']:.2f}")
    df_cascade = pd.DataFrame(rows)
    df_cascade.to_csv(TBL / "cascade_frontier.csv", index=False)

    # D. Provider matrix
    print("\n[ExtraD] Provider exclusivity matrix")
    df_prov = provider_matrix(df, weights, B=5e-3, K=8)
    print(df_prov.to_string()); df_prov.to_csv(TBL / "provider_matrix.csv", index=False)

    # C. Score-uncertainty robust
    print("\n[ExtraC] Score-uncertainty bootstrap (sigma=0.10, n_seeds=15)")
    df_boot, pool_freq = bootstrap_score_uncertainty(df, weights, B=5e-3, K=8, n_seeds=15, sigma=0.10)
    print(df_boot.to_string()); df_boot.to_csv(TBL / "bootstrap_runs.csv", index=False)
    print("\nPool inclusion frequency (across seeds):")
    print(pool_freq.to_string())
    pool_freq.to_csv(TBL / "bootstrap_pool_freq.csv")

    print("\nDONE")


if __name__ == "__main__":
    main()
