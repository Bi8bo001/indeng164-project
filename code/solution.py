"""
INDENG 164 Project: Curating a Robust LLM Pool
==============================================

A two-stage stochastic optimization framework for LLM routing.

Main formulations
-----------------
F1  Budget-Constrained 2SP   (BC-2SP) : max E[score] s.t. E[cost] <= B
F2  Quality-Constrained 2SP  (QC-2SP) : min E[cost]  s.t. E[score] >= Q
F3  Distributionally Robust  (DRO)    : F1 with worst-case prompt mix
F4  Randomized routing                : x in [0,1] (LP relaxation)

All four share:
    - first-stage  y_m in {0,1}   pool-selection (here-and-now)
    - second-stage x_pm in {0,1}  routing       (wait-and-see, indexed by realized prompt)
    - pool-size cap, storage budget, per-prompt fairness slack, optional provider rules

Baselines (single-shot, no first-stage):
    Oracle Routing            (LP lower-bound on cost / upper-bound on score)
    Single Best  (one model for all prompts)
    Single Best per Benchmark (one model per dataset)
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyomo.environ as pyo

# ----------------------------------------------------------------------
# Paths and global config
# ----------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "routerbench.csv"
OUT_DIR = HERE / "solution_outputs"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR = OUT_DIR / "figs"
FIG_DIR.mkdir(exist_ok=True)
TABLE_DIR = OUT_DIR / "tables"
TABLE_DIR.mkdir(exist_ok=True)

SOLVER_NAME = "appsi_highs"


def get_solver():
    s = pyo.SolverFactory(SOLVER_NAME)
    if not s.available(False):
        raise RuntimeError(f"Solver {SOLVER_NAME} not available.")
    return s


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_data(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["dataset"] = df["dataset"].astype(str)
    df["prompt_id"] = df["prompt_id"].astype(str)
    df["model"] = df["model"].astype(str)
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce").fillna(0.0)
    df = df.drop_duplicates(subset=["prompt_id", "model"], keep="first").copy()
    return df


def restrict_to_full_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only models that have a row for every prompt_id in the dataset.

    Some models in routerbench (e.g., deepseek-v3.1-terminus only covers 180 prompts)
    do not cover all 240 prompts. To make every (p,m) pair feasible without dummy
    imputation, we restrict to the maximal subset of models that have full coverage.
    """
    n_p = df["prompt_id"].nunique()
    counts = df.groupby("model")["prompt_id"].nunique()
    keep = counts[counts == n_p].index.tolist()
    dropped = counts[counts < n_p].index.tolist()
    print(f"  Full coverage models kept: {len(keep)}; dropped {len(dropped)}: {dropped}")
    return df[df["model"].isin(keep)].copy()


def make_dicts(df: pd.DataFrame):
    score = {}
    cost = {}
    dataset_of = {}
    for _, r in df.iterrows():
        p = r["prompt_id"]
        m = r["model"]
        score[(p, m)] = float(r["score"])
        cost[(p, m)] = float(r["cost"])
        dataset_of[p] = r["dataset"]
    prompts = sorted(df["prompt_id"].unique().tolist())
    models = sorted(df["model"].unique().tolist())
    datasets = sorted(df["dataset"].unique().tolist())
    return prompts, models, datasets, score, cost, dataset_of


def make_uniform_weights(prompts):
    return {p: 1.0 / len(prompts) for p in prompts}


# ----------------------------------------------------------------------
# Storage parameters (motivated estimates: open-source models occupy GB,
# API-only flagship models 0 GB because they live on the provider side).
# Heuristic by parameter count parsed from the model name.
# ----------------------------------------------------------------------
def estimate_storage_gb(model_name: str) -> float:
    """Approximate disk footprint (FP16) of locally deployed models in GB."""
    name = model_name.lower()
    flagship_api = (
        "gpt-5", "gpt-5-chat", "claude-sonnet-4", "gemini-2.5-pro",
        "gemini-2.5-flash", "glm-4.6",
    )
    if any(k in name for k in flagship_api):
        return 0.0
    # 235B MoE
    if "235b" in name:
        return 470.0  # ~A22B activated, but full weights ~470GB FP16
    # 9B / 8B / 7B
    if any(s in name for s in ("9b", "9-b", "9_b", "-9b")):
        return 18.0
    if any(s in name for s in ("8b", "8-b")):
        return 16.0
    if any(s in name for s in ("7b", "7-b")):
        return 14.0
    if "mini" in name or "nano" in name:
        return 12.0
    # Internlm S1, deepseek family (large hosted but with cost > 0): treat as API
    if "deepseek-v3" in name or "deepseek-r1" in name or "kimi-k2" in name or "intern-s1" == name.split("-")[0]:
        return 0.0
    return 12.0


PROVIDERS = {
    "openai": lambda n: n.startswith("gpt-"),
    "anthropic": lambda n: n.startswith("claude"),
    "google": lambda n: n.startswith("gemini"),
    "deepseek": lambda n: n.startswith("deepseek"),
    "qwen": lambda n: "qwen" in n.lower(),
    "intern": lambda n: "intern" in n.lower(),
    "glm": lambda n: n.lower().startswith("glm"),
    "kimi": lambda n: n.startswith("kimi"),
    "meta": lambda n: "llama" in n.lower(),
    "openthinker": lambda n: "openthinker" in n.lower(),
    "minicpm": lambda n: "minicpm" in n.lower(),
    "nvidia": lambda n: "nemotron" in n.lower() and "nvidia" in n.lower(),
    "ibm": lambda n: "granite" in n.lower(),
    "google_g": lambda n: n.startswith("gemma"),
    "fin": lambda n: n.lower().startswith("fin-"),
    "mimo": lambda n: n.lower().startswith("mimo"),
    "deephermes": lambda n: "deephermes" in n.lower(),
    "cogito": lambda n: "cogito" in n.lower(),
}


def model_provider(name: str) -> str:
    for p, fn in PROVIDERS.items():
        if fn(name):
            return p
    return "other"


# ----------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------
def oracle_routing(df, weights):
    """Per-prompt best score (ties broken by lowest cost). LP-equivalent of unconstrained max E[score]."""
    rows = []
    for p, sub in df.groupby("prompt_id"):
        # Among models with the highest score, pick the cheapest.
        max_s = sub["score"].max()
        cand = sub[sub["score"] == max_s].sort_values("cost").iloc[0]
        rows.append({
            "prompt_id": p,
            "model": cand["model"],
            "dataset": cand["dataset"],
            "weight": weights[p],
            "score": float(cand["score"]),
            "cost": float(cand["cost"]),
        })
    a = pd.DataFrame(rows)
    return {
        "policy": "Oracle (per-prompt best)",
        "avg_cost": float((a["weight"] * a["cost"]).sum()),
        "avg_score": float((a["weight"] * a["score"]).sum()),
        "assignments": a,
    }


def single_best(df, weights):
    rows = []
    for m, sub in df.groupby("model"):
        cm = (sub["prompt_id"].map(weights) * sub["cost"]).sum()
        sm = (sub["prompt_id"].map(weights) * sub["score"]).sum()
        rows.append({"model": m, "avg_cost": cm, "avg_score": sm})
    return pd.DataFrame(rows).sort_values("avg_score", ascending=False).reset_index(drop=True)


def single_best_per_benchmark(df, weights):
    """Pick per-dataset model maximizing weighted score, then aggregate."""
    rows = []
    chosen = {}
    for d, sub in df.groupby("dataset"):
        cand = []
        prompts_d = sub["prompt_id"].unique()
        for m, sub_m in sub.groupby("model"):
            if set(sub_m["prompt_id"]) != set(prompts_d):
                continue
            wd = sub_m["prompt_id"].map(weights)
            cand.append({
                "model": m,
                "score": float((wd * sub_m["score"]).sum()),
                "cost": float((wd * sub_m["cost"]).sum()),
            })
        c = pd.DataFrame(cand).sort_values(["score", "cost"], ascending=[False, True])
        best_m = c.iloc[0]["model"]
        chosen[d] = best_m
        for _, r in sub[sub["model"] == best_m].iterrows():
            rows.append({
                "prompt_id": r["prompt_id"],
                "model": best_m,
                "dataset": d,
                "weight": weights[r["prompt_id"]],
                "score": float(r["score"]),
                "cost": float(r["cost"]),
            })
    a = pd.DataFrame(rows)
    return {
        "policy": "Single Best per Benchmark",
        "selected_per_dataset": chosen,
        "avg_cost": float((a["weight"] * a["cost"]).sum()),
        "avg_score": float((a["weight"] * a["score"]).sum()),
        "assignments": a,
    }


# ----------------------------------------------------------------------
# Main BC-2SP
# ----------------------------------------------------------------------
def solve_bc2sp(
    df: pd.DataFrame,
    weights: dict,
    *,
    B: float | None = None,                 # cost budget; None = no budget
    K: int | None = None,                   # pool-size cap
    S_max: float | None = None,             # storage budget (GB)
    fairness_tau: float | None = None,      # per-prompt min quality with slack
    lambda_slack: float = 1.0,              # slack penalty in objective
    must_include: Iterable[str] = (),
    must_exclude: Iterable[str] = (),
    randomized: bool = False,               # x in [0,1] (LP relax)
    quiet: bool = True,
):
    """
    max  sum_p w_p sum_m Q_pm x_pm  -  lambda * sum_p w_p s_p
    s.t. sum_m x_pm = 1                                        (assignment)
         x_pm <= y_m                                            (linking)
         sum_m y_m <= K                                         (pool size)
         sum_m S_m y_m <= S_max                                 (storage)
         sum_{p,m} w_p C_pm x_pm <= B                          (budget)
         sum_m Q_pm x_pm + s_p >= tau,  s_p >= 0               (fairness)
         y_m = 1 for m in must_include; y_m = 0 for m in must_exclude
    """
    prompts, models, datasets, Q, C, dom = make_dicts(df)
    storage = {m: estimate_storage_gb(m) for m in models}

    mdl = pyo.ConcreteModel()
    mdl.P = pyo.Set(initialize=prompts)
    mdl.M = pyo.Set(initialize=models)

    mdl.w = pyo.Param(mdl.P, initialize=weights, within=pyo.NonNegativeReals)
    # Note: we index Q,C over P×M assuming full coverage (we restrict df beforehand).
    mdl.Q = pyo.Param(mdl.P, mdl.M, initialize={(p, m): Q[(p, m)] for p in prompts for m in models})
    mdl.C = pyo.Param(mdl.P, mdl.M, initialize={(p, m): C[(p, m)] for p in prompts for m in models})
    mdl.S = pyo.Param(mdl.M, initialize=storage)

    if randomized:
        mdl.x = pyo.Var(mdl.P, mdl.M, within=pyo.NonNegativeReals, bounds=(0, 1))
    else:
        mdl.x = pyo.Var(mdl.P, mdl.M, within=pyo.Binary)
    mdl.y = pyo.Var(mdl.M, within=pyo.Binary)

    if fairness_tau is not None:
        mdl.s = pyo.Var(mdl.P, within=pyo.NonNegativeReals)

    mdl.assign = pyo.Constraint(mdl.P, rule=lambda m, p: sum(m.x[p, k] for k in m.M) == 1)
    mdl.link = pyo.Constraint(mdl.P, mdl.M, rule=lambda m, p, k: m.x[p, k] <= m.y[k])

    if K is not None:
        mdl.pool_cap = pyo.Constraint(rule=lambda m: sum(m.y[k] for k in m.M) <= K)
    if S_max is not None:
        mdl.storage_cap = pyo.Constraint(rule=lambda m: sum(m.S[k] * m.y[k] for k in m.M) <= S_max)
    if B is not None:
        mdl.budget = pyo.Constraint(
            rule=lambda m: sum(m.w[p] * m.C[p, k] * m.x[p, k] for p in m.P for k in m.M) <= B
        )
    if fairness_tau is not None:
        mdl.fair = pyo.Constraint(
            mdl.P,
            rule=lambda m, p: sum(m.Q[p, k] * m.x[p, k] for k in m.M) + m.s[p] >= fairness_tau,
        )
    for inc in must_include:
        if inc in models:
            mdl.y[inc].fix(1)
    for exc in must_exclude:
        if exc in models:
            mdl.y[exc].fix(0)

    score_expr = sum(mdl.w[p] * mdl.Q[p, k] * mdl.x[p, k] for p in mdl.P for k in mdl.M)
    cost_expr = sum(mdl.w[p] * mdl.C[p, k] * mdl.x[p, k] for p in mdl.P for k in mdl.M)
    eps_cost = 1e-3  # tie-breaker: at score-saturation, prefer lower cost
    if fairness_tau is not None:
        slack_expr = sum(mdl.w[p] * mdl.s[p] for p in mdl.P)
        mdl.obj = pyo.Objective(
            expr=score_expr - lambda_slack * slack_expr - eps_cost * cost_expr,
            sense=pyo.maximize,
        )
    else:
        mdl.obj = pyo.Objective(expr=score_expr - eps_cost * cost_expr, sense=pyo.maximize)

    solver = get_solver()
    res = solver.solve(mdl, tee=False)

    selected_pool = [k for k in models if pyo.value(mdl.y[k]) > 0.5]
    rows = []
    for p in prompts:
        for k in models:
            xv = pyo.value(mdl.x[p, k])
            if xv > 1e-6:
                rows.append({
                    "prompt_id": p, "model": k, "dataset": dom[p],
                    "weight": weights[p], "x": xv, "score": Q[(p, k)], "cost": C[(p, k)],
                })
    a = pd.DataFrame(rows)
    avg_cost = float((a["weight"] * a["x"] * a["cost"]).sum())
    avg_score = float((a["weight"] * a["x"] * a["score"]).sum())
    return {
        "policy": ("Randomized BC-2SP" if randomized else "BC-2SP")
                  + ("" if fairness_tau is None else " + fairness")
                  + ("" if S_max is None else " + storage")
                  + ("" if K is None else f" (K={K})")
                  + ("" if B is None else f" (B={B:.4g})"),
        "B": B, "K": K, "S_max": S_max, "tau": fairness_tau, "lambda": lambda_slack,
        "selected_pool": selected_pool,
        "pool_size": len(selected_pool),
        "avg_cost": avg_cost,
        "avg_score": avg_score,
        "assignments": a,
        "obj_value": pyo.value(mdl.obj),
        "solver_status": str(res.solver.status),
        "term": str(res.solver.termination_condition),
    }


# ----------------------------------------------------------------------
# QC-2SP : min cost s.t. quality target
# ----------------------------------------------------------------------
def solve_qc2sp(
    df, weights, *,
    Q_target: float,
    K: int | None = None, S_max: float | None = None,
    must_include=(), must_exclude=(),
):
    prompts, models, datasets, Q, C, dom = make_dicts(df)
    storage = {m: estimate_storage_gb(m) for m in models}
    mdl = pyo.ConcreteModel()
    mdl.P = pyo.Set(initialize=prompts)
    mdl.M = pyo.Set(initialize=models)
    mdl.w = pyo.Param(mdl.P, initialize=weights)
    mdl.Q = pyo.Param(mdl.P, mdl.M, initialize={(p, m): Q[(p, m)] for p in prompts for m in models})
    mdl.C = pyo.Param(mdl.P, mdl.M, initialize={(p, m): C[(p, m)] for p in prompts for m in models})
    mdl.S = pyo.Param(mdl.M, initialize=storage)
    mdl.x = pyo.Var(mdl.P, mdl.M, within=pyo.Binary)
    mdl.y = pyo.Var(mdl.M, within=pyo.Binary)
    mdl.assign = pyo.Constraint(mdl.P, rule=lambda m, p: sum(m.x[p, k] for k in m.M) == 1)
    mdl.link = pyo.Constraint(mdl.P, mdl.M, rule=lambda m, p, k: m.x[p, k] <= m.y[k])
    mdl.qual = pyo.Constraint(
        rule=lambda m: sum(m.w[p] * m.Q[p, k] * m.x[p, k] for p in m.P for k in m.M) >= Q_target
    )
    if K is not None:
        mdl.pool_cap = pyo.Constraint(rule=lambda m: sum(m.y[k] for k in m.M) <= K)
    if S_max is not None:
        mdl.stor = pyo.Constraint(rule=lambda m: sum(m.S[k] * m.y[k] for k in m.M) <= S_max)
    for inc in must_include:
        if inc in models:
            mdl.y[inc].fix(1)
    for exc in must_exclude:
        if exc in models:
            mdl.y[exc].fix(0)
    mdl.obj = pyo.Objective(
        expr=sum(mdl.w[p] * mdl.C[p, k] * mdl.x[p, k] for p in mdl.P for k in mdl.M),
        sense=pyo.minimize,
    )
    res = get_solver().solve(mdl, tee=False)
    if str(res.solver.termination_condition) != "optimal":
        return {"policy": "QC-2SP", "Q_target": Q_target, "infeasible": True,
                "term": str(res.solver.termination_condition)}
    pool = [k for k in models if pyo.value(mdl.y[k]) > 0.5]
    rows = []
    for p in prompts:
        for k in models:
            if pyo.value(mdl.x[p, k]) > 0.5:
                rows.append({"prompt_id": p, "model": k, "dataset": dom[p],
                             "weight": weights[p], "score": Q[(p, k)], "cost": C[(p, k)]})
    a = pd.DataFrame(rows)
    return {
        "policy": "QC-2SP",
        "Q_target": Q_target, "K": K, "S_max": S_max,
        "selected_pool": pool, "pool_size": len(pool),
        "avg_cost": float((a["weight"] * a["cost"]).sum()),
        "avg_score": float((a["weight"] * a["score"]).sum()),
        "assignments": a,
    }


# ----------------------------------------------------------------------
# DRO BC-2SP : box ambiguity over benchmark proportions
# ----------------------------------------------------------------------
def solve_dro_bc2sp(
    df, prompts_uniform_weights, *,
    B: float, K: int | None = None, S_max: float | None = None,
    delta: float = 0.3,
):
    """
    Distributionally robust BC-2SP with a relative-box ambiguity set on the
    benchmark mixture w_d := Pr[xi in P_d] around w_d^0 = |P_d| / |P|:

        U(delta) = { w in simplex : |w_d - w_d^0| <= delta * w_d^0 ∀ d }.

    Within each domain prompts are uniformly weighted. Let
        f_d(x) := (1/|P_d|) sum_{p in P_d, m} Q_pm x_pm
    be the conditional score on domain d. The DRO problem is

        max_{y, x} min_{w in U(delta)} sum_d w_d f_d(x).

    Reparametrising w_d = w_d^0 (1 + e_d) with sum_d w_d^0 e_d = 0 and
    |e_d| <= delta, the inner LP has dual minimum
        min_w sum_d w_d f_d(x)
            = sum_d w_d^0 f_d(x)  -  delta * min_lambda sum_d w_d^0 |f_d - lambda|.

    Linearising the absolute value gives the equivalent MILP:

        max_{y, x, lambda, z}  sum_d w_d^0 f_d(x)  -  delta * sum_d w_d^0 z_d
        s.t.  z_d >= f_d(x) - lambda,  z_d >= lambda - f_d(x), ∀ d
              (BC-2SP constraints) and budget under nominal w.

    This adds |D|+1 variables and 2|D| constraints to BC-2SP.
    """
    prompts, models, datasets, Q, C, dom = make_dicts(df)
    storage = {m: estimate_storage_gb(m) for m in models}
    P_d = {d: [p for p in prompts if dom[p] == d] for d in datasets}
    nominal_d = {d: len(P_d[d]) / len(prompts) for d in datasets}  # = 1/|D| under uniform empirical

    mdl = pyo.ConcreteModel()
    mdl.P = pyo.Set(initialize=prompts)
    mdl.M = pyo.Set(initialize=models)
    mdl.D = pyo.Set(initialize=datasets)
    mdl.Q = pyo.Param(mdl.P, mdl.M, initialize={(p, m): Q[(p, m)] for p in prompts for m in models})
    mdl.C = pyo.Param(mdl.P, mdl.M, initialize={(p, m): C[(p, m)] for p in prompts for m in models})
    mdl.S = pyo.Param(mdl.M, initialize=storage)
    mdl.w_nominal = pyo.Param(mdl.P, initialize=prompts_uniform_weights)

    mdl.x = pyo.Var(mdl.P, mdl.M, within=pyo.Binary)
    mdl.y = pyo.Var(mdl.M, within=pyo.Binary)
    mdl.lam = pyo.Var(within=pyo.Reals)
    mdl.z = pyo.Var(mdl.D, within=pyo.NonNegativeReals)

    mdl.assign = pyo.Constraint(mdl.P, rule=lambda m, p: sum(m.x[p, k] for k in m.M) == 1)
    mdl.link = pyo.Constraint(mdl.P, mdl.M, rule=lambda m, p, k: m.x[p, k] <= m.y[k])
    if K is not None:
        mdl.pool_cap = pyo.Constraint(rule=lambda m: sum(m.y[k] for k in m.M) <= K)
    if S_max is not None:
        mdl.stor = pyo.Constraint(rule=lambda m: sum(m.S[k] * m.y[k] for k in m.M) <= S_max)
    mdl.budget = pyo.Constraint(
        rule=lambda m: sum(m.w_nominal[p] * m.C[p, k] * m.x[p, k] for p in m.P for k in m.M) <= B
    )

    # Conditional scores f_d(x), as Pyomo expressions (constructed lazily below).
    def f_d_expr(m, d):
        return (1.0 / len(P_d[d])) * sum(m.Q[p, k] * m.x[p, k] for p in P_d[d] for k in m.M)

    mdl.z_pos = pyo.Constraint(mdl.D, rule=lambda m, d: m.z[d] >= f_d_expr(m, d) - m.lam)
    mdl.z_neg = pyo.Constraint(mdl.D, rule=lambda m, d: m.z[d] >= m.lam - f_d_expr(m, d))

    nominal_score = sum(nominal_d[d] * f_d_expr(mdl, d) for d in datasets)
    mad_penalty = sum(nominal_d[d] * mdl.z[d] for d in datasets)
    mdl.obj = pyo.Objective(expr=nominal_score - delta * mad_penalty, sense=pyo.maximize)

    res = get_solver().solve(mdl, tee=False)
    pool = [k for k in models if pyo.value(mdl.y[k]) > 0.5]

    rows = []
    for p in prompts:
        for k in models:
            if pyo.value(mdl.x[p, k]) > 0.5:
                rows.append({"prompt_id": p, "model": k, "dataset": dom[p],
                             "weight": prompts_uniform_weights[p],
                             "score": Q[(p, k)], "cost": C[(p, k)]})
    a = pd.DataFrame(rows)
    worst_case = float(pyo.value(mdl.obj))
    return {
        "policy": "DRO BC-2SP",
        "B": B, "K": K, "S_max": S_max, "delta": delta,
        "selected_pool": pool, "pool_size": len(pool),
        "avg_cost_nominal": float((a["weight"] * a["cost"]).sum()),
        "avg_score_nominal": float((a["weight"] * a["score"]).sum()),
        "worst_case_score": worst_case,
        "lambda_star": float(pyo.value(mdl.lam)),
        "assignments": a,
    }


# ----------------------------------------------------------------------
# Experiment runners
# ----------------------------------------------------------------------
def sweep_budget_frontier(df, weights, B_grid, K=8):
    out = []
    for B in B_grid:
        r = solve_bc2sp(df, weights, B=B, K=K)
        out.append({"B": B, "avg_cost": r["avg_cost"], "avg_score": r["avg_score"],
                    "pool_size": r["pool_size"], "pool": "|".join(r["selected_pool"])})
    return pd.DataFrame(out)


def sweep_pool_size(df, weights, K_grid, B):
    out = []
    for K in K_grid:
        r = solve_bc2sp(df, weights, B=B, K=K)
        out.append({"K": K, "avg_cost": r["avg_cost"], "avg_score": r["avg_score"],
                    "pool_size": r["pool_size"], "pool": "|".join(r["selected_pool"])})
    return pd.DataFrame(out)


def sweep_dro_delta(df, weights, delta_grid, B, K):
    out = []
    for d in delta_grid:
        r = solve_dro_bc2sp(df, weights, B=B, K=K, delta=d)
        out.append({"delta": d, "avg_score_nominal": r["avg_score_nominal"],
                    "avg_cost_nominal": r["avg_cost_nominal"],
                    "worst_case_score": r["worst_case_score"],
                    "pool_size": r["pool_size"], "pool": "|".join(r["selected_pool"])})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------
# Per-benchmark breakdown
# ----------------------------------------------------------------------
def per_benchmark_breakdown(assignments_df, weights):
    out = []
    for d, sub in assignments_df.groupby("dataset"):
        n = len(sub)
        out.append({
            "dataset": d,
            "avg_cost": float((sub["weight"] * sub["cost"]).sum() / sub["weight"].sum()),
            "avg_score": float((sub["weight"] * sub["score"]).sum() / sub["weight"].sum()),
            "n_prompts": n,
        })
    return pd.DataFrame(out).sort_values("dataset").reset_index(drop=True)


# ----------------------------------------------------------------------
# Plotting helpers
# ----------------------------------------------------------------------
def plot_frontier(df_bc, df_oracle_pt, df_sb, df_sbb, df_dro, out_path):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    df_bc_sorted = df_bc.sort_values("avg_cost")
    ax.plot(df_bc_sorted["avg_cost"], df_bc_sorted["avg_score"],
            "o-", color="#2c7bb6", lw=2.0, markersize=7, label="BC-2SP (ours)")
    if df_dro is not None and len(df_dro):
        ax.plot(df_dro["avg_cost"], df_dro["avg_score"],
                "D-", color="#d7191c", lw=1.8, markersize=6, label="DRO BC-2SP")
    ax.scatter([df_oracle_pt["avg_cost"]], [df_oracle_pt["avg_score"]],
               marker="*", s=240, color="#fdae61", edgecolor="black",
               zorder=5, label="Oracle (per-prompt best)")
    ax.scatter(df_sb["avg_cost"], df_sb["avg_score"],
               marker="s", s=24, color="grey", alpha=0.5, label="Single Best (any single model)")
    ax.scatter([df_sbb["avg_cost"]], [df_sbb["avg_score"]],
               marker="^", s=120, color="#abd9e9", edgecolor="black",
               zorder=5, label="Single Best per Benchmark")
    ax.set_xscale("symlog", linthresh=1e-4)
    ax.set_xlabel("Weighted average cost per prompt (USD)")
    ax.set_ylabel("Weighted average score")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Cost-Quality Frontier on routerbench (240 prompts × 32 models)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_pool_size_curve(df_K, out_path):
    fig, ax1 = plt.subplots(figsize=(7.0, 4.5))
    ax1.plot(df_K["K"], df_K["avg_score"], "o-", color="#2c7bb6", lw=2, label="avg score")
    ax1.set_xlabel("Pool-size cap K")
    ax1.set_ylabel("Weighted avg score", color="#2c7bb6")
    ax1.tick_params(axis="y", labelcolor="#2c7bb6")
    ax1.grid(True, ls="--", alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(df_K["K"], df_K["avg_cost"], "s--", color="#d7191c", lw=1.5, label="avg cost")
    ax2.set_ylabel("Weighted avg cost (USD)", color="#d7191c")
    ax2.tick_params(axis="y", labelcolor="#d7191c")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_pool_composition_heatmap(df_K_with_pool, all_models, out_path):
    K_values = df_K_with_pool["K"].tolist()
    sel_lists = [set(p.split("|")) for p in df_K_with_pool["pool"].tolist()]
    # Order models by their "first appearance" in K-sweep, then by frequency
    appearance = {}
    for K_idx, sel in enumerate(sel_lists):
        for m in sel:
            if m not in appearance:
                appearance[m] = K_idx
    ordered_models = [m for m, _ in sorted(appearance.items(), key=lambda kv: kv[1])]
    ordered_models += [m for m in all_models if m not in ordered_models]
    # restrict to models that appear at least once
    ordered_models = [m for m in ordered_models if m in appearance]
    M = np.zeros((len(ordered_models), len(K_values)))
    for j, sel in enumerate(sel_lists):
        for i, m in enumerate(ordered_models):
            M[i, j] = 1.0 if m in sel else 0.0
    fig, ax = plt.subplots(figsize=(0.55 * len(K_values) + 2, 0.32 * len(ordered_models) + 1))
    ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=1.5)
    ax.set_yticks(range(len(ordered_models)))
    ax.set_yticklabels(ordered_models, fontsize=8)
    ax.set_xticks(range(len(K_values)))
    ax.set_xticklabels(K_values)
    ax.set_xlabel("Pool size cap K")
    ax.set_title("Pool composition vs K (B fixed)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_dro_robustness(df_dro, df_nominal_at_K, out_path):
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(df_dro["delta"], df_dro["avg_score_nominal"], "o-", lw=2, color="#2c7bb6",
            label="DRO solution: nominal score")
    ax.plot(df_dro["delta"], df_dro["worst_case_score"], "D--", lw=1.8, color="#d7191c",
            label="DRO solution: worst-case score")
    ax.axhline(df_nominal_at_K, color="grey", ls=":", label="Nominal BC-2SP (no DRO)")
    ax.set_xlabel("Box ambiguity radius δ")
    ax.set_ylabel("Score")
    ax.set_title("Price of distributional robustness")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def main():
    print("=" * 78)
    print("INDENG 164 — LLM Pool Curation via Two-Stage Stochastic Optimization")
    print("=" * 78)

    df_raw = load_data()
    print(f"Raw rows: {len(df_raw):,}; prompts: {df_raw['prompt_id'].nunique()}; models: {df_raw['model'].nunique()}")
    df = restrict_to_full_coverage(df_raw)
    n_p = df["prompt_id"].nunique()
    n_m = df["model"].nunique()
    print(f"After coverage restriction: prompts={n_p}, models={n_m}, rows={len(df):,}")

    weights = make_uniform_weights(sorted(df["prompt_id"].unique()))

    # ---- baselines ----
    print("\n[1] Baselines")
    sb_all = single_best(df, weights)
    print(f"  Best single model: {sb_all.iloc[0]['model']} -> score={sb_all.iloc[0]['avg_score']:.4f}, cost={sb_all.iloc[0]['avg_cost']:.6f}")
    sbb = single_best_per_benchmark(df, weights)
    print(f"  Single Best per Benchmark -> score={sbb['avg_score']:.4f}, cost={sbb['avg_cost']:.6f}, sel={sbb['selected_per_dataset']}")
    oracle = oracle_routing(df, weights)
    print(f"  Oracle (per-prompt best)  -> score={oracle['avg_score']:.4f}, cost={oracle['avg_cost']:.6f}")

    sb_all.to_csv(TABLE_DIR / "single_best_table.csv", index=False)
    sbb["assignments"].to_csv(TABLE_DIR / "sbb_assignments.csv", index=False)
    oracle["assignments"].to_csv(TABLE_DIR / "oracle_assignments.csv", index=False)

    # ---- BC-2SP cost-quality frontier (sweep B) ----
    print("\n[2] BC-2SP Pareto frontier (sweep B)")
    B_grid = [1e-5, 1e-4, 3e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1]
    t0 = time.time()
    df_B = sweep_budget_frontier(df, weights, B_grid, K=8)
    print(df_B.to_string())
    df_B.to_csv(TABLE_DIR / "frontier_B.csv", index=False)
    print(f"  ({time.time()-t0:.1f}s)")

    # ---- Diminishing returns of K ----
    print("\n[3] Diminishing returns of K (B = 5e-3)")
    K_grid = list(range(1, 11))
    df_K = sweep_pool_size(df, weights, K_grid, B=5e-3)
    print(df_K.to_string())
    df_K.to_csv(TABLE_DIR / "K_curve.csv", index=False)

    # ---- Storage budget ablation ----
    print("\n[4] Storage budget ablation (B = 5e-3, K = 8)")
    rows = []
    for S_max in [None, 200, 100, 50, 20, 0.0]:
        r = solve_bc2sp(df, weights, B=5e-3, K=8, S_max=S_max)
        rows.append({
            "S_max_GB": S_max if S_max is not None else "inf",
            "avg_score": r["avg_score"], "avg_cost": r["avg_cost"],
            "pool_size": r["pool_size"], "pool": "|".join(r["selected_pool"]),
        })
    df_S = pd.DataFrame(rows)
    print(df_S.to_string())
    df_S.to_csv(TABLE_DIR / "storage_ablation.csv", index=False)

    # ---- Fairness slack ablation ----
    print("\n[5] Fairness slack ablation (B = 5e-3, K = 8, lambda = 1.0)")
    rows = []
    for tau in [None, 0.0, 0.5, 0.8, 1.0]:
        r = solve_bc2sp(df, weights, B=5e-3, K=8, fairness_tau=tau, lambda_slack=1.0)
        rows.append({
            "tau": tau if tau is not None else "off",
            "avg_score": r["avg_score"], "avg_cost": r["avg_cost"],
            "pool_size": r["pool_size"], "pool": "|".join(r["selected_pool"]),
        })
    df_F = pd.DataFrame(rows)
    print(df_F.to_string())
    df_F.to_csv(TABLE_DIR / "fairness_ablation.csv", index=False)

    # ---- Provider sensitivity ----
    print("\n[6] Provider sensitivity (B = 5e-3, K = 8)")
    flagship_openai = ["gpt-5", "gpt-5-chat"]
    rows = []
    r = solve_bc2sp(df, weights, B=5e-3, K=8); rows.append({"setting": "free", "avg_score": r["avg_score"], "avg_cost": r["avg_cost"], "pool": "|".join(r["selected_pool"])})
    r = solve_bc2sp(df, weights, B=5e-3, K=8, must_exclude=flagship_openai); rows.append({"setting": "no OpenAI", "avg_score": r["avg_score"], "avg_cost": r["avg_cost"], "pool": "|".join(r["selected_pool"])})
    r = solve_bc2sp(df, weights, B=5e-3, K=8, must_include=["claude-sonnet-4"]); rows.append({"setting": "must Claude", "avg_score": r["avg_score"], "avg_cost": r["avg_cost"], "pool": "|".join(r["selected_pool"])})
    df_prov = pd.DataFrame(rows)
    print(df_prov.to_string())
    df_prov.to_csv(TABLE_DIR / "provider_ablation.csv", index=False)

    # ---- DRO ----
    print("\n[7] DRO BC-2SP (B = 5e-3, K = 8) — sweep delta")
    delta_grid = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
    df_dro_K8 = sweep_dro_delta(df, weights, delta_grid, B=5e-3, K=8)
    print(df_dro_K8.to_string())
    df_dro_K8.to_csv(TABLE_DIR / "dro_delta_curve.csv", index=False)

    # ---- DRO frontier (worst-case scores along B sweep) ----
    print("\n[8] DRO Pareto frontier (delta=0.3) along B sweep")
    rows = []
    for B in B_grid:
        r = solve_dro_bc2sp(df, weights, B=B, K=8, delta=0.3)
        rows.append({
            "B": B, "avg_cost": r["avg_cost_nominal"],
            "avg_score": r["avg_score_nominal"],
            "worst_case_score": r["worst_case_score"],
            "pool_size": r["pool_size"], "pool": "|".join(r["selected_pool"]),
        })
    df_dro_B = pd.DataFrame(rows)
    print(df_dro_B.to_string())
    df_dro_B.to_csv(TABLE_DIR / "dro_frontier_delta03.csv", index=False)

    # ---- QC-2SP (alternative formulation) ----
    print("\n[9] QC-2SP (sweep Q-target)")
    Q_grid = [0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98]
    rows = []
    for Q in Q_grid:
        r = solve_qc2sp(df, weights, Q_target=Q, K=8)
        if r.get("infeasible"):
            rows.append({"Q_target": Q, "infeasible": True}); continue
        rows.append({"Q_target": Q, "avg_score": r["avg_score"], "avg_cost": r["avg_cost"],
                     "pool_size": r["pool_size"], "pool": "|".join(r["selected_pool"])})
    df_QC = pd.DataFrame(rows)
    print(df_QC.to_string())
    df_QC.to_csv(TABLE_DIR / "qc2sp.csv", index=False)

    # ---- Randomized routing comparison ----
    print("\n[10] Randomized routing (LP relax) vs deterministic")
    rows = []
    for B in [1e-3, 5e-3, 2e-2]:
        r_det = solve_bc2sp(df, weights, B=B, K=8)
        r_rand = solve_bc2sp(df, weights, B=B, K=8, randomized=True)
        rows.append({
            "B": B,
            "det_score": r_det["avg_score"], "rand_score": r_rand["avg_score"],
            "rand_minus_det": r_rand["avg_score"] - r_det["avg_score"],
            "det_pool": "|".join(r_det["selected_pool"]),
        })
    df_R = pd.DataFrame(rows)
    print(df_R.to_string())
    df_R.to_csv(TABLE_DIR / "randomized_vs_det.csv", index=False)

    # ---- Per-benchmark breakdown for representative B ----
    print("\n[11] Per-benchmark breakdown (B = 5e-3, K = 8)")
    r = solve_bc2sp(df, weights, B=5e-3, K=8)
    bb = per_benchmark_breakdown(r["assignments"].assign(weight=r["assignments"]["weight"] * r["assignments"]["x"]),
                                 weights)
    print(bb.to_string())
    bb.to_csv(TABLE_DIR / "per_benchmark_breakdown.csv", index=False)

    # ---- Usage shares ----
    print("\n[12] Usage shares of each selected model (B = 5e-3, K = 8)")
    a = r["assignments"]
    a["weighted_x"] = a["weight"] * a["x"]
    usage = a.groupby("model")["weighted_x"].sum().sort_values(ascending=False)
    print(usage.to_string())
    usage.to_csv(TABLE_DIR / "usage_shares.csv")

    # ---- Plots ----
    print("\n[13] Plots")
    plot_frontier(df_B, oracle, sb_all, sbb, df_dro_B, FIG_DIR / "fig1_frontier.png")
    plot_pool_size_curve(df_K, FIG_DIR / "fig2_K_curve.png")
    all_models = sorted(df["model"].unique())
    plot_pool_composition_heatmap(df_K, all_models, FIG_DIR / "fig3_pool_composition.png")
    plot_dro_robustness(df_dro_K8, df_K[df_K["K"] == 8]["avg_score"].iloc[0], FIG_DIR / "fig5_dro_robustness.png")
    print("  saved figs to", FIG_DIR)

    # ---- Summary ----
    summary = {
        "n_prompts": int(n_p),
        "n_models": int(n_m),
        "B_grid": B_grid,
        "K_grid": K_grid,
        "delta_grid": delta_grid,
        "Q_grid": Q_grid,
        "best_single_model": str(sb_all.iloc[0]["model"]),
        "best_single_score": float(sb_all.iloc[0]["avg_score"]),
        "best_single_cost": float(sb_all.iloc[0]["avg_cost"]),
        "oracle_score": oracle["avg_score"],
        "oracle_cost": oracle["avg_cost"],
        "sbb_score": sbb["avg_score"],
        "sbb_cost": sbb["avg_cost"],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nDONE — outputs at", OUT_DIR)


if __name__ == "__main__":
    main()
