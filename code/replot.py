"""
Regenerate report figures in a friend-style (linear x-axis, simple matplotlib lines)
and write them to /workspace/optimization/report/figure/ for Overleaf.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import solution as sol

REPORT_FIG_DIR = Path("/workspace/optimization/report/figure")
REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)

ALPHA_GRID = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]


def alpha_sweep_oracle(df, weights):
    """Per-prompt best at each alpha, traces the unconstrained Pareto."""
    rows = []
    prompts, models, _, Q, C, dom = sol.make_dicts(df)
    for a in ALPHA_GRID:
        cost = 0.0
        score = 0.0
        for p in prompts:
            best_obj = None
            best = None
            for m in models:
                obj = C[(p, m)] - a * Q[(p, m)]
                if best_obj is None or obj < best_obj:
                    best_obj = obj
                    best = m
            cost += weights[p] * C[(p, best)]
            score += weights[p] * Q[(p, best)]
        rows.append({"alpha": a, "avg_cost": cost, "avg_score": score})
    return pd.DataFrame(rows)


def alpha_sweep_single_best(df, weights):
    rows = []
    for a in ALPHA_GRID:
        best_score = -np.inf
        best_cost = 0.0
        best_obj = None
        for m, sub in df.groupby("model"):
            wp = sub["prompt_id"].map(weights)
            sm = (wp * sub["score"]).sum()
            cm = (wp * sub["cost"]).sum()
            obj = cm - a * sm
            if best_obj is None or obj < best_obj:
                best_obj = obj
                best_cost = cm
                best_score = sm
        rows.append({"alpha": a, "avg_cost": best_cost, "avg_score": best_score})
    return pd.DataFrame(rows)


def alpha_sweep_single_best_per_benchmark(df, weights):
    rows = []
    for a in ALPHA_GRID:
        total_c, total_s = 0.0, 0.0
        for d, sub in df.groupby("dataset"):
            best_obj = None
            best_c = 0.0
            best_s = 0.0
            for m, sub_m in sub.groupby("model"):
                wp = sub_m["prompt_id"].map(weights)
                sm = (wp * sub_m["score"]).sum()
                cm = (wp * sub_m["cost"]).sum()
                obj = cm - a * sm
                if best_obj is None or obj < best_obj:
                    best_obj = obj
                    best_c = cm
                    best_s = sm
            total_c += best_c
            total_s += best_s
        rows.append({"alpha": a, "avg_cost": total_c, "avg_score": total_s})
    return pd.DataFrame(rows)


def bc2sp_frontier_for_plot(df, weights):
    B_grid = [1e-5, 1e-4, 3e-4, 5e-4, 1e-3, 2e-3, 5e-3]
    rows = []
    for B in B_grid:
        r = sol.solve_bc2sp(df, weights, B=B, K=8)
        rows.append({"B": B, "avg_cost": r["avg_cost"], "avg_score": r["avg_score"],
                     "pool_size": r["pool_size"]})
    return pd.DataFrame(rows)


def main():
    df = sol.restrict_to_full_coverage(sol.load_data())
    weights = sol.make_uniform_weights(sorted(df["prompt_id"].unique()))

    print("Computing alpha-swept curves for plot...")
    df_oracle = alpha_sweep_oracle(df, weights)
    df_sb = alpha_sweep_single_best(df, weights)
    df_sbb = alpha_sweep_single_best_per_benchmark(df, weights)
    print("Computing BC-2SP frontier...")
    df_bc = bc2sp_frontier_for_plot(df, weights)
    print(df_bc.to_string())

    # Save the curves to CSV alongside report
    df_oracle.to_csv(REPORT_FIG_DIR / "_curve_oracle.csv", index=False)
    df_sb.to_csv(REPORT_FIG_DIR / "_curve_sb.csv", index=False)
    df_sbb.to_csv(REPORT_FIG_DIR / "_curve_sbb.csv", index=False)
    df_bc.to_csv(REPORT_FIG_DIR / "_curve_bc2sp.csv", index=False)

    # Figure 1: cost-quality plane (linear x), friend-style
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.plot(df_oracle["avg_cost"], df_oracle["avg_score"],
            "o-", color="C0", lw=2.0, label="Oracle (per-prompt best)")
    ax.plot(df_sb["avg_cost"], df_sb["avg_score"],
            "s--", color="C1", lw=1.6, label="Single Best")
    ax.plot(df_sbb["avg_cost"], df_sbb["avg_score"],
            "^--", color="C2", lw=1.6, label="Single Best per Benchmark")
    ax.plot(df_bc["avg_cost"], df_bc["avg_score"],
            "D-.", color="C3", lw=2.0, label="BC-2SP (ours)")
    ax.set_xlabel("Weighted Average Cost (USD per prompt)")
    ax.set_ylabel("Weighted Average Score")
    ax.set_title("Routing Policies on the Cost-Performance Plane")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = REPORT_FIG_DIR / "fig1_frontier.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print("Wrote", out)

    # Figure 2: K-curve (one panel)
    K_grid = list(range(1, 11))
    rows = []
    for K in K_grid:
        r = sol.solve_bc2sp(df, weights, B=5e-3, K=K)
        rows.append({"K": K, "avg_score": r["avg_score"], "avg_cost": r["avg_cost"],
                     "pool_size": r["pool_size"], "pool": r["selected_pool"]})
    df_K = pd.DataFrame(rows)
    df_K.drop(columns=["pool"]).to_csv(REPORT_FIG_DIR / "_curve_K.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(df_K["K"], df_K["avg_score"], "o-", color="C0", lw=2.0)
    ax.set_xlabel("Pool-size cap $K$")
    ax.set_ylabel("Weighted Average Score")
    ax.set_title("Diminishing Returns of Pool Size at $B=\\$0.005$")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(K_grid)
    fig.tight_layout()
    out = REPORT_FIG_DIR / "fig2_K_curve.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print("Wrote", out)

    # Figure 3: Pool composition heatmap (kept similar to before)
    import numpy as np
    pools = df_K["pool"].tolist()
    appearance = {}
    for j, sel in enumerate(pools):
        for m in sel:
            if m not in appearance:
                appearance[m] = j
    ordered = [m for m, _ in sorted(appearance.items(), key=lambda kv: kv[1])]
    M = np.zeros((len(ordered), len(K_grid)))
    for j, sel in enumerate(pools):
        for i, m in enumerate(ordered):
            if m in sel:
                M[i, j] = 1.0
    fig, ax = plt.subplots(figsize=(0.55*len(K_grid)+2, 0.32*len(ordered)+1))
    ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=1.5)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(ordered, fontsize=8)
    ax.set_xticks(range(len(K_grid)))
    ax.set_xticklabels(K_grid)
    ax.set_xlabel("Pool size cap $K$")
    ax.set_title("Pool composition vs $K$ (B fixed)")
    fig.tight_layout()
    out = REPORT_FIG_DIR / "fig3_pool_composition.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print("Wrote", out)

    # Figure 5: DRO robustness curve (one panel)
    delta_grid = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
    rows = []
    for d in delta_grid:
        r = sol.solve_dro_bc2sp(df, weights, B=5e-3, K=8, delta=d)
        rows.append({"delta": d, "avg_score_nominal": r["avg_score_nominal"],
                     "avg_cost_nominal": r["avg_cost_nominal"],
                     "worst_case_score": r["worst_case_score"]})
    df_dro = pd.DataFrame(rows)
    df_dro.to_csv(REPORT_FIG_DIR / "_curve_dro.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(df_dro["delta"], df_dro["avg_score_nominal"],
            "o-", color="C0", lw=2.0, label="DRO solution: nominal score")
    ax.plot(df_dro["delta"], df_dro["worst_case_score"],
            "D--", color="C3", lw=1.8, label="DRO solution: worst-case score")
    ax.axhline(0.9833333, color="grey", ls=":", label="Nominal BC-2SP")
    ax.set_xlabel("Box ambiguity radius $\\delta$")
    ax.set_ylabel("Weighted Average Score")
    ax.set_title("Price of Distributional Robustness ($B=\\$0.005$, $K=8$)")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = REPORT_FIG_DIR / "fig5_dro_robustness.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print("Wrote", out)

    print("DONE")


if __name__ == "__main__":
    main()
