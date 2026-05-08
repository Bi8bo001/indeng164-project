"""Generate pool composition fig3 from cached K_curve.csv (no MILP)."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REPORT_FIG_DIR = Path("/workspace/optimization/report/figure")
CACHE = Path("/workspace/optimization/code/solution_outputs/tables/K_curve.csv")

TIER_OF = {
    "gpt-5": "Flagship API",
    "gpt-5-chat": "Flagship API",
    "gemini-2.5-pro": "Flagship API",
    "claude-sonnet-4": "Flagship API",
    "gemini-2.5-flash": "Mid-tier API",
    "glm-4.6": "Mid-tier API",
    "kimi-k2-0905": "Mid-tier API",
    "deepseek-r1-0528": "Mid-tier API",
    "deepseek-v3-0324": "Mid-tier API",
    "qwen3-235b-a22b-2507": "Large reasoning",
    "qwen3-235b-a22b-thinking-2507": "Large reasoning",
    "intern-s1": "Large reasoning",
    "Intern-S1-mini": "Large reasoning",
}
TIER_COLOR = {
    "Open-source 8B/9B": "#2c7bb6",
    "Large reasoning":   "#fdae61",
    "Mid-tier API":      "#abd9e9",
    "Flagship API":      "#d7191c",
}


def tier_of(name):
    return TIER_OF.get(name, "Open-source 8B/9B")


def main():
    df = pd.read_csv(CACHE)
    K_grid = df["K"].tolist()
    pools = [set(p.split("|")) for p in df["pool"]]

    appearance = {}
    for j, sel in enumerate(pools):
        for m in sel:
            appearance.setdefault(m, j + 1)

    tier_order = ["Flagship API", "Mid-tier API", "Large reasoning", "Open-source 8B/9B"]
    ordered_models = []
    for tier in tier_order:
        members = [m for m in appearance if tier_of(m) == tier]
        members.sort(key=lambda m: appearance[m])
        ordered_models.extend(members)

    fig, ax = plt.subplots(figsize=(8.0, 0.32 * len(ordered_models) + 1.6))
    yticks = []
    for i, m in enumerate(ordered_models):
        y = len(ordered_models) - 1 - i
        yticks.append(y)
        included = [K for K, pool in zip(K_grid, pools) if m in pool]
        if not included:
            continue
        runs = []
        start = included[0]; prev = start
        for K in included[1:]:
            if K == prev + 1:
                prev = K
            else:
                runs.append((start, prev))
                start = K; prev = K
        runs.append((start, prev))
        color = TIER_COLOR[tier_of(m)]
        for s, e in runs:
            ax.barh(y, e - s + 1, left=s - 0.4, height=0.7,
                    color=color, edgecolor="white", linewidth=0.6)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ordered_models, fontsize=9)
    ax.invert_yaxis()
    ax.set_xticks(K_grid)
    ax.set_xticklabels(K_grid)
    ax.set_xlim(0.4, max(K_grid) + 0.6)
    ax.set_ylim(-0.6, len(ordered_models) - 0.4)
    ax.set_xlabel("Pool size cap $K$", fontsize=10)
    ax.set_title("Pool composition vs $K$  ($B = \\$0.005$)", fontsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [mpatches.Patch(color=TIER_COLOR[t], label=t) for t in tier_order]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              fontsize=9, ncol=4, frameon=False)

    fig.tight_layout()
    out = REPORT_FIG_DIR / "fig3_pool_composition.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print("Wrote", out)


if __name__ == "__main__":
    main()
