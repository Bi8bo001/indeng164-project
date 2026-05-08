"""Build a minimal solution.ipynb that mirrors starter.ipynb's structure.

Only short bold-header markdown cells. Code cells contain the meat.
"""

import json
import re
import uuid
from pathlib import Path

# Read solution.py and split into named code blocks.
SRC = Path("/workspace/optimization/code/solution.py").read_text()


def slice_block(name: str) -> str:
    """Extract a top-level def or block by anchor regex."""
    pat = re.compile(rf"({re.escape(name)}.*?)(?=\n(?:def |# ----|class |if __name__|$))",
                     re.DOTALL)
    m = pat.search(SRC)
    return m.group(1).rstrip() if m else ""


def md(source: str):
    return {"cell_type": "markdown", "metadata": {}, "id": uuid.uuid4().hex[:8],
            "source": source}


def code(source: str):
    return {"cell_type": "code", "metadata": {}, "id": uuid.uuid4().hex[:8],
            "execution_count": None, "outputs": [], "source": source}


cells = []

cells.append(md(
    "# **IEOR 164 Project: Two-Stage Stochastic LLM Routing**\n\n"
    "by Yu Hin Liang, Jingwen Yang"
))

cells.append(md("# **Environment Setup**"))

cells.append(code(
    "# For Google Colab: install pyomo + highspy. Locally use the kernel that already has them.\n"
    "import sys\n"
    "if 'google.colab' in sys.modules:\n"
    "    %pip install pyomo >/dev/null 2>/dev/null\n"
    "    %pip install highspy >/dev/null 2>/dev/null\n"
    "\n"
    "import pyomo.environ as pyo\n"
    "SOLVER_NAME = 'appsi_highs'\n"
    "SOLVER = pyo.SolverFactory(SOLVER_NAME)\n"
    "assert SOLVER.available(False), f'Solver {SOLVER_NAME} not available.'"
))

cells.append(code(
    "from pathlib import Path\n"
    "import math, json, time, itertools\n"
    "from collections import defaultdict\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import pyomo.environ as pyo"
))

cells.append(code(
    "# ============================================================\n"
    "# Configuration\n"
    "# ============================================================\n"
    "CSV_PATH = 'routerbench.csv'\n"
    "OUTPUT_DIR = Path('solution_outputs')\n"
    "OUTPUT_DIR.mkdir(exist_ok=True)\n"
    "\n"
    "# Cost-budget grid (USD per prompt) for the BC-2SP frontier sweep.\n"
    "B_GRID = [1e-5, 1e-4, 3e-4, 5e-4, 1e-3, 2e-3, 5e-3]\n"
    "K_DEFAULT = 8\n"
    "EPS_TIEBREAK = 1e-3      # cost tie-breaker so the cost row is monotone in B\n"
    "RANDOM_SEED = 42"
))

cells.append(code(
    "# ============================================================\n"
    "# Utility functions\n"
    "# ============================================================\n"
    "def get_solver():\n"
    "    s = pyo.SolverFactory(SOLVER_NAME)\n"
    "    if not s.available(False):\n"
    "        raise RuntimeError(f'Solver {SOLVER_NAME} not available.')\n"
    "    return s\n"
    "\n"
    "def safe_float(x, default=0.0):\n"
    "    try:\n"
    "        if pd.isna(x): return default\n"
    "        return float(x)\n"
    "    except Exception:\n"
    "        return default"
))

cells.append(md("# **Data loading and preprocessing**"))

cells.append(code(
    "def load_data(csv_path: str = CSV_PATH) -> pd.DataFrame:\n"
    "    df = pd.read_csv(csv_path)\n"
    "    for c in ['dataset','prompt_id','model']:\n"
    "        df[c] = df[c].astype(str)\n"
    "    df['score'] = pd.to_numeric(df['score'], errors='coerce').fillna(0.0)\n"
    "    df['cost']  = pd.to_numeric(df['cost'],  errors='coerce').fillna(0.0)\n"
    "    df = df.drop_duplicates(['prompt_id','model']).copy()\n"
    "    return df\n"
    "\n"
    "def restrict_to_full_coverage(df):\n"
    "    n_p = df['prompt_id'].nunique()\n"
    "    counts = df.groupby('model')['prompt_id'].nunique()\n"
    "    keep = counts[counts == n_p].index.tolist()\n"
    "    return df[df['model'].isin(keep)].copy()\n"
    "\n"
    "def make_dicts(df):\n"
    "    Q, C, dom = {}, {}, {}\n"
    "    for _, r in df.iterrows():\n"
    "        Q[(r['prompt_id'], r['model'])] = float(r['score'])\n"
    "        C[(r['prompt_id'], r['model'])] = float(r['cost'])\n"
    "        dom[r['prompt_id']] = r['dataset']\n"
    "    prompts = sorted(df['prompt_id'].unique())\n"
    "    models  = sorted(df['model'].unique())\n"
    "    datasets= sorted(df['dataset'].unique())\n"
    "    return prompts, models, datasets, Q, C, dom\n"
    "\n"
    "def make_uniform_weights(prompts):\n"
    "    return {p: 1.0/len(prompts) for p in prompts}\n"
    "\n"
    "def estimate_storage_gb(name):\n"
    "    n = name.lower()\n"
    "    flagship = ('gpt-5','gpt-5-chat','claude-sonnet-4','gemini-2.5-pro',\n"
    "                'gemini-2.5-flash','glm-4.6')\n"
    "    if any(k in n for k in flagship):                  return 0.0\n"
    "    if '235b' in n:                                    return 470.0\n"
    "    if 'deepseek-v3' in n or 'deepseek-r1' in n or 'kimi-k2' in n: return 0.0\n"
    "    if any(s in n for s in ('9b','9-b')):              return 18.0\n"
    "    if any(s in n for s in ('8b','8-b')):              return 16.0\n"
    "    if any(s in n for s in ('7b','7-b')):              return 14.0\n"
    "    if 'intern-s1' == n.split('-')[0]:                 return 0.0\n"
    "    return 12.0"
))

cells.append(md("# **Baseline policies**"))

cells.append(code(
    "def oracle_routing(df, weights):\n"
    "    rows = []\n"
    "    for p, sub in df.groupby('prompt_id'):\n"
    "        max_s = sub['score'].max()\n"
    "        cand = sub[sub['score'] == max_s].sort_values('cost').iloc[0]\n"
    "        rows.append({'prompt_id': p, 'model': cand['model'], 'dataset': cand['dataset'],\n"
    "                     'weight': weights[p], 'score': float(cand['score']),\n"
    "                     'cost': float(cand['cost'])})\n"
    "    a = pd.DataFrame(rows)\n"
    "    return {'policy': 'Oracle (per-prompt best)',\n"
    "            'avg_cost':  float((a['weight']*a['cost']).sum()),\n"
    "            'avg_score': float((a['weight']*a['score']).sum()),\n"
    "            'assignments': a}\n"
    "\n"
    "def single_best(df, weights):\n"
    "    rows = []\n"
    "    for m, sub in df.groupby('model'):\n"
    "        wp = sub['prompt_id'].map(weights)\n"
    "        rows.append({'model': m,\n"
    "                     'avg_cost': float((wp*sub['cost']).sum()),\n"
    "                     'avg_score': float((wp*sub['score']).sum())})\n"
    "    return pd.DataFrame(rows).sort_values('avg_score', ascending=False).reset_index(drop=True)\n"
    "\n"
    "def single_best_per_benchmark(df, weights):\n"
    "    rows = []; chosen = {}\n"
    "    for d, sub in df.groupby('dataset'):\n"
    "        prompts_d = sub['prompt_id'].unique()\n"
    "        cand = []\n"
    "        for m, sub_m in sub.groupby('model'):\n"
    "            if set(sub_m['prompt_id']) != set(prompts_d): continue\n"
    "            wd = sub_m['prompt_id'].map(weights)\n"
    "            cand.append({'model': m, 'score': float((wd*sub_m['score']).sum()),\n"
    "                         'cost':  float((wd*sub_m['cost']).sum())})\n"
    "        c = pd.DataFrame(cand).sort_values(['score','cost'], ascending=[False, True])\n"
    "        best_m = c.iloc[0]['model']; chosen[d] = best_m\n"
    "        for _, r in sub[sub['model']==best_m].iterrows():\n"
    "            rows.append({'prompt_id': r['prompt_id'], 'model': best_m, 'dataset': d,\n"
    "                         'weight': weights[r['prompt_id']],\n"
    "                         'score': float(r['score']), 'cost': float(r['cost'])})\n"
    "    a = pd.DataFrame(rows)\n"
    "    return {'policy': 'Single Best per Benchmark',\n"
    "            'selected_per_dataset': chosen,\n"
    "            'avg_cost':  float((a['weight']*a['cost']).sum()),\n"
    "            'avg_score': float((a['weight']*a['score']).sum()),\n"
    "            'assignments': a}"
))

cells.append(md("# **BC-2SP: Budget-Constrained Two-Stage Stochastic MILP**"))
cells.append(code(slice_block("def solve_bc2sp(")))

cells.append(md("# **QC-2SP: cost minimization under quality target**"))
cells.append(code(slice_block("def solve_qc2sp(")))

cells.append(md("# **DRO BC-2SP: box ambiguity over benchmark proportions**"))
cells.append(code(slice_block("def solve_dro_bc2sp(")))

cells.append(md("# **Plotting Functionality**"))

cells.append(code(
    "ALPHA_GRID = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]\n"
    "\n"
    "def alpha_sweep_oracle(df, weights):\n"
    "    prompts, models, _, Q, C, _ = make_dicts(df)\n"
    "    rows = []\n"
    "    for a in ALPHA_GRID:\n"
    "        cost = score = 0.0\n"
    "        for p in prompts:\n"
    "            best = min(models, key=lambda m: C[(p,m)] - a*Q[(p,m)])\n"
    "            cost  += weights[p] * C[(p, best)]\n"
    "            score += weights[p] * Q[(p, best)]\n"
    "        rows.append({'alpha': a, 'avg_cost': cost, 'avg_score': score})\n"
    "    return pd.DataFrame(rows)\n"
    "\n"
    "def alpha_sweep_single_best(df, weights):\n"
    "    rows = []\n"
    "    for a in ALPHA_GRID:\n"
    "        best_obj = best_c = best_s = None\n"
    "        for m, sub in df.groupby('model'):\n"
    "            wp = sub['prompt_id'].map(weights)\n"
    "            sm = (wp*sub['score']).sum(); cm = (wp*sub['cost']).sum()\n"
    "            obj = cm - a*sm\n"
    "            if best_obj is None or obj < best_obj:\n"
    "                best_obj, best_c, best_s = obj, cm, sm\n"
    "        rows.append({'alpha': a, 'avg_cost': best_c, 'avg_score': best_s})\n"
    "    return pd.DataFrame(rows)\n"
    "\n"
    "def alpha_sweep_single_best_per_benchmark(df, weights):\n"
    "    rows = []\n"
    "    for a in ALPHA_GRID:\n"
    "        total_c = total_s = 0.0\n"
    "        for d, sub in df.groupby('dataset'):\n"
    "            best_obj = best_c = best_s = None\n"
    "            for m, sub_m in sub.groupby('model'):\n"
    "                wp = sub_m['prompt_id'].map(weights)\n"
    "                sm = (wp*sub_m['score']).sum(); cm = (wp*sub_m['cost']).sum()\n"
    "                obj = cm - a*sm\n"
    "                if best_obj is None or obj < best_obj:\n"
    "                    best_obj, best_c, best_s = obj, cm, sm\n"
    "            total_c += best_c; total_s += best_s\n"
    "        rows.append({'alpha': a, 'avg_cost': total_c, 'avg_score': total_s})\n"
    "    return pd.DataFrame(rows)\n"
    "\n"
    "def plot_frontier(df_oracle, df_sb, df_sbb, df_bc, output_path):\n"
    "    fig, ax = plt.subplots(figsize=(8.0, 4.8))\n"
    "    ax.plot(df_oracle['avg_cost'], df_oracle['avg_score'],\n"
    "            'o-', color='C0', lw=2.0, label='Oracle (per-prompt best)')\n"
    "    ax.plot(df_sb['avg_cost'], df_sb['avg_score'],\n"
    "            's--', color='C1', lw=1.6, label='Single Best')\n"
    "    ax.plot(df_sbb['avg_cost'], df_sbb['avg_score'],\n"
    "            '^--', color='C2', lw=1.6, label='Single Best per Benchmark')\n"
    "    ax.plot(df_bc['avg_cost'], df_bc['avg_score'],\n"
    "            'D-.', color='C3', lw=2.0, label='BC-2SP (ours)')\n"
    "    ax.set_xlabel('Weighted Average Cost (USD per prompt)')\n"
    "    ax.set_ylabel('Weighted Average Score')\n"
    "    ax.set_title('Routing Policies on the Cost-Performance Plane')\n"
    "    ax.grid(True, alpha=0.3); ax.legend(loc='lower right')\n"
    "    fig.tight_layout(); fig.savefig(output_path, dpi=180)\n"
    "    plt.show()"
))

cells.append(md("# **Experiments**"))

cells.append(code(
    "df = restrict_to_full_coverage(load_data(CSV_PATH))\n"
    "prompts, models, datasets, Q_all, C_all, dom = make_dicts(df)\n"
    "weights = make_uniform_weights(prompts)\n"
    "print(f'prompts={len(prompts)}, models={len(models)}, datasets={datasets}')\n"
    "\n"
    "oracle = oracle_routing(df, weights)\n"
    "sb_table = single_best(df, weights)\n"
    "sbb = single_best_per_benchmark(df, weights)\n"
    "print(f'Oracle: score={oracle[\"avg_score\"]:.4f}, cost=${oracle[\"avg_cost\"]:.6f}')\n"
    "print(f'Best single model: {sb_table.iloc[0][\"model\"]} -> '\n"
    "      f'score={sb_table.iloc[0][\"avg_score\"]:.4f}, cost=${sb_table.iloc[0][\"avg_cost\"]:.6f}')\n"
    "print(f'SBB: score={sbb[\"avg_score\"]:.4f}, cost=${sbb[\"avg_cost\"]:.6f}')"
))

cells.append(code(
    "# BC-2SP cost-quality frontier\n"
    "rows = []\n"
    "for B in B_GRID:\n"
    "    r = solve_bc2sp(df, weights, B=B, K=K_DEFAULT)\n"
    "    rows.append({'B': B, 'avg_cost': r['avg_cost'], 'avg_score': r['avg_score'],\n"
    "                 'pool_size': r['pool_size'], 'pool': '|'.join(r['selected_pool'])})\n"
    "df_bc = pd.DataFrame(rows)\n"
    "df_bc"
))

cells.append(code(
    "# Alpha-swept curves for the comparison plot\n"
    "df_oracle = alpha_sweep_oracle(df, weights)\n"
    "df_sb     = alpha_sweep_single_best(df, weights)\n"
    "df_sbb    = alpha_sweep_single_best_per_benchmark(df, weights)\n"
    "plot_frontier(df_oracle, df_sb, df_sbb, df_bc, OUTPUT_DIR / 'frontier.png')"
))

cells.append(code(
    "# RQ1: diminishing returns of K\n"
    "K_grid = list(range(1, 11))\n"
    "rows = []\n"
    "for K in K_grid:\n"
    "    r = solve_bc2sp(df, weights, B=5e-3, K=K)\n"
    "    rows.append({'K': K, 'avg_score': r['avg_score'], 'avg_cost': r['avg_cost'],\n"
    "                 'pool_size': r['pool_size'], 'pool': '|'.join(r['selected_pool'])})\n"
    "df_K = pd.DataFrame(rows); df_K"
))

cells.append(code(
    "# RQ2: distributional robustness\n"
    "rows = []\n"
    "for delta in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]:\n"
    "    r = solve_dro_bc2sp(df, weights, B=5e-3, K=K_DEFAULT, delta=delta)\n"
    "    rows.append({'delta': delta, 'nominal_score': r['avg_score_nominal'],\n"
    "                 'worst_case_score': r['worst_case_score'],\n"
    "                 'pool_size': r['pool_size'], 'pool': '|'.join(r['selected_pool'])})\n"
    "df_dro = pd.DataFrame(rows); df_dro"
))

cells.append(code(
    "# Constraint ablations: storage and provider rules\n"
    "rows = []\n"
    "for S_max in [None, 50, 20]:\n"
    "    r = solve_bc2sp(df, weights, B=5e-3, K=K_DEFAULT, S_max=S_max)\n"
    "    rows.append({'S_max_GB': S_max if S_max else 'inf',\n"
    "                 'avg_score': r['avg_score'], 'avg_cost': r['avg_cost'],\n"
    "                 'pool_size': r['pool_size']})\n"
    "for setting, kwargs in [('free', {}),\n"
    "                         ('no OpenAI', {'must_exclude': ['gpt-5', 'gpt-5-chat']}),\n"
    "                         ('must Claude', {'must_include': ['claude-sonnet-4']})]:\n"
    "    r = solve_bc2sp(df, weights, B=5e-3, K=K_DEFAULT, **kwargs)\n"
    "    rows.append({'S_max_GB': setting, 'avg_score': r['avg_score'],\n"
    "                 'avg_cost': r['avg_cost'], 'pool_size': r['pool_size']})\n"
    "pd.DataFrame(rows)"
))

cells.append(code(
    "# QC-2SP: dual variant\n"
    "rows = []\n"
    "for Q in [0.80, 0.90, 0.95, 0.98]:\n"
    "    r = solve_qc2sp(df, weights, Q_target=Q, K=K_DEFAULT)\n"
    "    if r.get('infeasible'):\n"
    "        rows.append({'Q_target': Q, 'infeasible': True}); continue\n"
    "    rows.append({'Q_target': Q, 'avg_score': r['avg_score'], 'avg_cost': r['avg_cost'],\n"
    "                 'pool_size': r['pool_size']})\n"
    "pd.DataFrame(rows)"
))

# Build notebook
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}
    },
    "cells": cells,
}
# Convert each source string to a list-of-lines (Jupyter convention)
for c in nb["cells"]:
    s = c["source"]
    if isinstance(s, str):
        c["source"] = s.splitlines(keepends=True)

out = Path("/workspace/optimization/code/solution.ipynb")
out.write_text(json.dumps(nb, indent=1))
print("Wrote", out, "with", len(nb["cells"]), "cells",
      "(md+code:", sum(c["cell_type"]=="markdown" for c in nb["cells"]),
      "+", sum(c["cell_type"]=="code" for c in nb["cells"]), ")")
