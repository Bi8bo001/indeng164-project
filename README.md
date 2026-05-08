# Curating a Robust LLM Pool: A Two-Stage Stochastic Optimization Approach for Cost-Constrained LLM Routing

Berkeley INDENG 164 Final Project, Spring 2026

Jingwen Yang, Yu Hin Liang

- Code: [`submit.ipynb`](submit.ipynb)
- Report: [`report.pdf`](report.pdf)

## Headline result

![Cost-quality Pareto frontier](report/figure/fig1_frontier.png)

| Policy | Score | Cost (\$/prompt) | Pool size |
|---|---:|---:|---:|
| Single Best (gpt-5)            | 0.8875 | 0.04255 | 1 |
| Single Best per Benchmark       | 0.8875 | 0.03782 | 4 |
| BC-2SP (ours), B = \$0.001        | 0.9667 | 0.00082 | 8 |
| **BC-2SP (ours), B = \$0.005**    | **0.9833** | **0.00323** | **8** |
| Oracle (per-prompt best)        | 0.9833 | 0.00170 | 32 |

At a budget of \$0.005 per prompt, the BC-2SP policy attains the per-prompt
oracle quality (0.983), which is **13× cheaper than running gpt-5 on every
prompt** at **9.5 percentage points higher** quality. A pool of K = 6 already
captures 99.7% of the achievable quality, and a ±70% shift in benchmark
proportions costs at most 1.2 pp of worst-case quality.
