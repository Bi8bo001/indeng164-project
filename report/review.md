宝宝，我认真对照了一遍 **project requirement + 你们 report 的内容**。结论先说：

## 总体判断

这份 report 的 **内容质量已经很强**，不是“刚好完成要求”的水平，而是明显在冲高分。
从 project rubric 看，最核心的几个点你们都覆盖了：

* **two-stage decision**：model pool selection + prompt routing；
* **stochastic optimization**：population form + SAA；
* **robustness**：DRO / benchmark mixture shift；
* **realistic constraints**：pool size、storage、fairness slack、provider rules；
* **Pyomo implementation**：明确写了 Pyomo + HiGHS；
* **visualization / frontier / ablation**：Pareto frontier、pool-size sweep、DRO sweep、constraint ablation；
* **business recommendations**：明确给了 production deployment 建议。

课程要求里明确说要提出 constrained optimization model，而不是 starter code 那种简单 (C-\alpha Q) weighted objective；并且要求模型包含 stochastic elements 和 multi-stage decisions。你们现在的报告正好把问题建成 two-stage stochastic MILP，并用 SAA 近似，这个方向非常对。

如果只看内容和方法，我会估计这是 **18–19+/20 的潜力稿**。但有几个地方需要修一下，否则可能被严谨的 grader 抓到。

---

# 1. Rubric 对照评分

## 1. Introduction：很完整，基本满分

Project 要求 introduction 里介绍 AI company、central problem、research questions，并说明为什么 two/multi-stage stochastic optimization 适合这个 context。

你们做得很好：

* 设定了 CodePilot AI；
* 明确说它是 Cursor-style coding assistant + STEM tutoring sidebar；
* 解释了 pool curation 和 routing 是两个不同时间尺度的决策；
* 明确写了 (y) 是 first-stage，(x_\xi) 是 second-stage；
* 提出两个 RQ：

  * RQ1: pool size / composition / diminishing returns；
  * RQ2: distributional robustness under prompt mix shift。

这一段内容很贴题，而且不是空泛背景。它直接把业务问题转成 OR structure。报告里也明确说 prompt distribution unknown，只观察到 240 samples，所以用 sample-average approximation，这很符合项目要求。

**评价：3/3 左右。**

---

## 2. Optimization Model：强，但有几个需要微调的严谨点

Project 对 optimization model 的要求最高，占 10 分：要定义 notation、decision variables、objective、constraints；要有 stochastic/robust/chance-constrained program；要从 population form 推到 solvable SAA form；如果有 linearization，需要推导。

你们这一部分整体非常强：

* 定义了 (y_m)、(x_{pm})、(s_p)；
* 有 population two-stage program；
* 有 SAA deterministic equivalent MILP；
* 有 budget-constrained formulation；
* 有 fairness slack；
* 有 storage cap；
* 有 provider rules；
* 有 DRO extension；
* 有 linearization；
* 还补充了 QC-2SP、randomized BC-2SP、cascade BC-2SP。

这些已经明显超过一个普通 deterministic MILP。尤其 project 明确说 simple deterministic model 最多 15 分，你们加入了 SAA、DRO、多约束、多 formulation 对比，已经避开这个问题。

不过这里有几个建议：

---

## 问题 A：DRO reformulation 的表述要更小心

你们写：

[
\min_w \sum_d w_d f_d(x)
========================

## \sum_d w_d^0 f_d(x)

\delta \cdot \min_\lambda \sum_d w_d^0 |f_d(x)-\lambda|.
]

这个思路在 **relative box + simplex equality** 下是有道理的，本质上是 bounded density perturbation 下的 worst-case expectation，weighted median 也合理。

但现在报告里说：

> Following the spirit of distributionally robust optimisation (Mohajerin Esfahani and Kuhn, 2018)

这个 citation 是 Wasserstein DRO 经典文献，但你们实际做的不是 Wasserstein ball，而是 **relative-box ambiguity set over benchmark mixture**。这不是大问题，但建议不要让老师觉得 citation 和 formulation 不完全对应。

建议改成：

> Motivated by distributionally robust optimization, we use a simple relative-box ambiguity set over the four benchmark mixture weights.

这样更准确。不要暗示自己用了 Wasserstein DRO。你们 personal takeaway / future work 里说以后做 Wasserstein DRO，反而说明现在不是 Wasserstein，这很好。

---

## 问题 B：Population model 和 SAA model 的 objective 不完全一致

Population form 里你们写的是：

[
\max \mathbb{E}[Q]
]

subject to budget、pool cap、storage cap。

但 SAA form 里变成：

[
\max \sum Qx - \lambda \sum s - \epsilon \sum Cx.
]

也就是 SAA 里多了 fairness slack penalty 和 cost tie-breaker。

这个不致命，但最好在文字中解释：

> The SAA model augments the population model with two implementation-level terms: a soft fairness penalty and a small cost tie-breaker.

否则严谨 grader 可能会问：population form 里没有 (s_p)，为什么 SAA form 里突然出现了 fairness target？

建议加一句即可，不需要大改。

---

## 问题 C：(\lambda) 和 (\epsilon) 的尺度需要再解释一点

你们写：

> (\lambda=1) unless otherwise stated，(\epsilon=10^{-3})。

因为 (Q_{pm}\in {0,1})，(\lambda=1) 确实能让错题 slack 有大惩罚。但这里有一个细节：

如果 (\tau=1)，那么任何 prompt 只要选错模型就产生 (s_p=1)，惩罚等价于再扣一次 quality。
目标会变成：

* correct：quality 1, slack 0 → contribution 1；
* wrong：quality 0, slack 1 → contribution (-1)。

这会非常强烈地惩罚错误。不是错，但需要说明这是 intentional：你们希望 solver strongly prefer correctness whenever feasible。

另外 report 目前没有清楚写 (\tau) 实验中取多少。你们 notation 里有 (\tau)，model 里有 (\tau)，但 experiments 只说 (\lambda=1)，没说 (\tau)。建议必须补一句：

> We set (\tau=1), so the slack records whether a prompt is left unsolved by its assigned model.

或者如果你们实际不是 1，就写真实值。这个会影响 model interpretation。

---

## 问题 D：Randomized and Cascade 只是 companion formulations，内容略薄

Project 里 cascading 是 optional，你们已经写了，很加分。

但现在 companion formulations 那一段很短，只说了：

> Cascade BC-2SP pairs ((m_1,m_2)) per prompt with a zero-cost verifier and escalates only on failure.

后面实验里又说 cascade marginal lift。这个可以接受，但如果想更稳，可以加一两句说明 cascade 的 expected cost / quality 怎么算：

如果 first model correctness 是 (Q_{pm_1})，zero-cost verifier 能检测 failure，则：

[
\text{expected quality} =
Q_{pm_1} + (1-Q_{pm_1})Q_{pm_2}
]

[
\text{expected cost} =
C_{pm_1} + (1-Q_{pm_1})C_{pm_2}
]

因为 (Q) 是 binary correctness，这个 formulation 会很自然。加这两行会让 cascade 部分从“提到过”变成“真的建模了”。

---

# 3. Data handling：基本合理，但要注意一个潜在扣分点

Project 原始数据是 **33 models, 240 prompts**。
你们 report 里说因为一个模型只覆盖 180 prompts，所以丢掉它，使用 32 fully-covered models。

这个处理是合理的，因为避免 ad hoc imputation。但有一个风险：

项目原话是 “curates a small but effective pool of models from a large candidate set”，而你们删掉一个 candidate model。如果 grader 很在意 “full candidate set”，可能会问为什么不用 missing-value strategy。

建议在 data section 或 limitation 加一句：

> We checked that the dropped model does not affect the main conclusions; alternatively, it could be included by imputing missing prompt-model pairs as infeasible assignments.

如果你们没有真的 check，就写保守一点：

> We discard it only to keep all prompt-model entries comparable; including it with infeasible assignments would be a straightforward extension.

这样更稳。

---

# 4. Experiment design：很强，覆盖得很全

Project 要求如果 solution 依赖 hyperparameters，要画 optimal frontier；如果做 ablation / parameter tuning，要展示发现。

你们做了：

* cost-quality Pareto frontier；
* budget sweep；
* pool size (K) sweep；
* pool composition；
* per-benchmark breakdown；
* DRO (\delta) sweep；
* storage/provider ablation；
* formulation comparison；
* baseline comparison。

这非常充分。

尤其是这几个结果很有说服力：

* (B=0.001) 时 score 0.967，cost 比 GPT-5 低很多；
* (B=0.005) 时达到 oracle score 0.983；
* (K=6) 已经接近 saturation；
* GPT-5 只处理 1.7% traffic，作为 backstop；
* distribution shift 到 (\delta=0.7) 也只损失约 1.17 pp worst-case score。

这部分是整篇 report 的最大优势。它不是只列一个 optimal solution，而是通过 frontier 和 ablation 说明了 **why this policy is production-suitable**。

---

# 5. 结果解释：整体很强，但有几处措辞/逻辑需要更严谨

## 问题 A：“Oracle” 的定义要更明确

你们用 oracle 作为 upper bound，这很好。但 Table 1 里：

* BC-2SP at (B=0.005): score 0.9833, cost 0.00323；
* Oracle: score 0.9833, cost 0.00170。

这里看起来 oracle 同样 score 但 cost 更低。所以严格来说，BC-2SP 达到了 oracle **quality**，但没有达到 oracle **cost**。

你们现在写：

> the policy attains the per-prompt oracle of 0.983

这个可以，但最好改成：

> attains the oracle quality score of 0.983

这样避免别人误解为整个 cost-quality pair 也达到了 oracle。

同理 abstract 里：

> drives the score to the per-prompt oracle of 0.983

可以，没问题；但正文要明确是 score oracle。

---

## 问题 B：“free open-source models” 的 cost=0 要解释部署成本和 inference cost 的区别

Report 里说九个模型 cost 为 0，因为 open-source 可以本地 serve。这个来自数据设定，但现实里本地 serve 并不是真正 zero cost，有 GPU amortization / latency / maintenance。你们用 storage cap 补了一部分现实性，但建议在 discussion 或 limitation 里加一句：

> We follow the dataset convention and treat open-source models as zero API-cost; in production, GPU serving cost could be added as another cost term.

这会显得更成熟，避免老师觉得你们过度宣称 “free”。

---

## 问题 C：“single model achieves only 0.742” 和 “Single Best GPT-5 0.8875” 之间可能冲突

正文 RQ1 里写：

> A single model achieves only 0.742 because no model alone covers all four benchmarks adequately.

但 Table 1 里 single best GPT-5 score 是 0.8875。

这看起来矛盾。可能原因是：

* RQ1 的 (K=1) sweep 在 (B=0.005) 下，GPT-5 因为 cost 太高不可行，所以只能选便宜模型，score 0.742；
* Table 1 的 Single Best GPT-5 没有 budget constraint，只是 baseline。

这个逻辑是通的，但现在没解释清楚。建议改成：

> Under the (B=0.005) budget, a one-model pool achieves only 0.742, because the best single model GPT-5 is too expensive to be feasible and no budget-feasible model covers all four benchmarks adequately.

这非常重要。否则 grader 可能以为你们数字不一致。

---

## 问题 D：Recommended setting 有点不一致

Table 1 caption 说 bolded row 是 recommended setting，但 parsed text 里看不出哪行 bold。正文 recommendation 又说：

> Maintain a six-model pool...

但实验主 setting 是 (K=8, B=0.005)，DRO 也用 (K=8)。这里有一点点 tension：

* RQ1 说 (K=6) sufficient；
* Figure/experiments 大多展示 (K=8)；
* recommendation 说 maintain six-model pool；
* abstract 说 (K=6) captures 99.7%。

这可以，但建议把 recommendation 写得更精确：

> Use (K=6) as the cost-conscious production default, and keep (K=8) as a diagnostic/upper-frontier setting for analysis.

或者：

> We recommend (K=6) when engineering overhead matters, since it captures nearly all attainable quality; (K=8) is useful when the firm wants to exactly match the oracle quality under the studied budget.

这样逻辑就完整了。

---

# 6. 是否满足 “at least two realistic constraints”？

满足，而且超过要求。

Project 给的 realistic constraints 示例包括：

* limited model pool size；
* storage/deployment constraints；
* provider/contract constraints；
* prompt coupling；
* fairness / minimum quality；
* worst-case performance。

你们用了：

1. pool-size cap；
2. storage budget；
3. fairness slack；
4. provider rules；
5. DRO worst-case mixture shift。

这部分可以算强项。
唯一建议是：provider rules 现在只在 model 部分一句 “fixing (y_m)” 和 ablation 里出现。可以稍微说清楚：

[
y_m = 0 \quad \forall m \in \text{OpenAI}
]

用于 exclusion scenario；

[
y_{\text{Claude}} = 1
]

用于 must-include contract scenario。

不用展开太多，但这样更像正式 constraint。

---

# 7. Pyomo / code 部分：基本达标，但 report 里还可以补一个 result-output checklist

Project 要求 “Attach your Pyomo implementation used to solve your model. Present optimal solutions in the code and report.”

你们写了：

* implemented in Pyomo；
* solver HiGHS；
* size: 8,224 binary variables / 8,163 constraints；
* solve time；
* notebook accompanies report；
* GitHub link in abstract。

这已经满足。但为了更稳，建议在 code 或 appendix 里明确输出：

* selected model pool；
* objective value；
* average score；
* average cost；
* usage share；
* per-benchmark score；
* constraint status；
* solver status / optimality gap。

报告正文不一定要写，但 notebook 里最好有。

---

# 8. 最大的内容风险：页数

Project 要求 **4–5 page comprehensive project report**。
你们现在 PDF parsed 是 7 页，其中：

* main text 到 page 5；
* references page 6；
* appendix page 7。

如果老师严格按 “4–5 page report” 算正文，不一定扣；但如果总页数限制包括 appendix/reference，可能有风险。

建议处理方式：

* 保持 main text 5 页以内；
* references 和 appendix 如果允许额外页，没问题；
* 如果不确定，最好把 appendix 标成 “Additional Numerical Results” 并确认 Gradescope 是否允许 appendix。
* 如果必须 5 页总长，那需要压缩 references 或把 appendix 放 code notebook，不放 report。

内容上值得保留的优先级：

1. model formulation；
2. frontier；
3. pool-size sweep；
4. DRO；
5. recommendation。

可以牺牲的：

* companion formulation 细节；
* references 过多；
* detailed numerical appendix。

---

# 9. 我觉得最需要改的 6 个点

按优先级排序：

## 1. 明确 (\tau) 的取值

现在 model 里有 fairness target，但 experiments 没写 (\tau)。这是最容易被问的。

加一句：

```latex
We set \(\tau=1\), so \(s_p\) measures the violation of a per-prompt correctness guarantee.
```

或者写真实值。

---

## 2. 解释 (K=1) score 0.742 和 GPT-5 score 0.8875 的区别

必须避免看起来数字冲突。

改成：

```latex
Under the \(B=\$0.005\) budget, a one-model pool achieves only 0.742, because GPT-5 is not budget-feasible as an always-on model.
```

---

## 3. 把 “attains oracle” 改成 “attains oracle quality”

更严谨。

---

## 4. DRO citation / wording 改准确

把 “Following the spirit of Wasserstein DRO” 类感觉削弱，明确是 relative-box ambiguity set。

---

## 5. Recommendation 里协调 (K=6) vs (K=8)

现在实验主图是 (K=8)，recommendation 是 (K=6)。要解释二者关系。

---

## 6. 补一句 open-source zero cost 的 limitation

避免现实性问题。

```latex
We follow the dataset convention that open-source models have zero API cost; GPU serving cost could be added as an additional deployment-cost term.
```

---

# 10. 这份 report 的质量定位

我会这样判断：

| 项目                      | 当前质量                                          |
| ----------------------- | --------------------------------------------- |
| 选题贴合度                   | 很高                                            |
| OR modeling             | 很强                                            |
| stochastic / robust 部分  | 强，有一点表述需精确                                    |
| multi-stage decision    | 明确                                            |
| constraints realism     | 强                                             |
| experiments             | 很完整                                           |
| visualization           | 充分                                            |
| business recommendation | 有说服力                                          |
| 潜在扣分点                   | 少量 notation / result consistency / page limit |
| 总体潜力                    | 高分稿                                           |

如果按 20 分粗估：

| Rubric                       |               分数预估 |
| ---------------------------- | -----------------: |
| Introduction                 |              3 / 3 |
| Optimization Model           |       8.5–9.5 / 10 |
| Pyomo Code                   |              2 / 2 |
| Presentation & Visualization |              2 / 2 |
| Discussion & Recommendations |          1.7–2 / 2 |
| Personal Takeaway            |              1 / 1 |
| **Total**                    | **18.2–19.5 / 20** |

扣分主要不会来自“做得不够”，而是来自 **严谨性小缝隙**：(\tau) 未说明、oracle wording、DRO wording、(K=1) 数字解释、(K=6/K=8) recommendation 关系。

---

# 最后一句判断

内容层面，这已经是一份 **很完整、很有 OR 味、也有真实 deployment insight 的 project report**。不是堆公式，而是真的把 LLM routing 转成了一个两阶段随机优化问题，并且用 frontier、pool-size sweep、DRO 和 ablation 把 business conclusion 支撑起来了。

现在最值得做的不是大改方法，而是做一轮 **严谨性修补 + 逻辑一致性修补 + 压缩表达**。这样会从“很强”变成“几乎没有明显扣分点”。
