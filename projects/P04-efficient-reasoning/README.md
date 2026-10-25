# P04 · 高效推理：思考早停与置信度剪枝的服务化

> **难度** ★★★ ｜ **算法 : 系统** 6 : 4 ｜ **建议人数** 1–2 ｜ **最低硬件** 1×24GB ｜ **基座** vLLM（Logits Processor / 引擎内请求控制）或 SGLang

## 1. 背景

推理模型（DeepSeek-R1、Qwen3 thinking 等）经常"想太多"：答案早已得出，模型还在反复验证。测试时扩展（test-time scaling）中的并行采样（如 self-consistency、多数投票）更是把计算量放大了 N 倍。**高效推理（Efficient Reasoning）** 因此成为热门算法方向：

- **单链早停**：DEER（在 "Wait" 等转折点检测置信度，决定是否提前结束思考）、EAT（追加 `</think>` 后看下一个 token 的熵，报告在 MATH-500 和 AIME-2025 上减少 13–21% token 而不掉点）、SpecExit（利用草稿模型的隐藏状态预测退出信号，提供了 vLLM 实现）；
- **思考预算控制**：vLLM 社区有为推理模型添加 reasoning budget logits processor 的 PR 与第三方实现；
- **并行采样的置信度剪枝**：DeepConf（Meta，ICLR 2026）在 vLLM 之上实现，用 token 级置信度在线终止低质量推理链。论文报告在 AIME 2025 上精度最高达到 99.9%，同时相比完整的并行思考减少最多 84.7% 的生成 token。

这些方法大多以"**节省了多少 token**"作为效率指标。但在真实服务中，**少生成 token 不等于同比例降低延迟或提升吞吐**：

## 2. 论文 → 引擎的鸿沟

| 问题 | 说明 |
|---|---|
| Token 节省 ≠ 吞吐提升 | 早停让请求提前结束，释放的 batch 槽位要被新请求填上，吞吐才会提升；延迟取决于最长的那条链 |
| 并行采样的长尾 | N 条链中最慢的那条决定请求延迟；剪枝必须剪掉"长尾" |
| 置信度计算的开销 | 需要每步读取 logprobs / 熵，或额外前向（例如 EAT 追加 `</think>` 探测、DEER 的试探性回答），这些都有代价 |
| 引擎接口 | 每步监控与终止需要 logits processor 或引擎内的请求控制；并行链共享前缀需要 prefix caching / `n` 采样的配合 |
| 精度评测的方差 | 推理任务方差大，需要多次采样才能得出可靠结论 |

## 3. 项目目标

1. 选择一种早停 / 剪枝方法（推荐单链方法 1 个 + DeepConf 类并行方法 1 个），在离线设置下复现论文的 token 节省与精度；
2. 将方法落地为引擎内的**在线**机制（logits processor 或引擎内的请求控制），而不是离线后处理；
3. 在在线服务负载下，测量**真实的延迟和吞吐收益**，并与"token 节省比例"对比；
4. 鸿沟分析 + 系统感知的改进（例如降低监控开销、批感知的剪枝策略）。

## 4. 任务分解

### 4.1 阶段一：算法复现（必做）
- 模型：DeepSeek-R1-Distill-Qwen-1.5B / 7B 或 Qwen3-1.7B / 8B（thinking 模式）；
- 任务：AIME24 / AIME25、MATH500、GSM8K、GPQA；
- 复现：精度 vs 生成 token 数；DeepConf 类方法的精度 vs 采样数 / 剪枝比例。

### 4.2 阶段二：引擎落地（必做）
- 单链早停：用 logits processor（注意 V1 / V2 model runner 接口差异）实现置信度监控与强制插入 `</think>`；需要额外探测前向的方法，要设计如何在引擎中高效实现；
- 并行剪枝：在引擎内维护同一问题多条链的置信度统计，在线终止低置信度的链（可参考 DeepConf 的 vLLM 实现，并评估其效率）；
- 正确性：关闭早停时与原引擎输出一致；早停逻辑的单元测试。

### 4.3 阶段三：鸿沟分析与改进（必做分析，改进为加分项）
- 在线负载（请求率扫描）下：请求延迟（P50 / P90）、系统吞吐（tokens/s、题 / 秒）、精度；
- 分解：token 节省带来的收益 vs 监控开销；并行采样中长尾链对延迟的影响；
- 系统感知改进：例如在高负载时更激进地剪枝、在低负载时保守；把监控计算融合进采样 kernel；与 prefix caching 结合以减少并行链的重复 prefill。

## 5. 关键知识点（汇报必须讲清楚）

**算法**
- 所选方法的置信度定义（熵、logprob、置信度分组统计等）及其背后的假设；
- 早停的风险：过早结束导致错误；如何在 token 节省与精度之间权衡；
- Self-consistency / 多数投票的原理，以及加权投票与置信度过滤的作用。

**系统**
- Logits processor 在引擎中的位置与执行时机；每步额外计算对吞吐的影响；
- continuous batching 下请求提前结束如何转化为吞吐提升；
- `n` 个并行采样如何共享前缀 KV（prefix caching / fork）；
- 为什么并行采样的延迟由最长链决定。

## 6. 实验设计

- **模型**：DeepSeek-R1-Distill-Qwen-7B 或 Qwen3-8B（thinking）；
- **任务**：AIME24 / 25、MATH500、GSM8K、GPQA-Diamond；每题多次采样；
- **必须产出**：
  1. 精度 vs 平均生成 token 数（离线）；
  2. 在线负载下：延迟、吞吐 vs 请求率（原引擎 vs 早停 / 剪枝）；
  3. "token 节省比例" vs "真实吞吐提升比例" 对比图，以及差距的解释；
  4. 监控开销分析。

## 7. 答辩追问示例

1. 你的方法节省了 40% 的 token，为什么吞吐只提升了 20%？
2. 并行采样 N=16 时，请求延迟由什么决定？你的剪枝如何缩短它？
3. 你的置信度信号在什么样的题目上会误判？举一个实际的例子。
4. Logits processor 每步要做什么计算？batch=128 时开销多大？
5. EAT 需要追加 `</think>` 探测下一个 token 的熵，这在引擎中怎么高效实现？

## 8. Agent 使用建议

- 评测流水线、多次采样统计、绘图交给 Agent；
- **风险**：推理任务的答案抽取与判分（数学答案的等价判定）很容易出错，影响所有结论，必须抽样人工检查；
- 精度结论要有足够的采样次数和置信区间，避免把噪声当成效果。

## 9. 参考资料

- DEER: Dynamic Early Exit in Reasoning Models：arXiv 2504.15895
- EAT: Entropy After `</think>` for reasoning model early exiting：arXiv 2509.26522
- SpecExit（利用投机解码草稿模型的隐藏状态做早停）
- Deep Think with Confidence（DeepConf，ICLR 2026）：arXiv 2508.15260 ；https://github.com/facebookresearch/deepconf
- Towards Concise and Adaptive Thinking in Large Reasoning Models: A Survey：arXiv 2507.09662
- Self-Consistency：arXiv 2203.11171
- vLLM Custom Logits Processors 文档（注意 V1 / V2 model runner 的接口差异）
