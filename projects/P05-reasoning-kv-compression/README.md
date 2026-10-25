# P05 · 推理模型的解码期 KV Cache 压缩

> **难度** ★★★★ ｜ **算法 : 系统** 5 : 5 ｜ **建议人数** 1–2 ｜ **最低硬件** 1×24GB ｜ **基座** vLLM 或 SGLang（也可先在 nano-vllm / mini-sglang 上验证）

## 1. 背景

推理模型（DeepSeek-R1、Qwen3 thinking 模式等）的特点是**输入短、输出极长**：一道数学题可能生成上万 token 的思维链。KV Cache 随输出线性增长，会限制并发 batch 大小，进而限制吞吐。

经典的 KV 压缩方法（SnapKV、H2O、StreamingLLM 等）主要针对**长输入**，在 prefill 后压缩一次。推理模型需要的是**解码期持续压缩**，而且思维链中有大量重复、自我验证的内容。近期出现了一批专门针对推理模型的方法：

- **R-KV**：结合注意力重要性与**冗余度**（key 向量相似度）打分，论文报告只保留 10% KV 就接近完整精度，吞吐提升 6.6×，batch 可增大 13×；
- **R-KVHash**（ICLR 2026）：用 SimHash 估计冗余度，降低 R-KV 的打分开销，解码吞吐再提升最多 2×；
- **ReasonAlloc**（2026）：解码期分层 KV 预算分配。

但这些方法的"吞吐提升"大多来自 HF 风格的实现（连续 KV 张量、静态 batch）。**在 Paged KV + Continuous Batching 的真实引擎中，解码期 token 淘汰面临根本性的冲突**，这正是本项目要解决的问题。

## 2. 论文 → 引擎的鸿沟

| 问题 | 说明 |
|---|---|
| 注意力分数不可得 | FlashAttention / paged decode kernel 不输出注意力权重；需要额外计算或修改 kernel |
| Paged KV 与 token 淘汰 | 淘汰的是零散的 token，而 KV 按 block 管理。需要做 block 内整理（compaction），或改用 block 级淘汰 |
| 每层 / 每 head 保留不同 token | 论文中常见，但引擎的 block table 通常所有层共享，直接实现会破坏内存管理 |
| 位置编码 | 淘汰后 RoPE 位置如何处理（保留原位置 vs 重新编号） |
| 与 prefix caching 冲突 | 压缩后的 KV 不再对应原始前缀 |
| 收益兑现 | 省下的显存要真正转化为更大的 batch，调度器必须知道压缩后的真实 KV 占用 |

## 3. 项目目标

1. 在 HF 上复现一种推理模型 KV 压缩方法（推荐 R-KV 或 R-KVHash），对齐论文精度；
2. 在 vLLM / SGLang 中实现**解码期 KV 压缩**，与 paged attention、continuous batching 兼容；
3. 让省下的显存真正转化为**更大的 batch 与更高的吞吐**；
4. 鸿沟分析，并尝试"系统友好"的算法改进（例如 block 级淘汰、低开销打分）。

## 4. 任务分解

### 4.1 阶段一：算法复现（必做）
- 模型：DeepSeek-R1-Distill-Qwen-1.5B / 7B 或 Qwen3-1.7B / 8B（thinking 模式）；
- 在 HF 上复现 R-KV（或 R-KVHash）：AIME24 / MATH500 / GSM8K 的精度随 KV 预算的变化；
- 同时测出"论文式"吞吐提升（作为对照）。

### 4.2 阶段二：引擎落地（必做）
- 设计文档（核心）：
  - 打分所需信息怎么得到？（例如只对最近一个窗口的 query 计算注意力分数，或用 key 相似度等不依赖注意力的打分）
  - 压缩多久触发一次？触发时如何在 block 内整理 KV、回收空 block？
  - 位置编码怎么处理？
  - 如何让调度器感知压缩后的 KV 占用，从而接纳更多请求？
- 实现：压缩算子（Triton 或 PyTorch）、block 整理与回收、与调度器的交互；
- 正确性：
  - 预算设为无穷大时，输出与原引擎一致；
  - 精度与 HF 参考实现对齐（同一 KV 预算下）。

### 4.3 阶段三：鸿沟分析与改进（必做分析，改进为加分项）
- 吞吐 vs 并发度：原引擎 / 你的实现（不同 KV 预算）；
- 开销分解：打分、整理、调度各占多少；
- 系统友好的改进：
  - **block 级淘汰**：以 block 为单位打分和淘汰，免去整理的开销，分析精度代价；
  - **低开销打分**：用 hash、量化、采样等方法降低打分开销；
  - **层间共享的淘汰决策**：所有层保留相同的 token 位置，降低内存管理复杂度。

## 5. 关键知识点（汇报必须讲清楚）

**算法**
- 所选方法的打分机制（重要性 + 冗余度）及其背后的观察（推理链中的冗余）；
- 解码期压缩与 prefill 后压缩（SnapKV）的区别；
- KV 压缩为什么在推理任务上比在检索类任务上更"安全"。

**系统**
- vLLM / SGLang 的 KV Cache 管理：block 分配、block table、空闲 block 回收；
- paged decode attention kernel 如何按 block table 读 KV；
- KV 显存与最大 batch、吞吐之间的定量关系；
- 为什么零散的 token 淘汰在 paged KV 中代价高。

## 6. 实验设计

- **模型**：DeepSeek-R1-Distill-Qwen-7B 或 Qwen3-8B（thinking）；
- **任务**：AIME24 / AIME25、MATH500、GSM8K；
- **必须产出**：
  1. 精度 vs KV 预算（HF 参考 vs 引擎实现）；
  2. 吞吐 vs 并发度（不同 KV 预算），以及最大可支持 batch；
  3. 开销分解；
  4. token 级 vs block 级淘汰的精度-吞吐对比（如做了改进）；
  5. 鸿沟瀑布图。

## 7. 答辩追问示例

1. 你的引擎实现中，打分用到的注意力信息是怎么来的？额外开销多少？
2. 一个 block 中淘汰了 10 个 token 之后，这个 block 怎么处理？什么时候能被回收？
3. 淘汰后 RoPE 位置怎么处理？两种选择对精度有什么影响？
4. 省下的 KV 显存是如何转化成更大 batch 的？调度器是怎么知道的？
5. 为什么 KV 预算 10% 时精度还能保持？在什么任务上会崩？
6. 论文的 6.6× 吞吐提升，你在引擎中做到了多少？差距在哪里？

## 8. Agent 使用建议

- 让 Agent 帮你先在 nano-vllm 这类精简引擎上实现原型，跑通后再迁移到 vLLM / SGLang；
- **风险**：block 整理与回收的边界条件（引用计数、正在被其他请求共享的 block）是高危区域。必须做"无限预算时输出一致"的回归测试和高并发压力测试；
- 精度评测的采样设置（temperature、采样次数）要与论文一致，推理任务方差大，需要多次采样取平均。

## 9. 参考资料

- R-KV：arXiv 2505.24133（官方代码见论文链接）
- R-KVHash（ICLR 2026）
- ReasonAlloc：arXiv 2606.11164
- SnapKV：arXiv 2404.14469；H2O：arXiv 2306.14048；StreamingLLM：arXiv 2309.17453
- PagedAttention / vLLM：arXiv 2309.06180
- kvpress（NVIDIA，HF 层面的 KV 压缩方法集合）：https://github.com/NVIDIA/kvpress
