# P06 · 长上下文稀疏注意力的引擎落地

> **难度** ★★★★ ｜ **算法 : 系统** 5 : 5 ｜ **建议人数** 1–2 ｜ **最低硬件** 1×80GB ｜ **基座** vLLM 或 SGLang + Triton

## 1. 背景

长上下文场景（长文档、RAG、Agent、代码库理解）中，**prefill 的注意力计算是 O(n²)**，128K 上下文的 TTFT 可能达到数十秒。注意力矩阵实际上高度稀疏，因此出现了大量**免训练的稀疏注意力**方法：

- **Prefill 稀疏**：MInference（离线搜索 head 的稀疏模式 + 在线估计稀疏索引）、FlexPrefill（按输入和 head 自适应调整稀疏模式与计算预算）、XAttention（基于反对角线打分的块稀疏）、SharePrefill（跨 head 共享稀疏模式）等；
- **Decode 稀疏**：Quest（按 page 估计重要性、只读取 top-k page）、DuoAttention（检索 head 与流式 head 分离）等；
- **原生稀疏（需训练）**：NSA、MoBA、DeepSeek-V3.2 的 DSA。

ACL 2026 Findings 的 *Sparse Frontier* 研究指出：对免训练方法而言，**细粒度的 per-query 重要性估计在 prefill 中并不实用**。原因有两个：估计本身有开销，而且缺乏能把细粒度稀疏转化为真实加速的 kernel。**块到块的选择是更实际的路线。** 这是一个典型的"算法必须适配硬件"的例子。

此外，大多数方法的开源实现停留在 HF 或旧版 vLLM 上（例如 FlexPrefill 仓库依赖 vLLM 0.5.4），与当前引擎的 chunked prefill、prefix caching、paged KV 并不兼容。

## 2. 论文 → 引擎的鸿沟

| 问题 | 说明 |
|---|---|
| Chunked prefill | 长 prompt 被切成多个 chunk 分步计算；稀疏模式需要在 chunk 粒度上重新设计 |
| Paged KV | 稀疏 kernel 需要按 block table 读取 KV，而不是连续张量 |
| 稀疏模式的估计开销 | 估计本身的开销可能吃掉稀疏带来的收益，尤其在中等长度（8K–32K）下 |
| 与 batch 的交互 | 多个长请求同时 prefill 时，不同请求、不同 head 的稀疏模式不同，影响 kernel 效率 |
| 精度 | 稀疏度与精度的权衡，在 needle / 多跳检索类任务上最敏感 |

## 3. 项目目标

1. 选择一种免训练的 prefill 稀疏注意力方法，在 HF 上复现精度与 kernel 加速；
2. 在 vLLM / SGLang 中把它实现为一个 attention backend（或现有 backend 的扩展），支持 **chunked prefill 与 paged KV**；
3. 在真实长上下文负载下测量 TTFT 与吞吐的真实改善；
4. 鸿沟分析，并基于系统约束改进稀疏模式的设计。

## 4. 候选算法池

| 方法 | 说明 |
|---|---|
| XAttention | 反对角线打分 + 块稀疏，块粒度天然适合 kernel |
| FlexPrefill | 自适应模式选择；需要处理 query-aware 模式在 chunk 上的估计 |
| MInference | 三种稀疏模式（A-shape、vertical-slash、block-sparse）+ 离线 head 配置 |
| SharePrefill | 跨 head 共享稀疏模式，降低估计开销 |
| Quest（decode） | 如果更关注长上下文 decode，可以选择这条路线：page 级 top-k 读取，与 paged KV 天然契合 |

## 5. 任务分解

### 5.1 阶段一：算法复现（必做）
- 模型：Llama-3.1-8B-Instruct / Qwen3-8B（支持 128K）；
- HF 上复现：RULER（多长度）、LongBench、needle-in-a-haystack 的精度；kernel 级加速（与 FlashAttention 对比）。

### 5.2 阶段二：引擎落地（必做）
- 读懂锁定版本中 attention backend 的接口、metadata 构建、chunked prefill 的执行方式；
- 用 Triton 实现**支持 paged KV 的块稀疏 prefill attention kernel**（可基于 Triton 的 FlashAttention 实现修改）；
- 实现稀疏模式估计（在每个 chunk 上），并接入引擎；
- 正确性：稀疏度为 0 时与原 backend 输出一致；精度与 HF 参考实现对齐。

### 5.3 阶段三：鸿沟分析与改进（必做分析，改进为加分项）
- TTFT vs 上下文长度（8K → 128K）：原引擎 / 你的实现；
- 开销分解：模式估计、稀疏 kernel、其他；
- 找出稀疏注意力开始有收益的**最短上下文长度**；
- 系统感知改进：例如降低估计频率（跨 chunk / 跨层复用稀疏模式）、调整块粒度以匹配 kernel tile、按 head 类型分组以提升 batch 内的规整性。

## 6. 关键知识点（汇报必须讲清楚）

**算法**
- 注意力稀疏性的来源与几种典型模式（attention sink、局部窗口、垂直线、斜线）；
- 所选方法的模式估计机制及其近似误差；
- 为什么稀疏注意力在检索类任务上更容易失败。

**系统**
- FlashAttention 的 tiling 与 online softmax，以及块稀疏如何在其中实现（跳过 tile）；
- 为什么细粒度稀疏难以转化为真实加速（tile 粒度、warp 利用率）；
- Chunked prefill 下的注意力计算：当前 chunk 的 query 对所有历史 KV；
- Amdahl 定律：attention 加速对端到端 TTFT 的贡献上限（MLP 部分不变）。

## 7. 实验设计

- **模型**：Llama-3.1-8B-Instruct / Qwen3-8B；
- **任务**：RULER（4K–128K）、LongBench、InfiniteBench 子集；
- **必须产出**：
  1. 精度 vs 稀疏度（HF vs 引擎）；
  2. TTFT vs 上下文长度，标出收益起点；
  3. 端到端 TTFT 分解（attention / MLP / 估计开销）；
  4. 长请求与短请求混合负载下的在线吞吐与 TTFT 分布。

## 8. 答辩追问示例

1. 128K 上下文下，attention 在 prefill 总计算中占多大比例？即使 attention 完全免费，TTFT 最多能降多少？
2. 你的稀疏 kernel 的块大小是多少？为什么选这个值？
3. chunked prefill 下，稀疏模式是每个 chunk 估计一次吗？估计的开销是多少？
4. 在 needle-in-a-haystack 上，你的方法在什么位置 / 长度下失败？为什么？
5. 你的方法和 DeepSeek DSA / NSA 这类需要训练的稀疏注意力相比，有什么优劣？

## 9. Agent 使用建议

- Agent 可以帮你从 Triton FlashAttention 教程出发改出块稀疏 kernel，但**边界处理（不满一个 tile、paged block 跨越 tile）**要重点审查，并与 PyTorch 参考实现逐元素对比；
- 精度评测要严格按基准的官方协议，否则数字不可比；
- 注意公平：baseline 应使用引擎默认的最优 attention backend。

## 10. 参考资料

- MInference：arXiv 2407.02490
- FlexPrefill（ICLR 2025）：arXiv 2502.20766
- XAttention：arXiv 2503.16428
- SharePrefill：arXiv 2505.19578
- The Sparse Frontier（ACL 2026 Findings）
- Quest：arXiv 2406.10774；DuoAttention：arXiv 2410.10819
- Native Sparse Attention：arXiv 2502.11089
- DeepSeek-V3.2 在 vLLM 中的稀疏注意力实现：https://vllm.ai/blog/2025-09-29-deepseek-v3-2
- FlashAttention-2：arXiv 2307.08691；Triton 教程中的 Fused Attention
