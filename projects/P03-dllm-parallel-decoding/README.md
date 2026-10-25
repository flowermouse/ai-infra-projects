# P03 · 扩散语言模型：并行解码策略的引擎落地

> **难度** ★★★★ ｜ **算法 : 系统** 5 : 5 ｜ **建议人数** 1–2 ｜ **最低硬件** 1×80GB ｜ **基座** SGLang（dLLM 算法接口）或 vLLM（Model Runner V2 / DiffusionGemma 路径）

## 1. 背景

扩散语言模型（Diffusion LLM, dLLM）从"逐 token 自回归"转向"在一个块内迭代去噪、并行确定多个 token"，被认为有望突破自回归解码的串行瓶颈。2025–2026 年这个方向发展很快：

- **模型**：LLaDA、Dream（全序列双向注意力）→ **块扩散**（Block Diffusion / BD3-LM：块间自回归、块内扩散，天然兼容 KV Cache）→ SDAR、Fast-dLLM v2（把 AR 模型转换成块扩散模型）→ LLaDA2.0（扩展到 100B MoE）→ DiffusionGemma（2026 年 6 月，首个被 vLLM 原生支持的 dLLM）；
- **推理算法**：Fast-dLLM（ICLR 2026：近似 KV Cache + 基于置信度的并行解码）、dKV-Cache、dLLM-Cache、d2Cache、FreeCache 等缓存方法，以及 DepCap、BlockBatch 等自适应并行解码策略；
- **系统**：dInfer（inclusionAI 的 dLLM 推理框架）；SGLang 基于 chunked prefill 机制加入了块扩散支持，提供可插拔的 dLLM 解码算法接口（`--dllm-algorithm`）；vLLM 用 Model Runner V2 的 `ModelState` 钩子支持 DiffusionGemma，复用了投机解码路径（去噪步相当于"整块拒绝"、提交步相当于"整块接受"）。

**问题在于**：dLLM 论文大多在 batch=1、HF 实现上报告加速；SGLang 博客也明确指出，dLLM 的批处理仍在完善中，很多 AR 模型已有的系统优化 dLLM 还没有。**dLLM 的并行解码优势在大 batch 下是否还存在？** 这正是需要系统和算法一起回答的问题。

## 2. 论文 → 引擎的鸿沟

| 问题 | 说明 |
|---|---|
| 并行度与 batch 的冲突 | dLLM 用"块内并行"利用空闲算力；batch 大时 GPU 已经饱和，并行解码的优势缩小 |
| 不定的去噪步数 | 基于置信度阈值的解码，每个请求每个块的步数不同，影响批调度和 CUDA Graph |
| 双向注意力与 KV Cache | 全序列双向的 dLLM 无法精确复用 KV，只能做近似缓存；块扩散的块间 KV 精确可复用，块内仍需重复计算 |
| 注意力 kernel | 需要 per-request 的 causal / bidirectional 混合 mask（vLLM 为此把 `causal` 标志改为 per-request 张量） |
| 质量与速度的权衡 | 并行解码阈值越激进越快，但质量下降；这种权衡在不同任务上差异很大 |

## 3. 项目目标

在 SGLang（推荐，接口更现成）或 vLLM 中，为块扩散模型实现**一种论文中的高效并行解码 / 缓存策略**，完成：
1. HF / 官方原型上复现论文结果（质量 + batch=1 加速）；
2. 在引擎中实现该策略，支持多请求批处理；
3. 在**多 batch size** 下与同规模 AR 模型（开启投机解码与不开两种）公平对比；
4. 鸿沟分析 + 系统感知的策略改进。

## 4. 候选算法池

| 方法 | 类型 | 说明 |
|---|---|---|
| Fast-dLLM v2 的分层并行解码 | 解码策略 | 块级 + 子块级置信度并行；SGLang roadmap 中列为待支持 |
| 基于置信度 / 熵的自适应阈值 | 解码策略 | 例如 DiffusionGemma 的熵预算接受、DepCap 的自适应块级并行 |
| 跨块流水（下一块提前开始去噪） | 解码策略 | 提高并行度；对调度器改动较大 |
| dLLM 的 KV 近似复用（dKV-Cache / d2Cache / Fast-dLLM DualCache） | 缓存 | 块内 KV 复用；与精确块间 KV 结合 |
| 批感知的阈值调度 | 系统感知策略 | batch 大时降低并行度、batch 小时提高并行度 |
| 其他 2026 新方法 | — | 自选，开题论证 |

## 5. 任务分解

### 5.1 阶段一：算法复现（必做）
- 选择模型：SDAR（1.7B / 4B / 8B）、Fast-dLLM v2-7B、LLaDA2.0-mini 等块扩散模型（以锁定版本引擎的支持情况为准）；
- 在官方实现上复现所选方法：GSM8K、MATH、HumanEval、MBPP 等任务的精度，以及 TPF（每次前向解码的 token 数）和 batch=1 吞吐。

### 5.2 阶段二：引擎落地（必做）
- 精读引擎的 dLLM 路径：
  - **SGLang**：dLLM 算法接口如何驱动 model runner 反复执行直到一个块解码完成；块扩散如何复用 chunked prefill；
  - **vLLM**：`ModelState` 钩子（`prepare_inputs` / `prepare_attn` / `custom_sampler` 等）；去噪步与提交步如何映射到投机解码的"接受 / 拒绝"；
- 以插件 / 新算法的方式实现所选策略；
- 正确性：与官方实现在相同随机种子下输出一致（或精度一致）；
- 性能：batch size 扫描（1 → 64+），记录 TPF、tokens/s、每步耗时。

### 5.3 阶段三：鸿沟分析与改进（必做分析，改进为加分项）
- 与同规模 AR 模型（如 SDAR 对应的 Qwen3 基座）在引擎中对比：AR / AR + 投机解码 / dLLM，三条吞吐-batch 曲线，找出 dLLM 优势消失的临界点；
- 分析：并行度、去噪步数分布、kernel 效率、调度开销分别贡献了多少差距；
- 系统感知改进：例如根据 batch 负载动态调整置信度阈值或块大小，在质量约束下最大化吞吐。

## 6. 关键知识点（汇报必须讲清楚）

**算法**
- 掩码扩散语言模型（Masked Diffusion LM）的训练目标与采样过程；
- 块扩散为什么能兼容 KV Cache；块大小的权衡；
- 置信度并行解码的原理，以及它为什么会损失质量（token 间的依赖被忽略）；Fast-dLLM 对此的理论分析。

**系统**
- dLLM 一次去噪前向的计算特征：块内 token 数 × batch 构成的"有效 batch"，以及它与 AR decode 的 roofline 对比；
- 为什么 dLLM 的优势主要在小 batch / 低延迟场景；
- 双向 / 因果混合 attention mask 在 kernel 中的实现；
- 不定去噪步数对批调度和 CUDA Graph 的影响。

## 7. 实验设计

- **模型**：块扩散模型 + 同规模 AR 对照模型；
- **任务**：GSM8K、MATH500、HumanEval、MBPP（覆盖数学与代码，二者的并行度特征不同）；
- **必须产出**：
  1. 质量-速度权衡曲线（阈值扫描）；
  2. 吞吐 vs batch size：dLLM（你的策略 / 默认策略）、AR、AR + 投机解码；
  3. TPF 与每步耗时随 batch 的变化；
  4. 鸿沟瀑布图：论文加速 → 引擎 batch=1 → 引擎大 batch。

## 8. 答辩追问示例

1. 掩码扩散语言模型的训练目标是什么？和 BERT 的 MLM 有什么关系和区别？
2. 块扩散中，块大小设为 4、32、256 各有什么影响（质量、并行度、KV 复用）？
3. 一次去噪前向在 batch=1、块大小=32 时，算术强度和 AR decode 相比如何？batch=64 时呢？
4. 置信度阈值从 0.9 降到 0.7，TPF 和精度怎么变？哪类任务退化最严重？为什么？
5. vLLM 为什么能用投机解码的路径支持 dLLM？"拒绝整个画布"在 KV 层面意味着什么？
6. 在什么场景下你会推荐部署 dLLM 而不是 AR + 投机解码？

## 9. Agent 使用建议

- dLLM 引擎路径较新、文档较少，可以让 Agent 帮你精读源码、画出执行流程，但务必自己用调试器跟踪一次完整的块解码；
- **风险**：dLLM 的随机性（掩码选择、采样）使"输出一致性"测试更难，需要固定所有随机源，或改用精度一致性测试；
- 公平对比是本项目的核心：AR baseline 必须开启引擎的全部优化（CUDA Graph、投机解码等）。

## 10. 参考资料

- LLaDA：arXiv 2502.09992；Dream 7B：arXiv 2508.15487
- Block Diffusion（BD3-LM）：arXiv 2503.09573
- SDAR：arXiv 2510.06303
- Fast-dLLM（ICLR 2026）：arXiv 2505.22618 ；https://nvlabs.github.io/Fast-dLLM/
- Fast-dLLM v2（ICLR 2026）：arXiv 2509.26328
- dKV-Cache：arXiv 2505.15781；dLLM-Cache：arXiv 2506.06295；FreeCache：arXiv 2505.21467
- dInfer：arXiv 2510.08666
- LLaDA2.0：arXiv 2512.15745
- SGLang dLLM 支持博客：https://lmsys.org/blog/2025-12-19-diffusion-llm
- vLLM DiffusionGemma 博客：https://vllm.ai/blog/2026-06-10-diffusion-gemma
- DepCap：arXiv 2604.15750；BlockBatch：arXiv 2605.29233
