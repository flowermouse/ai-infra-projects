# P07 · 多模态大模型视觉 Token 压缩的服务化

> **难度** ★★★★ ｜ **算法 : 系统** 6 : 4 ｜ **建议人数** 1–2 ｜ **最低硬件** 1×24GB ｜ **基座** vLLM 或 SGLang

## 1. 背景

多模态大模型（Qwen2.5-VL / Qwen3-VL、InternVL 等）中，一张高分辨率图片会被编码成数百到数千个视觉 token，视频更多。视觉 token 往往占输入的绝大部分，直接决定 prefill 时间和 KV Cache 大小。

视觉 token 压缩 / 剪枝是近两年非常活跃的算法方向：
- **LLM 内剪枝**：FastV（根据浅层注意力剪掉不重要的视觉 token）、SparseVLM（文本引导的剪枝）；
- **编码器侧压缩**：VisionZip（在视觉编码器输出端选取主导 token 并合并）、LLaVA-PruMerge 等；
- **2025–2026 年的新工作**：TopV（CVPR 2025，强调与 FlashAttention 和 KV Cache 兼容）、LearnPruner（ICLR 2026）、ERASE、ProViP 等。

但正如 TopV 指出的：**依赖注意力分数的剪枝与 FlashAttention 冲突；每步动态剪枝需要变长 KV，节省不了显存。** 绝大多数方法只在 HF 上、batch=1 下报告加速。在 vLLM / SGLang 这样的服务系统中，还要面对更多约束：

## 2. 论文 → 引擎的鸿沟

| 问题 | 说明 |
|---|---|
| Token 数必须在调度前确定 | 引擎的 multimodal processor 在调度前就确定了 placeholder token 数和 KV 分配；LLM 内部（第 K 层）的剪枝会让后续层的 token 数与已分配的不一致 |
| 注意力分数不可得 | FlashAttention 不输出注意力权重 |
| Prefix caching / 多模态缓存 | 引擎按图像哈希缓存视觉编码结果与 KV；若剪枝依赖文本 query，同一张图在不同问题下的 KV 就不同，缓存失效 |
| Chunked prefill | 视觉 token 可能被切到不同 chunk，影响依赖全局信息的剪枝 |
| 视觉编码器的开销 | 编码器侧本身可能是瓶颈，LLM 侧的剪枝无法减少这部分时间 |

## 3. 项目目标

1. 在 HF 上复现一种视觉 token 压缩方法，对齐论文精度；
2. 在 vLLM / SGLang 中实现该方法，正确处理 token 数确定、KV 分配、多模态缓存等系统问题；
3. 在**在线服务场景**（多请求并发、图文混合负载）中测量 TTFT、吞吐、显存的真实改善；
4. 鸿沟分析，并在系统约束下改进算法（例如"与 query 无关、可缓存"的压缩）。

## 4. 推荐路线

根据剪枝发生的位置，系统落地难度差别很大。开题时先想清楚走哪条路线：

| 路线 | 例子 | 系统难度 | 说明 |
|---|---|---|---|
| A. 编码器输出端压缩（与 query 无关） | VisionZip 类 | 较低 | token 数在进入 LLM 前确定，兼容多模态缓存，是推荐的起点 |
| B. 编码器输出端压缩（文本引导） | 部分 SparseVLM 变体 | 中 | 需要在进入 LLM 前获得文本信息，与缓存冲突 |
| C. LLM 第 K 层剪枝 | FastV、TopV | 高 | 需要支持"层间 token 数变化"，涉及 KV 分配与 attention metadata 的修改 |

## 5. 任务分解

### 5.1 阶段一：算法复现（必做）
- 模型：Qwen2.5-VL-7B 或 Qwen3-VL（同级别）；
- 任务：TextVQA、ChartQA、DocVQA、MMBench、视频理解（如 Video-MME 子集）；
- HF 上复现精度随保留 token 比例的变化，并测出"论文式"加速。

### 5.2 阶段二：引擎落地（必做）
- 读懂引擎的多模态流程：图像预处理 → placeholder token 数计算 → 视觉编码 → 嵌入合并 → LLM 前向；多模态缓存机制；
- 路线 A / B：修改 multimodal processor 与模型的 embedding 合并逻辑，使 placeholder 数等于压缩后的 token 数；
- 路线 C：设计"层间 token 数变化"的支持方案（例如在第 K 层之后对 KV 做 compaction，或者为前 K 层与后续层分别分配 KV）；
- 正确性：保留比例为 100% 时与原引擎一致；精度与 HF 参考实现对齐。

### 5.3 阶段三：鸿沟分析与改进（必做分析，改进为加分项）
- 在线负载（并发图文请求、不同分辨率）下的 TTFT、吞吐、最大并发；
- 时间分解：视觉编码 / 压缩 / LLM prefill / decode；
- 系统感知改进：例如设计与 query 无关、可缓存的压缩方法，并研究其精度代价；或对视频做跨帧的 token 合并。

## 6. 关键知识点（汇报必须讲清楚）

**算法**
- 视觉 token 冗余的来源（空间冗余、背景区域、视频的时间冗余）；
- 所选方法的打分 / 合并机制与假设；
- 为什么文本引导的剪枝通常精度更高、但系统代价更大。

**系统**
- 引擎中多模态请求的完整处理流程，特别是 placeholder token 数为何必须提前确定；
- 视觉 token 数对 prefill FLOPs、KV 显存、TTFT 的定量影响；
- 多模态缓存（编码器输出缓存、prefix caching）的工作方式；
- Amdahl 定律：视觉编码器占比对端到端收益的限制。

## 7. 实验设计

- **模型**：Qwen2.5-VL-7B / Qwen3-VL；
- **任务**：图像 VQA（TextVQA、ChartQA、DocVQA）、通用（MMBench）、视频（Video-MME 子集）；
- **必须产出**：
  1. 精度 vs 保留 token 比例（HF vs 引擎）；
  2. 在线服务：TTFT / 吞吐 vs 请求率（原引擎 vs 压缩）；
  3. 端到端时间分解；
  4. 多模态缓存命中下的收益变化（同一张图多个问题的场景）。

## 8. 答辩追问示例

1. 一张 1280×720 的图在你的模型里会变成多少个视觉 token？这对 prefill 时间和 KV 显存意味着什么？
2. 为什么引擎需要在调度前就知道 placeholder token 数？你是怎么处理的？
3. FastV 在第 2 层剪枝，需要第 2 层的注意力分数。在 FlashAttention 下怎么获得？代价多大？
4. 你的方法对 OCR / 文档类任务的影响为什么比对通用 VQA 大？
5. 同一张图被问了 10 个不同问题，你的方法能复用缓存吗？

## 9. Agent 使用建议

- 多模态流程在引擎中跨越多个模块（processor、encoder、模型、调度器），先让 Agent 帮你梳理调用链，再自己用调试器跟踪一个图文请求；
- **风险**：placeholder token 数与实际嵌入数不一致会导致难以排查的错误（错位或越界），必须写针对性的测试；
- 多模态评测的预处理（分辨率、`max_pixels` 等参数）要与论文一致。

## 10. 参考资料

- FastV：arXiv 2403.06764
- SparseVLM：arXiv 2410.04417
- VisionZip：arXiv 2412.04467
- TopV（CVPR 2025）：arXiv 2503.18278
- LearnPruner（ICLR 2026）：arXiv 2604.23950
- vLLM 多模态文档（Multimodal Inputs、新增多模态模型指南）；SGLang 多模态文档
- Qwen2.5-VL 技术报告：arXiv 2502.13923
