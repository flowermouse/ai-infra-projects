# Transformer → 混合线性注意力：蒸馏与部署

> 📦 **候选池项目**：初稿，尚未精修，暂不开放选题。正式开放前会重新调研和修改。

> **难度** ★★★★★ ｜ **算法 : 系统** 7 : 3 ｜ **建议人数** 2 ｜ **最低硬件** 2–8×80GB（蒸馏）+ 1×80GB（部署） ｜ **基座** 训练：HF / flash-linear-attention；部署：vLLM

## 1. 背景

全注意力的 KV Cache 随上下文线性增长，计算随上下文平方增长，这在长上下文和长推理链场景下代价很高。**混合架构**（少量全注意力层 + 大量线性注意力 / 状态空间层）已经成为主流方向：

- **线性注意力 / SSM 层**：Mamba-2、Gated DeltaNet（GDN）等；
- **生产级混合模型**：Qwen3-Next（GDN + 门控注意力）、Jamba、Nemotron-H 等，vLLM / SGLang 已经支持这类混合架构的高效推理；
- **从已有 Transformer 转换**：*The Mamba in the Llama*（复用注意力投影权重，把 Transformer 蒸馏成混合模型）、LoLCATs（低秩线性化）、Jet-Nemotron（PostNAS：在预训练模型上搜索并替换注意力模块，报告 64K 上下文下生成吞吐提升数十倍）。

这条路线算法空间很大：哪些层保留全注意力？用哪种线性注意力？怎么蒸馏？但**转换后的模型必须能被推理引擎高效执行**，否则理论上的 O(n) 优势无法兑现：线性注意力层需要高效的 chunked / recurrent kernel，混合模型需要引擎同时管理 KV Cache 和循环状态，prefix caching 等特性也需要特殊处理。

## 2. 论文 → 引擎的鸿沟

| 问题 | 说明 |
|---|---|
| 模型结构必须被引擎支持 | 最好选择引擎已支持的层类型（如 GDN、Mamba-2），否则需要自己实现层和 kernel |
| 状态管理 | 混合模型同时有 KV Cache（全注意力层）和固定大小的循环状态（线性层），引擎需要分别管理；prefix caching 对循环状态的支持更复杂 |
| 短上下文下可能更慢 | 线性注意力的 kernel 在短序列上效率不一定比 FlashAttention 高，优势只在长上下文显现 |
| 精度 | 蒸馏后的模型在检索类、长上下文召回任务上最容易退化 |

## 3. 项目目标

1. 以 Qwen3-1.7B / 4B（或同级别）为起点，把一部分注意力层替换为引擎已支持的线性注意力层（推荐 GDN 或 Mamba-2），通过蒸馏得到混合模型；
2. 研究层选择策略（保留哪些层为全注意力）对精度的影响；
3. 让混合模型在 vLLM 中运行（尽量复用已有的混合架构支持，例如按 Qwen3-Next 或其他已支持混合模型的结构组织权重），完成正确性验证；
4. 与原模型在多种上下文长度、batch size 下对比质量与真实吞吐 / 显存。

## 4. 任务分解

### 4.1 阶段一：蒸馏（必做）
- 选择线性层类型与初始化方式（例如复用注意力的 Q/K/V 投影）；
- 实现分阶段蒸馏：逐层对齐隐藏状态 → 端到端 logits 蒸馏 → （可选）SFT；
- 使用 flash-linear-attention 等库的高效训练 kernel；
- 评测：MMLU、GSM8K、常识推理，以及长上下文任务（RULER、needle-in-a-haystack）。

### 4.2 阶段二：层选择研究（必做）
- 比较至少两种层选择策略：均匀间隔保留、基于重要性（例如移除后精度下降最小的层）保留、基于检索能力的保留；
- 比较不同的全注意力层比例（如 1/4、1/8）。

### 4.3 阶段三：引擎部署（必做）
- 将权重组织成引擎可加载的混合模型（复用已有模型实现，或通过 out-of-tree 注册新模型）；
- 正确性：引擎输出与 HF 实现一致；
- 性能：上下文长度（2K → 128K）× batch size 扫描，测量吞吐、TTFT、TPOT、显存占用，与原模型对比。

### 4.4 进阶（加分）
- 与 KV 压缩（正式项目 P03）/ 稀疏注意力（候选池）结合：混合模型 + 剩余全注意力层的 KV 压缩 / 稀疏化；
- 研究混合模型对 prefix caching 的支持（循环状态的缓存策略）；
- 扩展到更大模型或推理（thinking）模型。

## 5. 关键知识点（汇报必须讲清楚）

**算法**
- 线性注意力的基本形式与循环形式的等价性；门控与 delta rule 的作用；
- Mamba-2 与线性注意力的联系（SSD 视角）；
- 蒸馏方法：隐藏状态对齐、logits 蒸馏、权重复用初始化；
- 为什么线性注意力在"精确检索"上弱于全注意力（固定大小状态的容量限制）。

**系统**
- 线性注意力的 chunk-wise 并行训练 / prefill 算法，以及 recurrent decode；
- 混合模型在引擎中的状态管理（KV Cache + 循环状态）；
- 吞吐优势从多长的上下文开始出现，以及原因（roofline + 显存）。

## 6. 实验设计

- **模型**：Qwen3-1.7B / 4B 起点；
- **任务**：通用（MMLU、ARC、HellaSwag）、数学（GSM8K）、长上下文（RULER 多长度、needle）；
- **必须产出**：
  1. 精度对比：原模型 vs 不同层选择 / 比例的混合模型；
  2. 吞吐、显存 vs 上下文长度（多 batch size）；
  3. 最大可支持 batch × 上下文长度；
  4. 蒸馏训练曲线与成本。

## 7. 答辩追问示例

1. 写出线性注意力的并行形式和循环形式，并说明它们为什么等价。
2. 为什么复用注意力的 Q/K/V 投影权重来初始化线性层是合理的？
3. 你的混合模型在 RULER 上哪类子任务退化最严重？为什么？
4. 在 4K 上下文、batch=8 时，混合模型比原模型快吗？128K 时呢？为什么？
5. 混合模型的 prefix caching 和纯 Transformer 有什么不同？

## 8. Agent 使用建议

- 蒸馏训练脚本、评测流水线可以交给 Agent；层的实现尽量复用 flash-linear-attention 等成熟库，不要让 Agent 从头手写 kernel；
- **风险**：权重命名、形状与引擎中模型实现的对应关系容易出错，需要逐层对比 HF 与引擎的输出；
- 两人分工建议：一人负责蒸馏与评测，一人负责部署与性能分析。

## 9. 参考资料

- The Mamba in the Llama: Distilling and Accelerating Hybrid Models：arXiv 2408.15237
- LoLCATs：arXiv 2410.10254
- Jet-Nemotron（NVIDIA，PostNAS）
- Gated Delta Networks：arXiv 2412.06464
- Mamba-2 / Transformers are SSMs：arXiv 2405.21060
- A Systematic Analysis of Hybrid Linear Attention：arXiv 2507.06457
- flash-linear-attention：https://github.com/fla-org/flash-linear-attention
- vLLM 中混合模型（Qwen3-Next、Mamba 类）的支持文档
