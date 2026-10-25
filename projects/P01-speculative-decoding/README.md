# P01 · 投机解码：从论文到真实加速

> **难度** ★★★★ ｜ **算法 : 系统** 5 : 5 ｜ **建议人数** 1–2 ｜ **最低硬件** 1×80GB（或 2×24GB） ｜ **基座** vLLM + Speculators，或 SGLang + SpecForge

## 1. 背景

投机解码（Speculative Decoding）用一个便宜的草稿器一次猜多个 token，再由目标模型并行验证。在拒绝采样的保证下，输出分布与目标模型**完全一致**。它是目前最重要、论文最多的推理加速方向之一：

- **草稿器设计**：独立小模型 → Medusa 多头 → EAGLE 系列（特征层草稿，EAGLE-3 融合多层特征）→ MTP（模型自带）→ **DFlash**（ICML 2026，用块扩散模型一次前向并行生成整块草稿，报告比 EAGLE-3 快约 2.5×）→ P-EAGLE（并行多 token 预测）……
- **草稿结构**：链式 → 静态树 → 动态树（EAGLE-2）；
- **免训练方法**：n-gram / Prompt Lookup、Lookahead Decoding、SuffixDecoding、自投机（跳层）等。

vLLM 与 SGLang 都有完整的投机解码框架，支持 EAGLE / EAGLE-3 / MTP / n-gram 等方法，DFlash 也已接入两个框架。**但论文中的加速与真实服务中的加速之间存在巨大鸿沟**：

- *Speculative Decoding: Performance or Illusion?*（MLSys 2026）首次在 vLLM 上系统测量了多种投机解码方法，发现**目标模型的验证占主导开销**，真实加速远低于理论上界；接受长度在不同 token 位置、请求、数据集之间差异很大；
- 论文几乎都只报告 batch=1 的结果，但投机解码的收益前提是 **GPU 有空闲算力**。batch 增大后系统转向 compute-bound，验证被拒绝的 token 是纯浪费。Meta 的生产测量显示，batch 从 2 增加到 48 时，vLLM 中 EAGLE 的加速从 1.3× 降到 0.7×，**变成了减速**；
- MagicDec 指出，在长上下文、大 batch 下瓶颈转向 KV Cache 读取，换用稀疏 KV 的草稿器反而能在大 batch 下拿到加速。

本项目要求你**完整走一遍"新方法落地 → 真实性能刻画 → 系统感知的算法改进"**，最终回答：**这个投机解码方法在真实服务中什么时候有用、有多大用、怎样才能更有用？**

## 2. 论文 → 引擎的鸿沟

| 问题 | 说明 |
|---|---|
| **与 continuous batching 共存** | 研究原型往往只支持 batch=1；引擎中不同请求的接受长度不同，需要处理不规则的 KV 增长与回滚 |
| **树形验证的 attention mask** | 树结构草稿需要定制的 tree attention mask，并与 paged attention kernel 兼容 |
| **CUDA Graph** | 动态树、动态投机长度会破坏静态 shape |
| **草稿器自身的开销** | 草稿器的 KV Cache、调度、额外前向都需要纳入引擎的内存与调度管理 |
| **验证代价随 batch 增长** | 论文的加速比公式假设验证代价恒定，真实引擎中它随 batch、序列长度、验证 token 数非线性变化 |
| **接受率非平稳** | 随领域、位置、请求变化，静态配置难以兼顾 |
| **训练-推理不一致** | 草稿器训练时用到的目标模型特征（如隐藏状态）与引擎推理时的数值可能不一致，导致接受率下降 |

## 3. 项目目标

1. **新方法落地**：选择一个尚未被（或未被很好地）集成到 vLLM / SGLang 的投机解码方法，在 HF 上复现，再在引擎中实现并保证正确；
2. **真实性能刻画**：在多 batch size、多领域、多上下文长度下，与无投机解码、引擎中已有的最强方法（如 EAGLE-3）公平对比，建立**验证代价模型**；
3. **系统感知的算法改进**：基于刻画结果，设计**批感知（batch-aware）的自适应投机策略**，根据负载与接受率动态决定投机长度（必要时关闭投机），在全部负载区间内做到不劣于"不投机"和"最佳静态配置"。

> 两人组队时的分工建议：一人主攻新方法的训练与引擎实现（阶段一、二），一人主攻性能刻画、代价模型与自适应策略（阶段二、三）。

## 4. 候选算法池

| 方法 | 类型 | 落地难点 / 看点 |
|---|---|---|
| HASS / GRIFFIN 等 EAGLE 改进 | 草稿器训练方法 | 主要改训练，推理侧可复用 EAGLE 路径；重点在训练与真实接受率 |
| PARD（并行草稿） | 草稿器结构 | 一次前向出多个草稿 token；需要改 proposer |
| DFlash（块扩散草稿器） | 草稿器结构 | 已有上游支持；可以**为新目标模型训练**草稿器并分析接受率，或研究它在大 batch 下的表现 |
| Lookahead Decoding | 免训练 | Jacobi 迭代 + n-gram 池；与 continuous batching 的结合较难 |
| SuffixDecoding / 检索式草稿 | 免训练 | 对 Agent、代码编辑等重复性负载很有效；需要维护全局后缀树 |
| 自投机（LayerSkip / SWIFT 等跳层） | 免训练或轻训练 | 草稿与目标共享权重和 KV，KV 管理是难点 |
| 其他 2025–2026 新论文 | — | 自选，需在开题时论证 |

## 5. 任务分解

### 5.1 阶段一：算法复现（必做）
- 在 HF / 研究原型上复现所选方法，测量各领域（MT-Bench、HumanEval、GSM8K 等）的**平均接受长度**与 batch=1 加速比；
- 如需训练草稿器，使用 Speculators（vLLM）或 SpecForge（SGLang）的训练流程，目标模型推荐 Llama-3.1-8B-Instruct 或 Qwen3-8B；
- 同时复现 EAGLE-3 baseline（可用公开草稿器权重，如 RedHatAI 或 SpecBundle 发布的模型）。

### 5.2 阶段二：引擎落地与性能刻画（必做）
- **读懂框架**：锁定版本中投机解码的 proposer、验证（rejection sampler）、KV 回滚、与调度器的交互；
- **实现**：新方法的 proposer（以及需要时的验证逻辑、树形 mask），支持多请求并发；
- **正确性**：greedy 下输出与不开投机解码逐 token 一致；采样下做分布一致性检验；
- **刻画**：
  - 扫描 并发度 × 投机长度 × 领域 × 上下文长度，对比 无投机 / EAGLE-3 / 新方法；
  - 测量单步验证耗时与 (batch size, 每请求验证 token 数, KV 长度) 的关系，拟合**验证代价模型**；
  - 测量接受率的可预测性：草稿器置信度、历史接受率、token 位置对接受与否的预测能力；
  - 时间分解：草稿开销、验证开销、调度 / CPU 开销。

### 5.3 阶段三：批感知自适应投机（必做基础版，进阶为加分项）
- **基础版（必做）**：基于代价模型与接受率估计，推导"期望吞吐最大化"的投机长度选择方法，实现**全局策略**（每步根据负载决定统一的投机长度，必要时关闭投机）；
- 处理 CUDA Graph 兼容性：例如把投机长度离散化为少数几档，每档 capture 一个 graph；
- 在变化的请求率（如 Azure trace 的到达模式）下评估：无投机 / 各静态配置 / 你的策略；
- **进阶（加分）**：
  - 请求级策略：不同请求不同投机长度，或基于草稿置信度提前终止草稿；
  - 把问题形式化为在线学习 / bandit 问题，与启发式策略对比；
  - 大 batch 下的草稿树动态剪枝；长上下文下结合 MagicDec 的稀疏 KV 草稿思路。

## 6. 关键知识点（汇报必须讲清楚）

**算法**
- 投机解码的拒绝采样，以及它为什么能保证输出分布不变（要能推导）；
- 期望加速比与接受率 α、投机长度 γ、草稿 / 目标代价比 c 的关系，以及这个公式在 batch > 1 时为什么不成立；
- 所选方法的核心思想与它相对 EAGLE-3 的改进点；
- 你的自适应策略的形式化（目标函数、约束、估计量）。

**系统**
- 引擎中投机解码的完整执行流程（propose → verify → accept / reject → KV 回滚）；
- Tree attention 的 mask 构造及其在 paged attention 中的实现；
- 不同 batch size 下 decode 的 roofline 分析；验证 k 个 token 的代价随 batch 的变化；
- 投机解码与 CUDA Graph、chunked prefill、prefix caching 的交互；为什么每请求不同的投机长度会降低 kernel 效率。

## 7. 实验设计

- **模型**：Llama-3.1-8B-Instruct 或 Qwen3-8B + 对应草稿器；
- **数据**：MT-Bench、HumanEval、GSM8K、SpecBench；在线负载用 ShareGPT + 泊松 / Azure trace 到达；
- **必须产出的图表**：
  1. 各领域的接受长度对比（新方法 vs EAGLE-3）；
  2. 加速比 vs 并发度曲线（新方法 / EAGLE-3 / 无投机），标出加速消失的临界点；
  3. 加速比热力图（投机长度 × 并发度）；
  4. 验证代价模型的拟合效果（预测 vs 实测）；
  5. 每步时间分解（草稿 / 验证 / 其他）；
  6. 自适应策略 vs 静态配置在全负载区间的吞吐曲线，以及动态负载下的时间序列（请求率、所选投机长度、吞吐）；
  7. 鸿沟瀑布图：论文报告 → HF 复现 → 引擎 batch=1 → 引擎大 batch。

## 8. 答辩追问示例

1. 推导：使用拒绝采样的投机解码，输出分布与目标模型一致。
2. 你的方法在 batch=1 和 batch=64 下的加速比分别是多少？从 roofline 的角度解释为什么差这么多。
3. 你的 proposer 在 CUDA Graph 下是怎么运行的？动态部分怎么处理？
4. 树形草稿的验证中，tree attention mask 长什么样？和 paged attention 如何配合？
5. 论文报告的加速比你复现到了多少？差距来自哪里？
6. 你的代价模型中，验证代价与哪些因素有关？为什么是这个函数形式？
7. 你的自适应策略在接受率估计不准时会怎样？响应速度和稳定性如何权衡？
8. 换到 H100 和 4090（算力 / 带宽比不同），临界并发度会怎么移动？你的策略需要重新标定吗？

## 9. Agent 使用建议

- Agent 擅长帮你梳理投机解码框架的调用链、写 proposer 骨架和扫描实验脚本；
- **KV 回滚、接受长度的统计口径**（是否包含 bonus token）、树 mask 的构造要逐行核对；
- 草稿器训练可以让 Agent 帮你配置和调参，但训练数据的选择（例如是否使用目标模型自己重新生成的回复）要你来决定并说明理由；
- 自适应策略的推导和形式化要由你来完成，能在白板上讲清楚；
- 一定要先写好"输出与非投机一致"的正确性测试，再做性能优化。修改投机长度逻辑时尤其容易破坏 KV 回滚和 CUDA Graph 的正确性。

## 10. 参考资料

- Fast Inference from Transformers via Speculative Decoding：arXiv 2211.17192
- EAGLE / EAGLE-2 / EAGLE-3：arXiv 2401.15077 / 2406.16858 / 2503.01840
- Medusa：arXiv 2401.10774
- DFlash: Block Diffusion for Flash Speculative Decoding（ICML 2026）：arXiv 2602.06036
- HASS：arXiv 2408.15766；GRIFFIN：arXiv 2502.11018；PARD：arXiv 2504.18583
- Lookahead Decoding：arXiv 2402.02057；SuffixDecoding：arXiv 2411.04975；LayerSkip：arXiv 2404.16710
- Speculative Decoding: Performance or Illusion?（MLSys 2026）：arXiv 2601.11580
- MagicDec：arXiv 2408.11049
- SmartSpec（基于 goodput 的投机解码）：arXiv 2406.14066
- *Batch Speculative Decoding Done Right*（2025，批量投机解码的正确性与效率问题）
- vLLM Speculators：https://github.com/vllm-project/speculators ；文档 https://docs.vllm.ai/projects/speculators/
- SpecForge：https://github.com/sgl-project/SpecForge
