# P01 · 投机解码新方法的引擎落地

> **难度** ★★★★ ｜ **算法 : 系统** 5 : 5 ｜ **建议人数** 1–2 ｜ **最低硬件** 1×80GB（或 2×24GB） ｜ **基座** vLLM + Speculators，或 SGLang + SpecForge

## 1. 背景

投机解码（Speculative Decoding）用一个便宜的草稿器一次猜多个 token，再由目标模型并行验证。在拒绝采样的保证下，输出分布与目标模型**完全一致**。它是目前最重要、也是论文最多的推理加速方向之一：

- **草稿器设计**：独立小模型 → Medusa 多头 → EAGLE 系列（特征层草稿，EAGLE-3 融合多层特征）→ MTP（模型自带）→ **DFlash**（ICML 2026，用块扩散模型一次前向并行生成整块草稿，报告比 EAGLE-3 快约 2.5×）→ P-EAGLE（并行多 token 预测）……
- **草稿结构**：链式 → 静态树 → 动态树（EAGLE-2）；
- **免训练方法**：n-gram / Prompt Lookup、Lookahead Decoding、SuffixDecoding、自投机（跳层）等。

vLLM 与 SGLang 都有完整的投机解码框架，并支持 EAGLE / EAGLE-3 / MTP / n-gram 等方法，DFlash 也已接入两个框架。**但大量新论文的方法仍停留在研究原型阶段**，研究原型上报告的加速比和真实引擎中的表现往往差距很大。

## 2. 论文 → 引擎的鸿沟

把一个新的投机解码方法放进真实引擎，需要面对：

| 问题 | 说明 |
|---|---|
| **与 continuous batching 共存** | 研究原型往往只支持 batch=1；引擎中不同请求的接受长度不同，需要处理不规则的 KV 增长与回滚 |
| **树形验证的 attention mask** | 树结构草稿需要定制的 tree attention mask，要与 paged attention kernel 兼容 |
| **CUDA Graph** | 动态树、动态长度会破坏静态 shape |
| **草稿器自身的开销** | 草稿器的 KV Cache、调度、额外前向都需要纳入引擎的内存与调度管理 |
| **验证代价随 batch 增长** | *Speculative Decoding: Performance or Illusion?*（MLSys 2026）在 vLLM 上发现，**目标模型的验证占主导开销**，真实加速离理论上界很远 |
| **训练-推理不一致** | 草稿器训练时的特征（如目标模型隐藏状态）与引擎推理时的数值可能不一致，接受率随之下降 |

## 3. 项目目标

选择一个**尚未被（或未被很好地）集成**到 vLLM / SGLang 的投机解码方法，完成：
1. HF 研究原型上复现论文的接受长度和加速比；
2. 在引擎投机解码框架中实现该方法，支持 continuous batching，保证输出正确；
3. 与引擎中已有的最强方法（如 EAGLE-3）在**多 batch size** 下做公平对比；
4. 完成鸿沟分析，并尝试至少一项"系统感知"的改进。

## 4. 候选算法池

| 方法 | 类型 | 落地难点 / 看点 |
|---|---|---|
| HASS / GRIFFIN 等 EAGLE 改进 | 草稿器训练方法 | 主要改训练，推理侧可复用 EAGLE 路径；重点在训练与真实接受率 |
| PARD（并行草稿） | 草稿器结构 | 一次前向出多个草稿 token；需要改 proposer |
| DFlash（块扩散草稿器） | 草稿器结构 | 已有上游支持，可做的方向是**为新目标模型训练**并分析接受率，或改进其在大 batch 下的表现 |
| Lookahead Decoding | 免训练 | Jacobi 迭代 + n-gram 池；与 continuous batching 的结合较难 |
| SuffixDecoding / 检索式草稿 | 免训练 | 对 Agent、代码编辑等重复性负载很有效；需要维护全局后缀树 |
| 自投机（LayerSkip / SWIFT 等跳层） | 免训练或轻训练 | 草稿与目标共享权重和 KV，KV 管理是难点 |
| 其他 2025–2026 新论文 | — | 自选，需在开题时论证 |

## 5. 任务分解

### 5.1 阶段一：算法复现（必做）
- 在 HF / 研究原型上复现所选方法，测量各领域（MT-Bench、HumanEval、GSM8K 等）的**平均接受长度**与 batch=1 加速比；
- 如需训练草稿器，使用 Speculators（vLLM）或 SpecForge（SGLang）的训练流程，目标模型推荐 Llama-3.1-8B-Instruct 或 Qwen3-8B。

### 5.2 阶段二：引擎落地（必做）
- 读懂锁定版本的投机解码框架：proposer、验证（rejection sampler）、KV 回滚、与调度器的交互；
- 实现新方法的 proposer（以及需要时的验证逻辑、树形 mask），支持多请求并发；
- 正确性：greedy 下输出与不开投机解码逐 token 一致；采样下做分布一致性检验；
- 性能：batch / 并发度扫描，与无投机解码、EAGLE-3（或引擎中已有最强方法）对比。

### 5.3 阶段三：鸿沟分析与改进（必做分析，改进为加分项）
- 鸿沟瀑布图：论文报告 → HF 复现 → 引擎 batch=1 → 引擎大 batch，每一步损失多少、原因是什么；
- 时间分解：草稿开销、验证开销、调度 / CPU 开销；
- 系统感知改进（任选）：减少草稿器开销（kernel 融合、CUDA Graph 化）、改进草稿结构使验证更"硬件友好"、按接受率动态调整等。注意与 P02 的分工，P02 专注大 batch 自适应策略。

## 6. 关键知识点（汇报必须讲清楚）

**算法**
- 投机解码的拒绝采样，以及为什么它能保证输出分布不变（要能推导）；
- 期望加速比与接受率 α、投机长度 γ、草稿 / 目标代价比 c 的关系；
- 所选方法的核心思想与它相对 EAGLE-3 的改进点。

**系统**
- 引擎中投机解码的完整执行流程（propose → verify → accept / reject → KV 回滚）；
- Tree attention 的 mask 构造及其在 paged attention 中的实现；
- 为什么验证代价随 batch 增大而上升（roofline 分析）；
- 投机解码与 CUDA Graph、chunked prefill、prefix caching 的交互。

## 7. 实验设计

- **模型**：Llama-3.1-8B-Instruct 或 Qwen3-8B（草稿器对应）；
- **数据**：MT-Bench、HumanEval、GSM8K、SpecBench；在线负载用 ShareGPT；
- **必须产出**：
  1. 各领域的接受长度对比（新方法 vs EAGLE-3）；
  2. 加速比 vs 并发度曲线（新方法 / EAGLE-3 / 无投机），标出临界点；
  3. 鸿沟瀑布图；
  4. 每步时间分解（草稿 / 验证 / 其他）。

## 8. 答辩追问示例

1. 推导：使用拒绝采样的投机解码，输出分布与目标模型一致。
2. 你的方法在 batch=1 和 batch=64 下的加速比分别是多少？为什么差这么多？
3. 你的 proposer 在 CUDA Graph 下是怎么运行的？动态部分怎么处理？
4. 树形草稿的验证中，tree attention mask 长什么样？和 paged attention 如何配合？
5. 论文报告的加速比你复现到了多少？差距来自哪里？
6. 如果换到 H100 和 4090，加速比的临界点会怎么移动？

## 9. Agent 使用建议

- Agent 擅长帮你梳理投机解码框架的调用链、写 proposer 的骨架，但 **KV 回滚、接受长度的统计口径**（是否包含 bonus token）、树 mask 的构造要逐行核对；
- 草稿器训练流程可以让 Agent 帮你配置和调参，但训练数据的选择（是否使用目标模型自己重新生成的回复）要你来决定并说明理由；
- 一定要先写好"输出与非投机一致"的正确性测试，再做性能优化。

## 10. 参考资料

- Fast Inference from Transformers via Speculative Decoding：arXiv 2211.17192
- EAGLE / EAGLE-2 / EAGLE-3：arXiv 2401.15077 / 2406.16858 / 2503.01840
- Medusa：arXiv 2401.10774
- DFlash: Block Diffusion for Flash Speculative Decoding（ICML 2026）：arXiv 2602.06036
- HASS：arXiv 2408.15766；GRIFFIN：arXiv 2502.11018；PARD：arXiv 2504.18583
- Lookahead Decoding：arXiv 2402.02057；SuffixDecoding：arXiv 2411.04975
- LayerSkip：arXiv 2404.16710
- Speculative Decoding: Performance or Illusion?（MLSys 2026）：arXiv 2601.11580
- vLLM Speculators：https://github.com/vllm-project/speculators ；文档 https://docs.vllm.ai/projects/speculators/
- SpecForge：https://github.com/sgl-project/SpecForge
