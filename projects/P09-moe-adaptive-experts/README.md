# P09 · MoE 自适应专家计算的引擎落地

> **难度** ★★★★ ｜ **算法 : 系统** 6 : 4 ｜ **建议人数** 1–2 ｜ **最低硬件** 1×80GB（开发可用 OLMoE 等小 MoE 在 24GB 上进行） ｜ **基座** vLLM 或 SGLang + Triton

## 1. 背景

MoE（Mixture-of-Experts）已是前沿开源大模型的主流架构，DeepSeek-V3/V4、Qwen3-MoE、Kimi K2、GLM、MiniMax 等主力模型几乎都是 MoE。MoE 用稀疏激活换取更大的模型容量，但"每个 token 固定激活 top-k 个专家"并不一定最优，由此出现了大量**自适应专家计算**的算法：

- **动态 top-k / top-p 路由**：简单 token 少用专家，困难 token 多用专家（如 *Harder Tasks Need More Experts*）；
- **专家剪枝 / 跳过**：去掉低频或低贡献的专家，或在推理时跳过权重很小的专家；
- **专家合并 / 压缩**：把相似专家合并，减少总参数；
- **共享专家与路由专家的权衡**等。

这些方法论文中常报告"激活参数减少 X%"，但**在真实引擎中，MoE 层的耗时并不和激活专家数成正比**：
- decode 小 batch 时 MoE 层是 memory-bound，耗时取决于**被激活的不同专家数（需要读取的权重量）**，而不是每个 token 的 top-k；
- fused MoE kernel 按专家把 token 分组并对齐到 block 大小，动态 top-k 会改变分组的规整性；
- 大 batch 下几乎所有专家都会被某个 token 激活，"跳过专家"的收益随 batch 迅速消失。

## 2. 论文 → 引擎的鸿沟

| 问题 | 说明 |
|---|---|
| FLOPs 与延迟脱钩 | 小 batch 时决定延迟的是读取的专家权重量，而非 FLOPs |
| Batch 内的专家覆盖 | batch 增大后，被激活的专家集合趋近全集，动态路由的收益下降 |
| Kernel 规整性 | 每 token 不同 k 会使 `moe_align_block_size` 之后的 padding 与分组变化，影响 kernel 效率 |
| CUDA Graph | 数据相关的专家数需要以静态 shape 实现 |
| 专家并行（EP） | 专家剪枝 / 合并会改变各 GPU 的负载分布 |

## 3. 项目目标

1. 选择一种自适应专家计算方法，在 HF 上复现精度与"激活参数减少量"；
2. 在 vLLM / SGLang 的 fused MoE 路径中实现，保证正确性；
3. 在**多 batch size** 下测量真实加速，并解释它与"激活参数减少量"之间的差距；
4. 提出"批感知 / kernel 感知"的改进，例如以"batch 内被激活的专家总数"为优化目标的路由策略。

## 4. 候选算法池

| 方法 | 说明 |
|---|---|
| 动态 top-p 路由 | 每 token 按累积概率选择专家数；需处理 token 间 k 不同 |
| 推理时专家跳过 | 路由权重低于阈值的专家不计算（免训练） |
| 专家剪枝 | 基于校准集统计删除专家，模型结构改变，部署最简单 |
| 专家合并 | 合并相似专家，减少权重读取量 |
| **批感知路由（研究点）** | 在 batch 内鼓励 token 共享专家，减少被激活的专家总数，从而减少权重读取 |

## 5. 任务分解

### 5.1 阶段一：算法复现（必做）
- 模型：OLMoE-1B-7B（开发）、Qwen3-30B-A3B（主实验，80GB 卡上 BF16 或 FP8）；
- HF 上复现所选方法：精度（MMLU、GSM8K、HumanEval 等）与平均激活专家数。

### 5.2 阶段二：引擎落地（必做）
- 读懂引擎的 MoE 路径：router → top-k → token 按专家排序对齐 → grouped GEMM（gate_up → 激活 → down）→ 加权合并；
- 实现所选方法：修改 router 与 fused MoE 的调用（如支持每 token 不同 k 的 mask 实现），必要时修改 Triton kernel；
- 正确性：关闭自适应时与原实现输出一致；精度与 HF 参考对齐。

### 5.3 阶段三：鸿沟分析与改进（必做分析，改进为加分项）
- MoE 层耗时 vs batch size：原实现 / 你的方法；同时记录"每 token 平均激活专家数"和"batch 内被激活的不同专家数"；
- 建立 MoE 层耗时的预测模型（与哪个量最相关？）；
- 批感知 / kernel 感知改进，并在精度约束下验证真实加速。

## 6. 关键知识点（汇报必须讲清楚）

**算法**
- MoE 路由（top-k gating、负载均衡损失、共享专家）；
- 所选方法的原理，以及它为什么能（或不能）保持精度；
- 专家激活分布的特征：跨层、跨领域、跨 token 位置的差异。

**系统**
- Fused MoE kernel 的数据流与 `moe_align_block_size` 中 padding 的作用；
- 小 batch 与大 batch 下 MoE 层的瓶颈（权重读取 vs 计算）；
- batch 内专家覆盖率随 batch size 的变化（可以用概率模型估算）；
- 专家并行与 all-to-all 的基本概念。

## 7. 实验设计

- **模型**：OLMoE-1B-7B、Qwen3-30B-A3B；
- **任务**：MMLU、GSM8K、HumanEval、多语言任务（专家使用模式不同）；
- **必须产出**：
  1. 精度 vs 平均激活专家数；
  2. MoE 层耗时 / 端到端吞吐 vs batch size（原实现 vs 你的方法）；
  3. "batch 内被激活的不同专家数" vs batch size（理论估算 vs 实测）；
  4. 鸿沟分析：激活参数减少 X% → 真实加速 Y% 的差距分解。

## 8. 答辩追问示例

1. 128 个专家、top-8 路由，batch=16 时，期望有多少个不同的专家被激活？batch=64 呢？（假设路由均匀）
2. 你的方法让平均激活专家数减少了 30%，为什么 decode 只快了 5%？
3. 每 token 不同的 k，在 fused MoE kernel 中是怎么实现的？对 padding 有什么影响？
4. 动态路由在 CUDA Graph 下怎么处理？
5. 如果部署时开启专家并行（EP=8），你的方法的收益会变大还是变小？

## 9. Agent 使用建议

- 让 Agent 帮你插桩收集专家激活统计、写分析脚本；
- 修改 fused MoE 路径时，**索引计算和 padding** 是 bug 高发区，必须与 PyTorch 朴素实现逐元素对比；
- 性能对比注意公平：路由、排序的开销都要计入。

## 10. 参考资料

- Harder Tasks Need More Experts: Dynamic Routing in MoE Models：arXiv 2403.07652
- Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts LLMs：arXiv 2402.14800
- DeepSeek-V3 技术报告：arXiv 2412.19437；DeepSeekMoE：arXiv 2401.06066
- OLMoE：arXiv 2409.02060
- MegaBlocks：arXiv 2211.15841
- vLLM fused_moe 源码：`vllm/model_executor/layers/fused_moe/`（以锁定版本为准）
- DeepSeek EPLB：https://github.com/deepseek-ai/EPLB
