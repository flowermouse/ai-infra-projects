# P12 · RL 训推不一致：算法校正 vs 系统对齐

> **难度** ★★★★ ｜ **算法 : 系统** 6 : 4 ｜ **建议人数** 1–2 ｜ **最低硬件** 2×80GB（或 2×24GB + 0.6B 模型） ｜ **基座** vLLM（rollout）+ HF / FSDP（训练），或 verl / slime 等框架

## 1. 背景

在 GRPO / PPO 等 RL 后训练中，rollout 由推理引擎（vLLM / SGLang）生成，训练时由训练框架（FSDP / Megatron）重新计算 logprob。理论上两者应该相同，但实际上：

- 推理引擎与训练框架使用不同的 kernel、不同的并行切分、不同的精度处理；
- 推理 kernel 通常**不具备 batch invariance**：同一样本的计算结果会随 batch 中其他样本变化（Thinking Machines Lab，2025-09）；
- 为了 rollout 吞吐，推理端常使用 FP8 等低精度。

结果是 **"训推不一致"**：本应 on-policy 的算法实际上变成了 off-policy，严重时导致训练崩溃。这个问题在 2025–2026 年受到高度关注，并出现了两条解决路线：

- **算法校正路线**：承认不一致，用重要性采样修正。例如 Truncated Importance Sampling（TIS）、Masked Importance Sampling（MIS），以及各种序列级 / token 级的校正与过滤方法；
- **系统对齐路线**：从根本上消除不一致。vLLM 提供 batch invariance 模式（`VLLM_BATCH_INVARIANT=1`）；2026 年 10 月，vLLM 发布训推 bitwise 一致的 RL 方案，训练前向直接调用推理 kernel，使 rollout 与训练的 logprob 差为 0，并报告 rollout 吞吐约为基线的 88–91%；SkyRL 提出 IsoExec。

**两条路线各有代价**：算法校正有偏差和方差问题，系统对齐有性能开销和工程复杂度。**哪种更好？在什么条件下？** 这正是一个需要同时理解算法与系统的研究问题。

## 2. 项目目标

1. 搭建一个最小的 GRPO 训练系统（vLLM rollout + HF / FSDP 训练），或基于 verl / slime 等框架；
2. **量化训推不一致**：在不同设置（BF16 / FP8 rollout、不同 batch 组成、不同 TP）下测量 logprob 差异的分布；
3. 实现并对比至少两种**算法校正**方法（如 TIS、MIS 或序列级校正）；
4. 实现**系统对齐**方案（启用 vLLM batch invariance 模式，并让训练端与之对齐），测量其性能开销；
5. 在同一任务上对比：不处理 / 算法校正 / 系统对齐，比较训练稳定性、最终精度和 wall-clock 效率。

## 3. 论文 → 引擎的鸿沟

| 问题 | 说明 |
|---|---|
| 不一致的来源复杂 | kernel 选择、batch 组成、精度、并行切分都会引入差异，需要逐项隔离 |
| 算法校正依赖准确的推理端 logprob | 需要从引擎中拿到与采样分布一致的 logprob，注意采样参数（temperature、top-p）的影响 |
| 系统对齐的开销 | batch-invariant kernel 牺牲了部分并行策略，rollout 吞吐会下降 |
| 训练端对齐的难度 | 训练端要使用与推理端数值一致的前向实现，同时保留可用的反向传播 |

## 4. 任务分解

### 4.1 阶段一：搭建与量化（必做）
- 模型：Qwen3-0.6B / 1.7B；任务：GSM8K（规则奖励），可选 MATH 子集；
- 搭建同步 GRPO（rollout 与训练分离，权重同步可使用 vLLM 的 Native RL APIs）；
- 量化不一致：token 级 logprob 差、序列级 KL，按 token 位置、概率区间、batch 组成分组统计；对比 BF16 与 FP8 rollout。

### 4.2 阶段二：算法校正（必做）
- 实现 TIS、MIS（或其他校正 / 过滤方法），理解它们的偏差-方差权衡；
- 在 RL 训练中对比 reward 曲线、训练稳定性（是否崩溃）、测试集精度。

### 4.3 阶段三：系统对齐（必做）
- 在 vLLM 中启用 batch invariance 模式，验证同一样本在不同 batch 组成下 logprob 完全一致；
- 让训练端的前向与推理端数值对齐（例如在训练前向中使用相同的 batch-invariant kernel，参考 vLLM 训推一致性方案的思路，用带 backward 的 `autograd.Function` 包装推理 kernel）；
- 测量 rollout 吞吐与训练吞吐的开销。

### 4.4 阶段四：综合对比与分析（必做）
- 同一任务、同一计算预算下对比三种方案：reward vs 步数、reward vs wall-clock、最终精度；
- 分析：在什么设置下算法校正就足够？什么情况下必须系统对齐？

### 4.5 进阶（加分）
- 扩展到异步 RL（staleness > 0），分析 off-policy 程度与训推不一致的叠加效应；
- 扩展到 MoE 模型（路由不一致会放大差异）或长链推理任务；
- 提出新的校正方法，或"部分对齐"方案（只对齐影响最大的算子）。

## 5. 关键知识点（汇报必须讲清楚）

**算法**
- PPO / GRPO 的目标函数；on-policy 假设在其中的作用；
- 重要性采样的偏差与方差；TIS / MIS 的原理与区别；
- 为什么很小的 logprob 差异（如 1e-3 量级）也可能导致训练崩溃（长序列累积、低概率 token 的放大）。

**系统**
- 推理 kernel 为什么不具备 batch invariance（split-K、split-KV、归约顺序随 batch 变化）；
- vLLM batch invariance 模式的实现思路与代价；
- 训推 bitwise 一致需要满足的条件（kernel 确定性、batch 无关、prefill / decode 路径一致、跨 GPU 一致）；
- rollout 与训练之间的权重同步。

## 6. 实验设计

- **模型 / 任务**：Qwen3-0.6B 或 1.7B + GSM8K；
- **对比配置**：不处理 / TIS / MIS / 系统对齐；BF16 与 FP8 rollout；
- **必须产出**：
  1. 训推 logprob 差异的分布（多种设置）；
  2. 各方案的 reward vs 步数、reward vs wall-clock 曲线（每个配置 2–3 个种子）；
  3. 系统对齐的吞吐开销；
  4. 最终精度对比表与分析结论。

## 7. 答辩追问示例

1. 为什么 `torch.mm(a[:1], b)` 和 `torch.mm(a, b)[:1]` 的结果可能不同？
2. TIS 截断重要性权重引入了什么偏差？截断阈值怎么选？
3. FP8 rollout 的 logprob 差异比 BF16 大多少？哪些 token 差异最大？
4. 系统对齐让 rollout 慢了多少？主要开销来自哪些算子？
5. 你的结论是什么：什么场景下推荐算法校正，什么场景下推荐系统对齐？

## 8. Agent 使用建议

- GRPO 训练循环、统计分析脚本可以交给 Agent，但 **loss 的实现细节**（token 级 vs 序列级归一化、mask、重要性权重的计算位置）必须逐行核对；
- **风险**：所有一致性测试必须使用 **bitwise** 比较（`torch.equal`），不能用 `allclose` 代替；
- RL 实验方差大，每个配置至少 2–3 个随机种子，结论要谨慎。

## 9. 参考资料

- Defeating Nondeterminism in LLM Inference（Thinking Machines Lab，2025-09）：https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- *Your Efficient RL Framework Secretly Brings You Off-Policy RL Training*（TIS，技术博客，2025）
- *When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch*（技术博客，2025）
- vLLM Batch Invariance 文档：https://docs.vllm.ai/usage/batch_invariance.html
- vLLM 训推 bitwise 一致性（2026-10）：https://vllm.ai/blog/2026-10-14-train-inference-consistency
- vLLM Native RL APIs（2026-05）：https://vllm.ai/blog/2026-05-28-native-rl-apis
- vLLM RL 博客合集（含 IsoExec、vime 等）：https://vllm.ai/blog/tags/reinforcement-learning
- DeepSeekMath（GRPO）：arXiv 2402.03300
- verl：https://github.com/verl-project/verl ；slime：https://github.com/THUDM/slime
