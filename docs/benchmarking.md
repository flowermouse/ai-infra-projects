# 实验与评测规范

本项目集的核心诉求是"**真实**的加速收益"，所以对实验的可信度要求比一般算法论文更高。本文档规定所有项目都要遵守的评测规范。

---

## 1. 双重评测：质量 + 效率

每个项目都必须**同时**报告：

1. **质量**：算法是否保持了模型能力？
   - 无损方法（如投机解码）：验证输出与 baseline 一致（greedy 逐 token 一致；采样时做分布一致性检验）；
   - 有损方法（如剪枝、量化、KV 压缩、早停）：在**下游任务**上评测精度，不能只看困惑度；
   - **在引擎内**测质量：HF 上的精度不代表引擎内的精度，因为实现细节和数值都可能不同。
2. **效率**：在**真实引擎、真实配置**下的加速。

最终要呈现的是**质量-效率权衡曲线**（Pareto 前沿），而不是单个数字。

## 2. 什么算"真实加速"

报告加速比时必须满足以下条件，否则只能算"论文式加速"：

- [ ] **Baseline 是开启了全部常规优化的上游引擎**：CUDA Graph、FlashAttention / FlashInfer、prefix caching（如适用）、chunked prefill 等，且配置与你的方法完全一致；
- [ ] **报告多个 batch size / 并发度 / 请求率下的结果**：至少覆盖 1、中等（如 8–16）、较大（如 64 以上，或接近显存上限）三档。只报告 batch=1 的结果不算完成；
- [ ] **报告端到端指标**：吞吐（tokens/s）、TPOT、TTFT。kernel 级加速只能作为辅助分析；
- [ ] **在线服务场景至少一组**：用 `vllm bench serve` / `sglang.bench_serving` 做请求率扫描；
- [ ] **加速比的分母写清楚**：相对谁？哪个配置？

此外，**建议同时报告"论文式加速"**（HF、batch=1 下测得的数字），以及从它到"真实加速"之间每一步的损耗。这就是"鸿沟分析"，是报告中最有价值的部分之一。

## 3. 指标定义（统一口径）

| 指标 | 定义 |
|---|---|
| **TTFT** | 请求到达到第一个输出 token 返回的时间（含排队） |
| **TPOT** | (E2E − TTFT) / (输出 token 数 − 1) |
| **ITL** | 相邻输出 token 的时间间隔 |
| **Throughput** | 输出 tokens/s（或总 tokens/s，需写明） |
| **Goodput** | 满足 SLO 的请求速率 |
| **接受长度（投机解码 / dLLM）** | 每次目标模型前向平均产出的 token 数，需写明是否包含 bonus token |
| **TPF（dLLM）** | Tokens Per Forward，每次前向平均解码出的 token 数 |

延迟类指标至少报告 P50 和 P90 / P99。

## 4. 推荐工具

- 效率：`vllm bench serve`、`vllm bench throughput`、`vllm bench latency`（旧版本为 `benchmarks/` 下的脚本）；`python -m sglang.bench_serving`、`sglang.bench_one_batch`；
- 质量：`lm-evaluation-harness`（支持 vLLM 后端）、`lighteval`、`evalscope`；数学推理可用各模型官方的评测脚本；
- Profiling：`nsys`、PyTorch Profiler、vLLM 的 torch profiler 接口（以锁定版本为准）。

## 5. 可复现性要求

每份实验报告必须写明：

1. **硬件**：GPU 型号与数量、互联方式、CPU、内存；
2. **软件**：上游框架 commit hash、CUDA / PyTorch / 驱动版本；
3. **模型**：模型名、精度、关键启动参数（max_num_seqs、max_num_batched_tokens、gpu_memory_utilization、是否开 CUDA Graph 等）；
4. **数据**：数据集、采样方式、请求数、随机种子；
5. **过程**：warmup 方式；每个数据点至少重复 **3 次**，报告均值和方差；
6. **脚本**：所有实验的一键复现脚本放在 `experiments/` 下。

## 6. 常见陷阱 checklist

- [ ] Baseline 是否开了 CUDA Graph？关掉 CUDA Graph 的 baseline 会让很多"加速"变得虚高；
- [ ] 输出长度是否受控（`ignore_eos` / `max_tokens`）？有损方法可能改变输出长度，导致吞吐不可比，要分别报告"每请求延迟"和"每 token 吞吐"；
- [ ] 早停 / 剪枝类方法：token 数减少带来的加速和每 token 变快带来的加速，要**分开报告**；
- [ ] 测 prefix cache 相关功能时，每轮实验之间是否重置了缓存？
- [ ] 高并发下，客户端本身是否成为瓶颈？
- [ ] 质量评测的 prompt 模板、截断方式、采样参数是否与论文 / baseline 一致？
- [ ] 结果"好得不正常"时，先找 bug。

## 7. 图表要求

必须产出的通用图表：

1. **加速比 vs batch size（或请求率）曲线**：标出加速消失的临界点；
2. **质量-效率权衡曲线**（有损方法）；
3. **鸿沟瀑布图**：从"理论加速"到"HF 实测"再到"引擎 batch=1"、"引擎大 batch"，每一步损失了多少，各自的原因是什么；
4. **Profiling 时间分解图**：你的算法引入的额外开销在哪里。

每张图都要配一段文字：**看到了什么、为什么会这样**。
