# P08 · 低比特量化算法的部署：从精度到真实加速

> **难度** ★★★★ ｜ **算法 : 系统** 6 : 4 ｜ **建议人数** 1–2 ｜ **最低硬件** 1×24GB（FP8 需 Ada / Hopper；FP4 需 Blackwell） ｜ **基座** vLLM（compressed-tensors / 自定义量化方法）+ LLM Compressor

## 1. 背景

量化是最成熟、部署最广泛的推理加速手段，但论文和部署之间同样存在很大鸿沟：

- **仅权重量化（W4A16 / W8A16）**：GPTQ、AWQ 等，主要加速 memory-bound 的 decode，依赖 Marlin 等高效混合精度 kernel；
- **权重 + 激活量化（W8A8 / W4A8 / W4A4）**：SmoothQuant、QuaRot、SpinQuant、FlatQuant 等。旋转类方法用（在线）Hadamard / 可学习变换消除激活离群值，使 W4A4 精度可用。FlatQuant 报告 LLaMA-3-70B W4A4 精度损失 < 1%，并用融合 kernel 拿到了 2.3× prefill、1.7× decode 加速；
- **新的低精度格式**：FP8 已成为主流部署格式，MXFP4 / NVFP4 等微缩放格式随 Blackwell 普及；
- **算法-系统协同设计**：QServe 的 W4A8KV4 方案说明，量化算法必须针对 kernel 的实际约束设计（例如反量化开销、CUDA core 与 Tensor Core 的分工），否则理论加速无法兑现。

许多量化论文的精度结果很好，但**要么没有可用的 kernel，要么需要在线变换而引擎不支持，要么在大 batch 下加速消失**。

## 2. 论文 → 引擎的鸿沟

| 问题 | 说明 |
|---|---|
| Kernel 支持 | 新的量化格式 / 粒度（如特定 group size、非对称、混合精度）可能没有现成高效 kernel |
| 在线变换 | 旋转类方法需要在线 Hadamard 变换（尤其在 down_proj 前和 attention 中），需要修改模型代码并融合进 kernel |
| batch 与瓶颈转移 | W4A16 在小 batch（memory-bound）有效，大 batch（compute-bound）下可能比 FP16 / FP8 还慢；W4A4 / FP4 才能加速 compute-bound 阶段 |
| 精度在引擎内的复现 | 引擎的 kernel 实现（累加精度、scale 处理）可能与论文的模拟量化不同 |
| 硬件依赖 | INT4 Tensor Core、FP8、FP4 的支持随 GPU 代际变化 |

## 3. 项目目标

选择一种量化算法（或你自己的改进），完成：
1. 模拟量化（fake quant）复现论文精度；
2. 生成引擎可部署的量化模型：优先使用 compressed-tensors 格式和已有 kernel，必要时实现自定义量化方法或在线变换；
3. 在引擎中验证精度，并在**多 batch size** 下测量 prefill / decode 的真实加速，与 BF16、FP8、主流 W4A16 方案对比；
4. 鸿沟分析，并尝试"kernel 友好"的算法改进。

## 4. 候选路线

| 路线 | 例子 | 说明 |
|---|---|---|
| A. 改进算法 + 复用已有 kernel | 改进 GPTQ / AWQ 的校准方法、混合精度分配，输出 Marlin / Machete 等 kernel 支持的格式 | 系统工作量较小，重点在算法与精度；必须分析加速是否随 batch 消失 |
| B. 旋转类 W4A8 / W4A4 部署 | QuaRot / SpinQuant / FlatQuant | 需要实现在线变换并接入引擎，系统工作量大，加速潜力也大 |
| C. 新格式（FP8 / MXFP4 / NVFP4） | 为 FP4 设计更好的 scale 选择或旋转 | 需要对应硬件；最贴近 2026 年的部署趋势 |
| D. KV Cache 量化 | KIVI 风格的 INT4 / INT2 KV | 与 P05 的思路相关，需要定制 attention kernel |

## 5. 任务分解

### 5.1 阶段一：算法复现（必做）
- 模型：Qwen3-8B / Llama-3.1-8B（可补充 Qwen3-30B-A3B 等 MoE）；
- 模拟量化复现：WikiText2 困惑度 + 下游任务（MMLU、GSM8K、ARC、HellaSwag 等，用 lm-evaluation-harness）；
- 与 RTN、GPTQ、AWQ、FP8 等 baseline 对比。

### 5.2 阶段二：引擎部署（必做）
- 路线 A：用 LLM Compressor（或自己的工具）导出 compressed-tensors 格式，确认使用的是哪个 kernel；
- 路线 B / C / D：在 vLLM 中实现自定义量化方法（`QuantizationConfig` + `LinearMethodBase`，通过注册机制接入），在模型代码中插入在线变换，并尽量融合到 kernel（Triton）中；
- 正确性：引擎内精度与模拟量化对齐（差异需能解释）。

### 5.3 阶段三：鸿沟分析与改进（必做分析，改进为加分项）
- 加速比 vs batch size（prefill 与 decode 分开）：BF16 / FP8 / W4A16 / 你的方法；
- Kernel 级分析：GEMM 的实际 TFLOPS / 带宽利用率；在线变换和反量化的开销占比；
- Kernel 友好的改进：例如调整量化粒度以匹配 kernel tile、减少在线变换的位置、混合精度分配时把 kernel 效率纳入目标。

## 6. 关键知识点（汇报必须讲清楚）

**算法**
- 对称 / 非对称、per-tensor / per-channel / per-group 量化；
- 激活离群值问题，SmoothQuant 的迁移思想，旋转为什么能消除离群值（以及为什么旋转是"可吸收"的）；
- GPTQ 的二阶误差补偿思想；
- FP8 / MXFP4 / NVFP4 的格式差异（指数位、尾数位、scale 粒度）。

**系统**
- 量化 GEMM kernel 的基本结构：加载低比特权重 → 反量化 → Tensor Core 计算；反量化在 CUDA core 上的开销；
- 为什么 W4A16 只在 memory-bound 时有效（roofline 分析）；
- 在线 Hadamard 变换的计算复杂度与融合方法；
- 不同 GPU 代际对低精度格式的硬件支持。

## 7. 实验设计

- **模型**：Qwen3-8B / Llama-3.1-8B（必做），一个 MoE 或更大模型（选做）；
- **精度**：困惑度 + 至少 5 个下游任务，包含数学推理（量化对推理任务往往更敏感）；
- **必须产出**：
  1. 精度对比表（模拟量化 vs 引擎内）；
  2. decode / prefill 加速比 vs batch size（多方案）；
  3. 在线服务吞吐 vs 请求率；
  4. kernel 时间分解（GEMM 主体 / 反量化 / 在线变换）；
  5. 精度-吞吐 Pareto 图。

## 8. 答辩追问示例

1. 为什么 W4A16 在 batch=1 时能接近 4× 加速 GEMV，batch=256 时却可能比 BF16 还慢？
2. 旋转矩阵为什么可以被"吸收"进权重？哪些位置的旋转无法吸收、必须在线计算？
3. 你的方法在引擎中的精度比模拟量化低了 X 点，可能的原因有哪些？
4. NVFP4 和 MXFP4 的 scale 粒度有什么不同？对精度有什么影响？
5. 你使用的 kernel 在反量化上花了多少时间？怎么测的？

## 9. Agent 使用建议

- 量化校准、评测流水线可以交给 Agent；
- **风险**：量化的 pack / unpack 格式（比特排布、scale 的存储顺序）极易出错且错误往往"静默"，例如精度只是轻微下降。必须用逐层输出对比来验证；
- 自定义量化方法接入引擎时，注意 TP 切分对 group 量化的影响。

## 10. 参考资料

- GPTQ：arXiv 2210.17323；AWQ：arXiv 2306.00978；SmoothQuant：arXiv 2211.10438
- QuaRot：arXiv 2404.00456；SpinQuant：arXiv 2405.16406；FlatQuant：arXiv 2410.09426
- QServe（W4A8KV4 系统协同设计）：arXiv 2405.04532
- KIVI：arXiv 2402.02750
- OCP Microscaling（MX）格式规范；NVIDIA NVFP4 相关技术博客
- LLM Compressor：https://github.com/vllm-project/llm-compressor
- vLLM 量化文档与源码 `vllm/model_executor/layers/quantization/`
