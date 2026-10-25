# 环境建议：硬件 / 软件 / 模型 / 数据

## 1. 硬件分档

| 档位 | 典型配置 | 适合的项目 |
|---|---|---|
| **S 档** | 1× RTX 3090/4090/A10（24GB） | P00、P03、P04 |
| **M 档** | 1× A100/H100（80GB）或 2× 24GB | P01、P02 |
| **L 档** | 4–8× A100/H100 | P02 的进阶部分（转换训练），以及各项目在大模型上的验证 |

注意事项：
- **加速比与硬件强相关**：算力 / 带宽比不同的 GPU，roofline 拐点不同，投机解码、dLLM、量化等的收益会明显变化。报告中必须写明硬件，有条件的话在两种 GPU 上验证结论；
- **低精度格式的硬件支持**：FP8 需要 Ada（4090 / L40）或 Hopper（H100 / H800 / H20）及以上；NVFP4 / MXFP4 的原生支持需要 Blackwell；
- **batch-invariant kernel**（候选池中的 RL 训推一致性项目会用到）通常要求 compute capability ≥ 8.0（Ampere 及以上）。

## 2. 软件

- **推理引擎**：vLLM 或 SGLang，**开题时锁定一个 commit**；
  - vLLM 近期版本默认使用 Model Runner V2，部分插件接口（如 logits processor）与旧版 V1 runner 不同；
  - SGLang 的 dLLM 支持、投机解码等功能更新频繁，注意查看对应版本的文档；
- **Python 环境**：推荐 `uv`；只改 Python 代码时，vLLM 可以复用预编译 kernel（如 `VLLM_USE_PRECOMPILED=1 uv pip install -e .`，以锁定版本的官方文档为准）；
- **Kernel 开发**：Triton 优先；
- **训练**：PyTorch FSDP、HF Transformers / TRL；投机解码草稿器用 Speculators（vLLM）或 SpecForge（SGLang）；量化用 LLM Compressor；线性注意力用 flash-linear-attention；
- **评测**：lm-evaluation-harness、lighteval、evalscope；
- **Profiling**：Nsight Systems、PyTorch Profiler。

## 3. 模型建议

先用**小模型快速迭代**，主实验用 7B–8B 级别模型，有条件再在更大模型上验证结论。

| 用途 | 推荐模型 |
|---|---|
| 快速开发 | Qwen3-0.6B / 1.7B（或同代更新的小模型） |
| Dense 主实验 | Qwen3-8B、Llama-3.1-8B-Instruct |
| 推理（思考）模型 | DeepSeek-R1-Distill-Qwen-1.5B / 7B、Qwen3 thinking 模式 |
| 投机解码 | 目标模型 Llama-3.1-8B / Qwen3-8B + 公开的 EAGLE-3 / DFlash 草稿器 |
| 扩散语言模型 | SDAR（1.7B / 4B / 8B）、Fast-dLLM v2-7B、LLaDA2.0-mini、LLaDA-8B、Dream-7B；以引擎支持情况为准 |
| 多模态 | Qwen2.5-VL-7B、Qwen3-VL 同级别模型 |
| MoE | OLMoE-1B-7B（开发）、Qwen3-30B-A3B（主实验） |
| 长上下文 | Llama-3.1-8B-Instruct、Qwen3-8B（128K） |

## 4. 数据集与基准

| 类别 | 数据集 |
|---|---|
| 通用能力 | MMLU、ARC、HellaSwag、WikiText2（困惑度） |
| 数学推理 | GSM8K、MATH500、AIME24 / AIME25 |
| 代码 | HumanEval、MBPP、LiveCodeBench |
| 对话 / 投机解码 | MT-Bench、SpecBench、ShareGPT |
| 长上下文 | RULER、LongBench、InfiniteBench、needle-in-a-haystack |
| 多模态 | TextVQA、ChartQA、DocVQA、MMBench、Video-MME |
| 在线负载 | ShareGPT、LMSYS-Chat-1M、Azure LLM Inference Traces、BurstGPT |
