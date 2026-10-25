# P00 · 热身：推理引擎速成，从 Roofline 到 vLLM 插件

> **所有人必做** ｜ **周期** 2–3 周 ｜ **硬件** 1×24GB 即可 ｜ **基座** vLLM（SGLang 选做）

## 目标

在正式项目开始前，用 2–3 周建立对推理引擎的基本直觉，掌握后续项目都会用到的工具链。完成后你应该能够：

- 用 roofline 模型**预测**一个推理配置的性能，并用实测验证；
- 熟练使用 benchmark 和 profiling 工具；
- 知道 vLLM 中一个请求是怎么被调度和执行的；
- 写出一个最简单的 vLLM 扩展，并测出它的开销。

## 任务

### T1：Roofline 预测 vs 实测（必做）

1. 选定模型（Qwen3-1.7B 或 Qwen3-8B）和你的 GPU，手算：
   - 权重大小、每 token 的 KV Cache 大小；
   - batch = 1 / 8 / 32 / 128 时，decode 一步的理论耗时下界（分别按带宽和算力计算，取较大者）；
   - 给定输入长度下的 prefill 理论耗时。
2. 用 `vllm bench latency` / `vllm bench throughput` 实测，与预测对比，解释偏差。
3. 画一张 "batch size → decode 每步耗时 / 吞吐" 的曲线，标出从 memory-bound 转向 compute-bound 的拐点。

### T2：HF vs vLLM 的差距（必做）

1. 用 HF Transformers `generate()` 与 vLLM 跑同样的批量生成任务，对比吞吐；
2. 在 vLLM 中逐项关闭 CUDA Graph（`enforce_eager`）、prefix caching、chunked prefill，观察性能变化；
3. 写一段分析：**为什么只在 HF 上测出的加速比不可信？** 这是后续所有项目的出发点。

### T3：Profiling（必做）

1. 用 `nsys` 或 PyTorch Profiler 抓取 vLLM 一个 decode step 和一个 prefill step；
2. 找出时间都花在哪里：attention、GEMM、通信、采样、CPU 调度开销各占多少；
3. 对比 batch = 1 与 batch = 64 下的时间分布变化。

### T4：代码走读（必做）

在你锁定的 vLLM 版本中，跟踪一个请求的完整生命周期，**自己画**一张时序图，至少覆盖：API Server → Engine Core → Scheduler（KV block 分配）→ Model Runner（输入准备、CUDA Graph）→ Attention backend → Sampler → 输出处理。图中标出与你**后续项目**相关的模块。

> 可以让 Agent 帮你定位代码，但图必须自己画，并能逐个模块讲清楚。

### T5：写一个最小插件（必做）

任选其一：
- **自定义 Logits Processor**：例如"当某请求连续生成 N 个相同 token 时强制输出 EOS"（注意你的版本中 V1 / V2 model runner 的接口差异）；
- **Out-of-tree 模型**：通过插件机制注册一个模型变体（如把 Qwen3 的某层替换成你自己的实现），并验证输出与原模型一致。

测量插件在不同 batch size 下引入的额外开销。

### T6（选做）：在 SGLang 上重复 T1–T3

对比两个框架在同一配置下的性能，并尝试解释差异。

## 交付物

一份 5–10 页的热身报告（包含 T1–T5 的数据、图和分析）+ 代码。在组会上做 10 分钟汇报。

## 参考

- [docs/core-knowledge.md](../../docs/core-knowledge.md)、[docs/benchmarking.md](../../docs/benchmarking.md)、[docs/engine-extension-guide.md](../../docs/engine-extension-guide.md)
- vLLM 文档：Architecture Overview、Custom Logits Processors、Plugin System
- nano-vllm（约 1200 行的精简实现，适合辅助理解）：https://github.com/GeeeekExplorer/nano-vllm
- Mini-SGLang：https://lmsys.org/blog/2025-12-17-minisgl
- *Making Deep Learning Go Brrrr From First Principles*（Horace He，关于 compute / memory / overhead bound 的经典博客）
