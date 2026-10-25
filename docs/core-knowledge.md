# 核心知识清单

无论选哪个项目，下面这些知识都是**所有同学必须掌握的公共基础**，结题答辩时会随机抽问。各项目还有自己的专项知识点（算法 + 系统），见各项目文档。

> 这里的"掌握"不是背定义，而是能**估算**、能**推导**、能**用它解释你在实验里看到的现象**。

---

## 1. 理论加速 vs 真实加速（本项目集最核心的概念）

| 知识点 | 掌握标准 |
|---|---|
| FLOPs ≠ 延迟 | 能举例说明"FLOPs 减半但延迟几乎不变"的情形，并解释原因 |
| Compute-bound vs Memory-bound | 能判断一个算子 / 一个阶段（prefill、decode、某个 kernel）属于哪一类，并说明依据 |
| Roofline 模型 | 能计算算术强度（FLOPs / Bytes），并结合硬件的峰值算力和带宽判断瓶颈 |
| Batch size 的影响 | 能解释为什么"batch=1 下有效的加速方法"在大 batch 下会失效（投机解码、dLLM、稀疏化……） |
| 不规则性的代价 | 能解释为什么 token 级稀疏、动态 shape、per-request 不同的计算路径难以拿到真实加速（kernel 效率、CUDA Graph、padding） |
| 系统开销 | 能识别 CPU 调度开销、kernel launch 开销、Python 开销在小 batch、小模型下的占比 |
| Amdahl 定律 | 只优化 attention，端到端最多能加速多少？能用 profiling 数据估算 |

**自测题：**
1. 一个 8B 模型在 H100 上 BF16 推理，batch=1 时 decode 的理论 tokens/s 上限是多少（HBM 带宽约 3.35 TB/s）？batch=256 时瓶颈会转移到哪里？
2. 某 token 剪枝方法让 attention 的 FLOPs 减少 50%，但在 batch=64、上下文 2K 的设置下端到端只快了 3%。请给出至少两个可能的原因。
3. 投机解码每步验证 5 个 token。为什么在 batch=1 时这几乎是"免费的"，在 batch=128 时却不是？

## 2. LLM 推理基础

| 知识点 | 掌握标准 |
|---|---|
| Prefill 与 Decode | 两阶段的计算特征差异；为什么 prefill 是 compute-bound、decode 是 memory-bound |
| KV Cache | 能**手算**给定模型的 KV Cache 大小；知道 GQA / MLA / 滑动窗口 / 线性注意力对它的影响 |
| 注意力变体 | MHA / MQA / GQA / MLA 的区别；FlashAttention 的基本思想（tiling + online softmax），以及它**不输出注意力矩阵**这一点对很多算法的影响 |
| 采样 | greedy / temperature / top-p / top-k 的实现位置；logits processor 的作用 |

**自测题：**
1. Qwen3-8B（36 层，8 个 KV head，head_dim=128）在 BF16 下每个 token 的 KV Cache 占多少字节？一张 24GB 显卡加载完权重后，大约还能放多少 token？
2. 很多 KV 压缩和 token 剪枝算法依赖注意力分数，而 FlashAttention 不输出注意力矩阵。有哪些办法拿到所需的分数？各自的代价是什么？

## 3. 推理引擎核心机制

| 知识点 | 掌握标准 |
|---|---|
| PagedAttention | block table、物理 / 逻辑 block 的映射；它为什么让"删除中间 token"变得困难 |
| Continuous Batching | iteration-level scheduling；不同请求处于不同阶段时如何混在一个 batch 中 |
| Chunked Prefill | 长 prompt 切块，与 decode 混合调度；token budget 的含义 |
| Prefix Caching | vLLM 基于 block hash vs SGLang 的 RadixAttention；什么样的算法会破坏 prefix 复用 |
| CUDA Graph | 为什么 decode 需要 CUDA Graph；它对动态 shape 和动态控制流的限制 |
| 投机解码框架 | 引擎中 propose → verify → rejection sample → KV 回滚 的流程 |

**自测题：**
1. 画出你锁定版本的 vLLM（或 SGLang）中，一个请求从进入调度器到输出 token 的流程图，并标出你的算法要修改的位置。
2. 你的算法会让每个请求每一步的计算量不同（例如动态投机长度、动态 top-k、动态剪枝率）。这和 CUDA Graph、continuous batching 如何共存？

## 4. 质量与效率的双重评测

| 知识点 | 掌握标准 |
|---|---|
| 效率指标 | TTFT / TPOT / ITL / 吞吐 / Goodput 的定义；为什么必须报告**不同 batch size / 请求率**下的结果 |
| 质量指标 | 你的任务对应的精度指标（准确率、pass@k、接受长度、困惑度……），以及为什么困惑度往往不够 |
| 无损 vs 有损 | 投机解码是"分布无损"的；剪枝 / 量化 / 早停是有损的。有损方法必须报告"质量-效率"权衡曲线 |
| 数值差异 | 同一模型在 HF 与 vLLM 下的输出可能不同（kernel、精度、batch 组成）。要区分"算法带来的精度损失"和"引擎的数值差异" |

## 5. 工具与工程能力

- 能用 `nsys` / PyTorch Profiler 看懂一个推理 step 的时间分解，定位你的算法在哪里花了时间；
- 能读懂 vLLM / SGLang 的启动参数和日志；
- 能写出可复现的 benchmark（见 [benchmarking.md](benchmarking.md)）；
- 会写简单的 Triton kernel，至少能读懂并修改现有 kernel（P05 / P06 / P08 / P09 尤其需要）。
