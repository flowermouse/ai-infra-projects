# 候选池

这里是**尚未开放**的候选选题。它们和正式项目遵循同一条主线（算法复现 → 引擎落地 → 鸿沟分析与系统感知的改进），但目前只是初稿：调研深度、任务拆分、实验设计都还没有精修。

开放流程：从候选池中挑选 → 重新调研最新进展 → 精修文档 → 移入 [`projects/`](../projects/) 并分配编号。

| 候选项目 | 方向 | 难度 | 最低硬件 | 备注 |
|---|---|---|---|---|
| [自回归 → 块扩散：模型转换、训练与部署](ar-to-block-diffusion/README.md) | 扩散语言模型 | ★★★★★ | 4–8×80GB | 目前作为 [P02](../projects/P02-dllm-inference/README.md) 的进阶部分 |
| [长上下文稀疏注意力的引擎落地](sparse-prefill-attention/README.md) | 注意力 | ★★★★ | 1×80GB | XAttention / FlexPrefill / MInference 等 |
| [多模态大模型视觉 Token 压缩的服务化](vlm-token-compression/README.md) | 多模态 | ★★★★ | 1×24GB | FastV / VisionZip / TopV 等 |
| [低比特量化算法的部署](low-bit-quantization/README.md) | 模型压缩 | ★★★★ | 1×24GB | 旋转量化、FP8 / FP4 |
| [MoE 自适应专家计算的引擎落地](moe-adaptive-experts/README.md) | MoE | ★★★★ | 1×80GB | 动态 top-k、专家剪枝、批感知路由 |
| [Transformer → 混合线性注意力：蒸馏与部署](hybrid-linear-attention-distill/README.md) | 模型架构 | ★★★★★ | 2–8×80GB | GDN / Mamba-2 混合模型 |
| [RL 训推不一致：算法校正 vs 系统对齐](rl-train-inference-mismatch/README.md) | RL 训练 | ★★★★ | 2×80GB | TIS / MIS vs batch-invariant 推理 |

也欢迎提出新的候选选题，只要符合"算法 → 引擎落地 → 真实收益"的主线即可。
