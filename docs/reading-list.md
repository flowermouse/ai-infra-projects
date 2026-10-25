# 阅读清单

按主题整理。**★ 为所有人必读**（热身阶段读完），其余按项目选读。各项目文档末尾还有更具体的参考资料。

---

## 0. 推理系统基础（★ 所有人必读）

- ★ Efficient Memory Management for LLM Serving with **PagedAttention**（vLLM，SOSP'23）：arXiv 2309.06180
- ★ **SGLang**: Efficient Execution of Structured Language Model Programs（RadixAttention）：arXiv 2312.07104
- ★ **Sarathi-Serve**（chunked prefill，OSDI'24）：arXiv 2403.02310
- ★ **FlashAttention-2**：arXiv 2307.08691
- ★ *Making Deep Learning Go Brrrr From First Principles*（Horace He，技术博客）
- ★ nano-vllm 源码：https://github.com/GeeeekExplorer/nano-vllm
- ★ **Speculative Decoding: Performance or Illusion?**（MLSys 2026）：arXiv 2601.11580。这篇是"论文加速 vs 真实加速"的范例，强烈推荐所有人读

## 1. 投机解码（P01、P02）

- Fast Inference from Transformers via Speculative Decoding：arXiv 2211.17192
- Medusa：arXiv 2401.10774
- EAGLE / EAGLE-2 / EAGLE-3：arXiv 2401.15077 / 2406.16858 / 2503.01840
- DFlash（块扩散草稿器，ICML 2026）：arXiv 2602.06036
- MagicDec：arXiv 2408.11049
- SmartSpec（基于 goodput 的投机解码）：arXiv 2406.14066
- Lookahead Decoding：arXiv 2402.02057；SuffixDecoding：arXiv 2411.04975；LayerSkip：arXiv 2404.16710

## 2. 扩散语言模型（P03、P04）

- LLaDA：arXiv 2502.09992；Dream 7B：arXiv 2508.15487
- Block Diffusion（BD3-LM）：arXiv 2503.09573
- Fast-dLLM：arXiv 2505.22618；Fast-dLLM v2：arXiv 2509.26328
- SDAR：arXiv 2510.06303
- dKV-Cache：arXiv 2505.15781；dLLM-Cache：arXiv 2506.06295
- dInfer：arXiv 2510.08666
- LLaDA2.0：arXiv 2512.15745
- SGLang dLLM 支持：https://lmsys.org/blog/2025-12-19-diffusion-llm
- vLLM DiffusionGemma：https://vllm.ai/blog/2026-06-10-diffusion-gemma

## 3. KV Cache 压缩与稀疏注意力（P05、P06）

- StreamingLLM：arXiv 2309.17453；H2O：arXiv 2306.14048；SnapKV：arXiv 2404.14469
- R-KV：arXiv 2505.24133
- Quest：arXiv 2406.10774；DuoAttention：arXiv 2410.10819
- MInference：arXiv 2407.02490；FlexPrefill：arXiv 2502.20766；XAttention：arXiv 2503.16428
- Native Sparse Attention：arXiv 2502.11089
- The Sparse Frontier（ACL 2026 Findings）

## 4. 多模态 Token 压缩（P07）

- FastV：arXiv 2403.06764；SparseVLM：arXiv 2410.04417；VisionZip：arXiv 2412.04467
- TopV（CVPR 2025）：arXiv 2503.18278

## 5. 量化（P08）

- GPTQ：arXiv 2210.17323；AWQ：arXiv 2306.00978；SmoothQuant：arXiv 2211.10438
- QuaRot：arXiv 2404.00456；SpinQuant：arXiv 2405.16406；FlatQuant：arXiv 2410.09426
- QServe：arXiv 2405.04532
- KIVI：arXiv 2402.02750

## 6. MoE 与混合架构（P09、P10）

- DeepSeekMoE：arXiv 2401.06066；DeepSeek-V3 技术报告：arXiv 2412.19437
- Harder Tasks Need More Experts：arXiv 2403.07652
- Not All Experts are Equal：arXiv 2402.14800
- MegaBlocks：arXiv 2211.15841
- Mamba-2 / Transformers are SSMs：arXiv 2405.21060；Gated DeltaNet：arXiv 2412.06464
- The Mamba in the Llama：arXiv 2408.15237；LoLCATs：arXiv 2410.10254

## 7. 高效推理与 RL（P11、P12）

- DEER：arXiv 2504.15895；EAT：arXiv 2509.26522
- DeepConf（ICLR 2026）：arXiv 2508.15260
- Towards Concise and Adaptive Thinking in Large Reasoning Models: A Survey：arXiv 2507.09662
- DeepSeekMath（GRPO）：arXiv 2402.03300
- Defeating Nondeterminism in LLM Inference（Thinking Machines Lab，2025-09）
- vLLM 博客 RL 系列（Native RL APIs、训推 bitwise 一致性等）：https://vllm.ai/blog/tags/reinforcement-learning

## 8. 持续跟踪的信息源

- vLLM 博客：https://vllm.ai/blog ；GitHub 中带 `RFC` / `roadmap` 标签的 Issue
- LMSYS / SGLang 博客：https://lmsys.org/blog
- 会议：MLSys、OSDI / SOSP、NeurIPS / ICML / ICLR 中的高效推理论文
