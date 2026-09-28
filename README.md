# TriEvo：从零预训练的病毒基因组 DNA 语言模型

[简体中文（主版）](README.md) | [English](README.en.md)

> 一项独立的从零预训练研究：语料构建、5 种 tokenizer 方案对比、模型实现，以及横跨三个模型规模（8.4M → 0.95B）的 **52 个受控实验**——训练栈设计遵循 Stanford CS336（Assignment 1–2）方法论，全部独立实现。

**一句话总结**：在 425 Mbp 病毒基因组语料上，参数规模在 ~85M 处饱和（再放大 11 倍参数，验证 loss 仅改善 0.0002 nats/new-base）；最优 per-base 模型（1.330）**低于实测 0–8 阶全部 Markov 基线**（阶梯 1.384 → 1.355），约一半增益来自 8 阶窗口之外的统计结构；6 倍压缩的 6-mer stride-6 tokenizer 收敛为上下文无关的 6-mer 边际预测器——对确实存在的跨 token 信号（≥0.25 nats/token）提取率为零。

---

## 1. 项目介绍与动机

基因组基础模型（DNABERT-2、Nucleotide Transformer、HyenaDNA、Evo）都面临两个通常靠直觉而非受控实验回答的设计问题：

1. **DNA 应该如何 tokenization？** 逐碱基（无压缩、最高分辨率）、k-mer（固定压缩、更大词表），还是学习式 BPE？压缩买到上下文长度——但它*卖掉了*什么？
2. **小语料规模下，放大参数还有用吗？** 病毒基因组数量可观（RefSeq 约 2 万条完整基因组），但与网络文本相比极其有限。参数–数据的平衡点在哪里？

TriEvo 用一份无泄漏语料和一组按 cohort 组织的受控实验回答这两个问题——规模刻意控制在一个人能跑完、能完全理解的程度，同时保证每一个论断都能溯源到具体 run。

## 2. 数据

| 阶段 | 细节 |
|---|---|
| 来源 | NCBI RefSeq 病毒基因组：**19,575** 条完整基因组 |
| 去冗余 | BLAST all-vs-all，**95% ANI** 聚类 → **17,339** 条代表基因组 |
| 语料规模 | 约 425 Mbp；**339,868,143** 训练 tokens（per-base tokenizer） |
| 切分 | **基因组级** 8:1:1 train/val/test——先聚类后切分，test 基因组的同源序列不出现在训练集 |
| 扩增 | 反向互补（RC）序列扩增（用于大规模训练轮次） |
| 存储 | uint16 预分词数组；EOS 拼接 + 随机起点定长窗口采样；context length 2,048（另评估过 4,096） |

## 3. 方法与技术栈

**Tokenizer 研究**——5 种方案在相同模型配置下训练：

| 方案 | 词表 | Stride | 压缩率 |
|---|---:|---|---|
| per-base | 8 | 1 | 无（1 token / 碱基） |
| 3-mer | 68 | 1 | 无（1 token / 碱基） |
| 6-mer | 4,100 | 1 / 2 / 6 | 1× / 2× / 6× |

**架构**——decoder-only Transformer，从零实现（无框架生成代码）：

- RoPE 位置编码；RMSNorm（pre-norm）；SwiGLU FFN（3 矩阵）
- 无 bias；不共享 embedding（untied）；标准多头注意力（MHA）
- Context length 2,048

| 配置 | d_model | d_ff | 层数 | 头数 | 参数量 |
|---|---:|---:|---:|---:|---:|
| tiny | 256 | 1,024 | 8 | 8 | 8.4M |
| small | 768 | 2,048 | 12 | 12 | 85M |
| large | 1,280 | 5,120 | 36 | 20 | ≈0.95B |

**手写训练栈**：

- AdamW、cosine 学习率调度、梯度裁剪、混合精度；多卡 DDP
- **Triton FlashAttention-2（forward + backward）**，fp32 累加，与参考实现对齐验证（数值误差 < 1e-2，fp16 标准容差）
- 训练于异构集群（V100-32GB / L40S-46GB / H200-143GB）；large 轮次为 4×GPU DDP；tiny / small / large 吞吐分别为 215k / 100k / 14k tokens/s

**Triton FA2 系统基准**（单卡 RTX 4090，torch 2.3.1；3 dtype × 4 head_dim × 5 序列长度；强制 PyTorch SDPA 三后端逐一对比，并用 profiler 实锤后端 dispatch）：

- 对朴素 attention（bf16）：长序列（16k）**快 5–36×**，加速比随序列长度增长——符合 IO 复杂度理论预期
- 对 SDPA mem-efficient 后端：d ≤ 32 全面领先（+1.1–2.7×），d=128 长序列领先 +1.5×
- 对官方 FlashAttention-2 kernel：d∈{16,64}、seq=4096 处**快 1.1–1.2×**；d≥32 长序列**落后 1.3–2.8×**，差距归因于流水线深度（num_stages）与 tiling——已知且有界的优化差距
- fp16 与 bf16 测量逐点一致（±2%），互为复现性验证
- 通过手算 SMEM 预算（96/99 KB）并按 head_dim 剪枝 autotune 配置，解决 d=128 shared-memory 超限

## 4. 实验协议

**52 个有训练数据的 run**，按 cohort 组织，只在受控协议*内部*做比较：

| Cohort | 协议 | Run 数 | 回答的问题 |
|---|---|---:|---|
| C1a | ctx2048、bs32、lr 3e-4、无 RC、固定 6000 步 | 15 | 数据集干预（完整 vs 95% ANI 去冗余） |
| C1b | ctx256、bs64 | 5 | 上下文长度干预 |
| C2 | lr 1e-5、RC、去冗余数据、bs8、early-stop | 15 | **三点 scaling 曲线**（唯一跨规模受控队列） |
| C3 | 仅 large：5 tokenizer × 4 lr | 20 | **Tokenizer × LR 敏感性网格** |
| C4 | RC on/off 配对（small、6-mer s6） | 2 | RC 扩增干预 |

**统一度量**：不同 stride/词表 tokenizer 的 per-token loss 不可直接比较。所有跨 tokenizer 比较使用 **nats/new-base = best_val_loss ÷ stride**（≈ bits-per-base ÷ ln 2），并对照两个基线：

- 均匀分布基线：每 token ln(vocab)（1.386 / 4.159 / 8.318 nats）
- **碱基频率零模型：1.3838 nats/base（实测）**——train 拟合 unigram、val 评估的交叉熵（`design/markov_baseline/`；旧值 1.370 为理论估计，实测作废）。*Gain* = 1.3838 − nats/new-base——模型在碱基组成之外提取到的信号量。
- **0–8 阶 Markov 阶梯（实测）**：1.3838 → 1.3554 nats/base（train 拟合、val 评估、带 backoff）——局部统计可解释信号的总上界。

**红线**：跨 cohort 的 loss 差异只报告、不作为证据（batch size、LR、停止规则、RC 跨队列同时变动，混杂无法分离）。

## 5. 结果

### 5.1 参数 scaling 饱和——瓶颈在数据，不在模型

![Scaling](design/unified_analysis/fig2_scaling_controlled_cohort.png)

受控 C2 队列（跨规模协议完全一致）的 per-base 验证 loss：

| | tiny 8.4M | small 85M | large ≈0.95B |
|---|---:|---:|---:|
| nats/new-base | 1.3481 | **1.3311** | **1.3309** |
| 到达 best 消耗 tokens | 16.7M | 23.6M | 94.4M |
| epoch 占比 | 0.05 | 0.07 | 0.28 |

三条相互印证的证据：

1. **11× 参数（85M → 0.95B）只换来 0.0002 nats/new-base**——而 8.4M → 85M 换来 0.017。曲线是平的，不是缓慢下降。
2. **不是训练不足**：0.95B 模型消耗了 85M 4 倍的 tokens 才到达*相同*的 loss；且 4 档学习率下所有 large run 都在到达 best 后回升（early-stop 由过拟合触发，而非预算截断）。
3. **与 Chinchilla 一致**：D = 340M unique tokens 时，计算最优参数量 ≈ D/20 ≈ **17M**。tiny（8.4M）在最优以下；small 超出 5×；large 超出 56×。85M 以上 scaling 变平，正是数据受限 scaling 理论的预期。

相对实测频率零模型的总提取信号：**≈ 0.0536 nats/base（约占 1.3838 的 3.9%）**——在 ~85M 参数处已耗尽。其中约一半（0.0252）超出 0–8 阶 Markov 可解释范围（见 5.6）。

### 5.2 stride-6 tokenizer 收敛至 6-mer 边际预测器

![Best config matrix](design/unified_analysis/fig3_best_config_matrix.png)

**全部 20 个 large run（4 档学习率）**中，6-mer stride-6 验证 loss 恒定在 **8.233–8.244**——跨学习率极差仅 0.010。

机制逐层分解：

1. stride=1 时相邻 6-mer token 共享 5/6 碱基，next-token 预测退化为"复制 5 个已知碱基 + 预测 1 个新碱基"——任务难度与 per-base 相同（tiny 6-mer s=1 达到 1.42，接近该下限）。
2. stride=6 拆除重叠：每个 token 含 6 个真正未知的碱基。上下文无关预测器的理论落点是 **6-mer 块边际分布**——按链式分解恰等于 0–5 阶 Markov 交叉熵之和 **8.240** nats/token（各阶实测：1.3838+1.3778+1.3755+1.3692+1.3675+1.3658）。
3. 实测 8.233–8.244 将理论落点 8.240 夹在中间（±0.007）——模型**完整学会了 token 内全部结构**（较独立碱基 ×6=8.303 低 0.06，含二核苷酸至 5 阶窗口统计），**跨 token 结构提取 ≈ 0**。

**解读**：6× 压缩买到 12,288 bp 有效上下文（2,048 tokens），而语料中确实存在跨 token 信号（per-base 模型击败 8 阶 Markov 达 0.025 nats/base，折合 ≥0.25 nats/token）——但 stride-6 对这部分信号的提取率为 **0%**：每 token 6 碱基的联合预测任务耗尽了模型容量，长程依赖从未被学到。上下文付了全款，从未被使用。（边界：Evo 以 7B 参数 / ~100× 数据证明大词表 tokenizer 在更大预算下可行；本发现限定于 ≤0.95B 参数 / 340M tokens 预算区。）

### 5.3 Tokenizer 排名是学习率的函数

![LR sensitivity](design/unified_analysis/fig1_lr_sensitivity_large.png)

large LR 扫描（C3）揭示三方交互：

- **per-base 对 LR 鲁棒**（0.95B）：4 档 LR 全落 1.3302–1.3500（极差 0.02），最优 3e-5
- **3-mer 对 LR 敏感**：1.3422（1e-5）→ 1.5198（3e-4），极差 0.18
- **6-mer 家族 4 档 LR 全部失败**（≥ 6.77）

tiny 规模上排序**恰好反转**：3e-4 对*所有* tokenizer 更优（per-base 1.322 vs 1.348）。**任何在单一固定 LR 下做的 tokenizer 对比，测的是该 LR 下的排名，不是 tokenizer 的排名**——这对一般 tokenizer 研究（包括已发表的）都是方法学警示。

### 5.4 数据干预

![Data interventions](design/unified_analysis/fig4_data_interventions.png)

- **95% ANI 去冗余（tiny @3e-4）**：per-base 1.332 → 1.322；6-mer s=1 **1.872 → 1.421** 且发散消除（唯一跑满 6000 步无崩溃的 6-mer run）。数据卫生对重叠词表 tokenizer 收益最大——但 6-mer s=2 反而*劣化*（1.536 → 1.920），该结论不能外推到所有 k-mer 配置。
- **RC 扩增**：唯一干净配对（small、6-mer s6）Δ = 0.0005 nats/base——**不可解读**，因为落点位于 stride-6 塌缩区。RC 扩增目前没有任何支持或反对的受控证据（三臂解耦实验——逐字重复 vs RC-相对-逐字重复，Muennighoff 式——已设计于 `design/rc_experiment/DESIGN.md`，尚未运行）。

### 5.5 一个未定论的优化异常，如实报告

6-mer s=1 / s=2 随规模单调劣化（s=1：1.42 → 2.25 → 6.77），梯度范数跨规模爆炸（lr 1e-5 档 max：tiny 1.8 → small 58 → large **1.9×10⁶**）。我们明确**不**将其解读为"6-mer tokenization 在大规模更差"——这是优化不稳定、LR-规模失配与预算不足的未定混合征兆，作为未完成项如实标注，而非编入叙事。

### 5.6 模型效果判定：0–8 阶 Markov 基线阶梯

在 per-base 打包流（train 339.9M / val 41.7M tokens）上实测（train 拟合 → val 评估交叉熵，带 backoff；脚本与数据见 `design/markov_baseline/`）：

| k | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CE (nats/base) | 1.3838 | 1.3778 | 1.3755 | 1.3692 | 1.3675 | 1.3658 | 1.3597 | 1.3576 | 1.3554 |

![Markov ladder](design/markov_baseline/fig5_markov_ladder.png)

三个判定：

1. **最优 per-base 模型（1.3302）低于全部 0–8 阶 Markov 基线**——较 8 阶（1.3554）低 0.0252 nats/base。0–8 阶局部统计总共只值 0.0284，模型超出 8 阶的部分与之几乎等量：**总增益 0.0536 中约 47% 来自 8 阶窗口之外的统计结构**（k > 8 未测，不区分"更高阶局部"与"真正长程"）。
2. **模型是目前全项目对任务熵率最紧的上界估计器**：熵率 ≤ 1.3302 < 8 阶 1.3554 < … < unigram 1.3838。熵率下界未知——"熵受限"判定依旧不可作。
3. **阶梯在 k=3、k=6 处跳变**（相邻增益 0.0063 / 0.0061，约 3 倍于其他阶）——密码子周期性：编码区为主的病毒基因组 3 周期核苷酸使用被干净检出。

附带：train 按 EOS 切出 13,871 条序列 = 17,339 × 0.8 ✓；val 自拟合 vs train→val 各阶差 ≤ 0.005 → 切分无分布漂移。Markov 以 forward-only train 拟合（模型训练含 RC 扩增），对 forward val 该基线只强不弱——模型的领先是保守估计。

## 6. 核心结论

1. **参数 scaling 饱和，语料是约束**。同协议下 8.4M → 85M → 0.95B 得到 1.3481 → 1.3311 → 1.3309 nats/new-base；11× 参数只降 0.0002。D = 340M 的 Chinchilla 最优参数约 17M。
2. **最优模型低于实测 0–8 阶全部 Markov 基线**（1.330 vs 阶梯 1.384→1.355），约一半增益（0.025 nats/base）超出 8 阶统计——模型学到的是真实结构，不是局部统计的记忆。
3. **纯压缩 tokenizer（6-mer s=6）收敛为上下文无关的 6-mer 边际预测器**（该预算区）：8.233–8.244 nats/token ≈ 0–5 阶 Markov 基线之和 8.240，跨 4 档 LR 恒定；对确实存在的跨 token 信号（≥0.25 nats/token）提取率为零——压缩买到的 12,288 bp 上下文，从未被使用。
4. **Tokenizer 排名依赖 LR 与规模**（0.95B 上 per-base 鲁棒 / 3-mer 敏感 / 6-mer 不稳；8.4M 上排序反转）——单一 LR 的 tokenizer 对比存在混淆。
5. **数据卫生（ANI 去冗余）主要稳定重叠词表 tokenizer**（6-mer s=1：1.872 → 1.421，发散消除）。
6. **手写 Triton FA2 达到 fused kernel 量级**：长序列较朴素 attention 快 5–36×；性能位于官方 flash 与 mem-efficient 两个后端之间；差距已定位并有归因。

## 7. 局限性（如实陈述）

- 本发布**不含下游任务评测**，结论严格限于 held-out 语言建模 loss。
- "参数饱和"的准确含义是**"该语料、近单 epoch 预算下，参数无法提取更多信号"**——不是"不可约熵"的论断。0–8 阶 Markov 基线已实测（5.6）：全部为熵率**上界**（模型自身给出最紧的 1.3302），熵率下界仍未知，"熵受限"判定不可作。
- 所有 run 消耗 **< 0.5 epoch**；硬件异构（V100/L40S/H200），部分 run 被节点抢占中断；RC 效应**无受控估计**。
- 跨 cohort 比较只报告、不解读——这是设计决定。

## 8. 仓库与可复现性

代码发布进行中，规划布局遵循 `configs/ src/ scripts/ tests/ results/`；上文引用的分析产物已在仓库中：

```
design/unified_analysis/        # 52-run 总表 + 4 张图 + 完整分析（ANALYSIS.md）
design/experiment_summary/      # R1 loss 表（20 runs）
design/large_experiment_summary/# R2 large/small/tiny 汇总（32 runs）
design/markov_baseline/         # 0–8 阶 Markov 基线（脚本 + 实测结果 + 阶梯图）
design/rc_experiment/           # 三臂 RC 干预实验设计（尚未运行）
benchmark_results/              # Triton FA2 基准套件（fp16/bf16/fp32）+ 汇总
```

本 README 中每个数字均可溯源至 `design/unified_analysis/master_experiments.csv`、`design/markov_baseline/markov_baseline_results.csv` 或 `benchmark_results/benchmark_summary.md`。

## 9. Roadmap

- [x] 语料构建与无泄漏切分（17,339 基因组）
- [x] 5 种 tokenizer 方案研究 + nats/new-base 归一化
- [x] tiny（8.4M）/ small（85M）/ large（0.95B）训练——52 runs
- [x] Triton FA2 + 系统基准（3 dtype × 4 head_dim × 5 序列长度）
- [x] 0–8 阶 Markov 基线阶梯（2026-09 实测，`design/markov_baseline/`）
- [ ] 三臂 RC 扩增实验（已设计，`design/rc_experiment/DESIGN.md`）
- [ ] Markov 阶梯扩展至 k>8（区分更高阶局部与真正长程结构）
- [ ] 代码发布与下游任务评测

## 致谢

训练栈设计遵循 [Stanford CS336（Language Modeling from Scratch）](https://stanford-cs336.github.io/) Assignment 1–2 的方法论，全部独立实现。

## 联系方式

Jiang Wei — [GitHub](https://github.com/weijiang34) · wjiang34-c@my.cityu.edu.hk
