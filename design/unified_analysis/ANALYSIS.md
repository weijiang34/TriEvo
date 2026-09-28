# TriEvo 实验结果系统分析（52 runs 统一整编）

> 生成于 2026-09-27。数据源：`design/experiment_summary/loss_table.csv`（R1，20 runs）、`design/large_experiment_summary/{tiny,small,large}_loss_summary.csv`（R2，32 runs）。
> 整编脚本：`build_unified_analysis.py`；机器可读总表：`master_experiments.csv`。
> 说明：根目录 `large_loss_summary.csv` 与 `design/large_experiment_summary/large_loss_summary.csv` 为同一数据两份（前者多一列 `early_stop_step`），本分析采用后者。

---

## 0. 穿针引线的总框架

这 52 个 run 散落在 2 轮、4 类协议、5 个 tokenizer、3 个规模、4 档学习率上——**不能放进一张表里直接比大小**。系统性分析 = 四层结构：

```
第 0 层  登记表：所有 run 整编成一张 tidy 表（master_experiments.csv）
第 1 层  统一度量：nats/new-base 归一化 + 两个基线（解决"跨 tokenizer loss 不可比"）
第 2 层  受控队列：按协议切 cohort，只在队列内做比较，跨队列只报告不下结论
第 3 层  论证链：每个受控比较 → 一条可写进简历/面试的结论
```

### 第 1 层：统一度量（一切的前提）

- **nats/new-base = best_val_loss ÷ stride**（stride: perbase/3-mer/6-mer s=1 → ÷1；s=2 → ÷2；s=6 → ÷6）。跨 tokenizer 唯一合法的比较层。
- 基线一：**token 级均匀分布** ln(vocab)（perbase 1.386 / 3-mer 4.159 / 6-mer 8.318 nats）。
- 基线二：**碱基级频率零模型 1.370 nats/base**（经验值）。定义主效应量：**gain = 1.370 − nats/new-base**（正=模型从数据提取了超出碱基组成的信号）。

### 第 2 层：受控队列划分

| Cohort | 定义 | runs | 能回答的问题 |
|---|---|---|---|
| **C1a** | R1：ctx2048、bs32、lr3e-4、无RC、固定6000步 | 15 | 数据集干预（full vs rep95，仅 tiny 有配对） |
| **C1b** | R1：ctx256、bs64、lr3e-4、无RC | 5 | 上下文长度干预（仅 small） |
| **C2** | R2：lr1e-5、RC、rep95、bs8、early-stop | 15 | **三点 scaling 曲线**（唯一跨规模受控队列） |
| **C3** | R2 large：同 C2 协议，lr ∈ {1e-5,3e-5,1e-4,3e-4} | 20 | **tokenizer × LR 敏感性**（最干净的 2D 网格） |
| **C4** | R2 small kmer6s6 @3e-4：RC on/off 配对 | 2 | RC 扩增干预（唯一干净配对，但落在退化区） |

**红线：跨 cohort 的 loss 差异一律不作为 tokenizer/规模结论的证据**（bs 32→8、LR 3e-4→1e-5、固定步数→early-stop、RC on/off 同时变动，混杂无法分离）。

---

## 1. 核心发现（第 3 层论证链）

### F1 — per-base 饱和：参数不是瓶颈（→ 主线 B：数据受限 scaling）

C2 队列（同协议 1e-5/RC/rep95/bs8）per-base 三点：

| | tiny 8.4M | small 85M | large ≈0.94B |
|---|---|---|---|
| nats/new-base | 1.3481 | 1.3311 | 1.3309 |
| best step | 1020 | 1440 | 1440 |
| tokens 至 best* | 16.7M | 23.6M | 94.4M |
| epoch 占比 | 0.05 | 0.07 | 0.28 |

*按 large=4×4090 DDP（65,536 tok/step）、tiny/small 单卡（16,384 tok/step）计；tiny/small 卡数未经日志核实。

**Tokens-to-best 三重证据（回应"是否训练量不足"）**：① large 消耗 small 的 4 倍 tokens（94M vs 24M）到达 best 却只打平（1.3309 vs 1.3311）——非"没喂够"；② large 全部 4 档 LR 的 run 在 best 后 val 均回升（post-optimum），early-stop 由过拟合触发而非预算截断；③ 全项目 perbase 最优为 tiny @3e-4/bs32（1.3220，消耗 157M tokens）——0.95B 在 4 档 LR 下从未超过 8.4M 的最好成绩。

**Chinchilla 一致性重构**：D=340M unique tokens → Chinchilla 最优参数 ≈ D/20 ≈ 17M：tiny(8.4M) 在最优以下，small(85M) 超 5×，large(0.95B) 超 56×。flat scaling（small→large 零增益）恰是数据受限区的理论预期而非反常。限定：所有 run < 0.5 epoch，"参数饱和"指"该语料、近单 epoch 预算下参数不可提取"，非"不可约熵"（后者需高阶 Markov 基线）。

8.4M → 85M 提升 0.017，85M → 0.94B（**11× 参数**）仅提升 **0.0002**。相对频率零模型的总提取信号 gain ≈ 0.040 nats/base（≈1.370 的 2.9%），在 ~85M 参数处已耗尽。**结论：可提取信号被数据规模/任务熵锁死，继续 scale 参数无收益——三点曲线本身就是"数据受限"的证据。**
（附注一：tiny 在 C1a 协议下 @3e-4 曾达 1.3220，优于 C2 中的 1.3481——tiny 的最优 LR 在 3e-4 端；plateau 1.32–1.33 跨协议稳健，但 ±0.01 的协议间差异不可解读。
附注二：频率零模型是熵的**上界参照而非下界**，"数据受限"的精确判定还需高阶 Markov 基线（待补，见 09-27 模拟面试记录）；本节结论严格来说是"**参数饱和**"——这一半已经曲线本身证明。）

### F2 — stride=6 钉死在独立基线：压缩买走了全部可学信号（→ 主线 A：tokenizer 困境）

kmer6s6 在所有 20 个 large run（4 档 LR）+ small/tiny 各配置上 val loss 恒定在 **8.233–8.244**，÷6 后 = **1.372–1.374 nats/base**，与频率零模型 1.370 几乎重合，仅比自身均匀基线 8.318 低 0.08。
机制（stepwise 分解，2026-09-28 补充）：
1. **stride=1 的"免费脚手架"**：相邻 token 重叠 5/6 碱基 → next-token 任务 = "复制 5 个已知碱基 + 预测 1 个新碱基"，条件熵 ≈ 单碱基熵 ≈ 1.37 nats/token——与 per-base 任务难度相同。tiny kmer6 实测 1.42，接近该下限，证明 tiny 学会了复制技巧（induction-head 式操作）。
2. **stride=6 拆除脚手架**：相邻 token 零重叠 → next token = 6 个真正未知的碱基，条件熵下限跳升至 ≥ 6×1.370 = 8.22 nats/token。同一架构面对的每 token 任务难度 ×6。
3. **落点位置的算术**：实测 8.233–8.244 恰好等于"独立碱基频率模型的 6 次连乘"（6×1.3722 ≈ 8.233）——比均匀分布 ln(4096)=8.318 低 0.085（学到了碱基组成），比 8.22 高 ≤0.02（跨碱基结构 ≈ 0）。即模型收敛为一个**上下文无关的碱基组成预测器**：看到了上文，却只用边际频率。
4. **LR 平坦的含义**：4 档 LR 下 8.2337–8.2441 极差仅 0.010——不是"学得慢"，是"这个任务的可达信号在组成层已耗尽"。
5. **交易的另一侧**：6× 压缩买到 12,288 bp 覆盖（2048 tokens）。若语料的预测信号是长程的，这笔交换可能划算；但此处可提取信号（0.04 nats/base）几乎全部是局部的——上下文买来了却无处使用，成本全付、收益为零。
6. **结论边界**：Evo（7B 参数、~100× 数据）证明 big tokenizer 在更大预算下可行——本发现限定于 ≤0.95B 参数 / 340M tokens 预算区，"失效边界"是预算条件的边界而非普适判决。
细微修正：tiny @3e-4 曾达 1.3525（gain +0.018，粗体见 fig3），即退化区上沿仍随规模进一步塌缩（tiny +0.018 → small +0.003 → large −0.002）。

### F3 — 6-mer s=1/s=2 随规模单调劣化：未解释的优化异常（不能当 tokenizer 结论用）

rep95 数据上各规模最优配置：

| tokenizer | tiny | small | large |
|---|---|---|---|
| 6-mer s=1 | 1.4214 | 2.2528 | **6.7676** |
| 6-mer s=2 | 1.9195 | 2.2995 | 3.4131 |

tiny 能学会"移位技巧"（s=1 时下一 token 与当前共享 5/6 碱基，理论下限 ≈1.33），large 却停在 6.77——比可达下限高 5.4 nats。梯度范数随规模爆炸（1e-5 档 max：tiny 1.8 → small 58 → large **1.9×10⁶**；large kmer3 也达 2.1×10⁴）。
**判定：这是优化不稳定 / LR-规模未匹配 / 预算不足的混合征兆，不是 tokenizer 本质的判决。** 佐证：tiny kmer6 @1e-5 跑满 6000 步 best=final 仍在下降（未收敛）；large kmer6 early-stop 判定收敛于 6.77，但梯度尺度差 3 个数量级。在排除发散前，6.77 不得写进任何叙事作为"6-mer 差"的证据——反而是一个可以主动亮出的"我知道我的实验哪里没做完"的展示点。

### F4 — LR × tokenizer × 规模三方交互（方法学发现，面试可用）

large 4 档 LR 扫描（fig1）：
- per-base 对 LR **鲁棒**：1.3302–1.3500（全档极差 0.02），最优 3e-5；
- 3-mer 对 LR **敏感**：1.3422 (1e-5) → 1.5198 (3e-4)，极差 0.18；
- 6-mer 家族 4 档 LR 全部失败（≥6.77）。

而 tiny 上恰恰相反：3e-4 对所有 tokenizer 更优（perbase 1.322 vs 1.348；kmer6 1.421 vs 5.277，fig4b）。**最优 LR 随规模下移的速率依赖 tokenizer——tokenizer 排名是 LR 的函数，任何固定单一 LR 的 tokenizer 对比都有混淆。** 你 R1 的原始对比（3e-4 单点）实际测的是"3e-4 下的排名"，不是 tokenizer 排名。

### F5 — 数据干预（fig4）

- **去冗余（full → rep95，tiny @3e-4）**：perbase 1.332→1.322、kmer3 1.334→1.329（微改善）；kmer6 **1.872→1.421**（大幅改善且消除发散，唯一跑满 6000 步无崩溃的 kmer6 run）；kmer6s2 反而 1.536→1.920。结论：数据卫生对重叠词表 tokenizer 的训练稳定性收益最大，但 s=2 的劣化提示该结论不能外推到所有 k-mer 配置。
- **RC 扩增**：唯一干净配对（small kmer6s6 @3e-4）Δ=0.0005 nats/base——**不可估计**，因为落点在 F2 退化区。RC 的真实效应目前没有任何受控证据，简历与面试均不得引用"RC 扩增有效"。

---

## 2. 混杂与不可比性声明（写进任何报告前先自查）

| 混杂源 | 具体表现 | 处置 |
|---|---|---|
| 有效 batch | R1 bs=32 vs R2 bs=8（若 R2 为多卡 DDP 则有效 batch 又回到 ~32，事件文件未记录卡数，**待向集群日志核实**） | 跨轮比较仅报告不结论 |
| 学习率 | R1 全部 3e-4；R2 tiny/small 仅 1e-5、large 4 档 | F4 已量化其影响 |
| 协议 | 固定 6000 步（walltime 截断普遍未跑满）vs early-stop | 用 best_val_loss 而非 final |
| 硬件 | R1 混布 V100/L40S/H200 且多次换节点重启；R2 集中在 gpu3 节点 | loss 与硬件无关（数值等价），但 step 预算与吞吐被硬件混杂 |
| GPU 型号推断 | k171→L40S、k202→H200 为推断，k207 型号存疑 | 表中保留原始节点名 |
| 收敛性 | 3 个 run best=final 仍在下降；small_kmer3_train R1 崩溃无数据；wd=5 组从未运行 | 已在总表 status 列标记 |

吞吐参考（R2，gpu3 节点）：tiny 215k tok/s（0.9–1.6GB）→ small 100k（3.3–3.4GB）→ large 14k（21.4–21.6GB）。large 差异非吞吐造成（README 已确认同族吞吐稳定）。

---

## 3. 论证链 → 简历/面试叙事映射

| 叙事素材 | 证据 | 出处 |
|---|---|---|
| "压缩率买上下文、卖信息量"（主线 A 的核心句） | F2（s=6 退化为零模型）+ 6-mer s=1 归一化最差 | fig1b / fig2b / fig3 |
| "tokenizer 排名是 LR 的函数"（方法学贡献） | F4 三方交互 | fig1 / fig4b |
| "参数已饱和，瓶颈在数据"（主线 B 的核心句） | F1 三点曲线 | fig2a |
| "我知道自己实验的边界"（诚实加分项） | F3 异常 + RC 不可估计 + 混杂清单 | §1 F3 / §2 |

30 秒电梯版可直接引用的数字：425M 碱基、52 个受控 run、8.4M→0.94B（11×）val loss 仅降 0.0002 nats/base、s=6 tokenizer 恰好等于频率零模型 1.370。

## 4. 可选的最小补充实验（仅在时间允许时）

1. large kmer6 @3e-6 + 梯度裁剪（验证 F3 是优化问题而非本质）；
2. small perbase @3e-4 / rep95 / 无RC（补齐 C1a 的 small 行，使 R1 数据集对比跨规模成立）；
3. small perbase RC on/off 配对（当前 RC 效应完全无受控证据）。

以上均非投递前置条件（与 2026-09-22 决策一致）。
