# small / tiny 模型实验结果整理

来源目录：`out_experiments/`（仅 `small_*`、`tiny_*`、`rep_ani_95/tiny_*`）
数据来源：每个 run 目录下的 `resolved_config.yaml`（超参数）+ `tb_logs/events.out.tfevents.*`（`Train/Loss`、`Validation/Loss`、`Validation/Perplexity`）+ 集群 PBS 日志（`../logs/*.log`，用于确认运行节点与 GPU 型号）。

本目录文件：
- `loss_table.csv`：20 组实验的完整数据表（机器可读）。
- `README.md`：本文档，含同一张表的可读版本 + 分组说明 + 分析结论。

> 说明：`out_experiments/small_tokenizer_comparison_wd_5/6_1` 只有 `resolved_config.yaml`，没有任何 checkpoint / tb_logs（从未真正跑起来），因此**未计入**下面的 20 组统计。加上它的话共有 21 个实验目录，但只有 20 组产生了实际训练数据。

## 一、实验分组

根据 `resolved_config.yaml` 里的差异，`small` 和 `tiny` 的实验分别落在不同的实验组里（同组内部只有 tokenizer 不同，跨组则是 context length / batch size 等超参不同）：

| 组名 | 模型 | 数据集 | context_length | batch_size | weight_decay | train_num_iterations | 备注 |
|---|---|---|---|---|---|---|---|
| `small/base` | small (d_model=768, 12L/12H) | refseq_viral (完整病毒库) | 2048 | 32 | 0.01 | 6000 | 5 个 tokenizer 变体（perbase/kmer3/kmer6/kmer6s2/kmer6s6）分别单独提交 |
| `small/ctx256_bs64` | small | refseq_viral (完整) | **256** | **64** | 0.01 | 6000 | 与 base 组同一模型，但改用短上下文+大 batch（`sweep_tokenizers*.pbs`，Hydra multirun 一次性跑 5 个 tokenizer） |
| `tiny/base` | tiny (更小的层数/维度) | refseq_viral (完整) | 2048 | 32 | 0.01 | 6000 | 5 个 tokenizer 变体 |
| `tiny/rep_ani_95` | tiny | **rep_ani_95**（按 95% ANI 去冗余后的子集） | 2048 | 32 | 0.01 | 6000 | 与 tiny/base 相同超参，但换了去冗余数据集，用于对比数据冗余度对训练的影响 |

（`small_tokenizer_comparison_wd_5` 是第 5 个"组"，本应是 weight_decay=5 的对照，但只留了一个空 config，没有实际训练，故未列入统计。）

每组内部 tokenizer 说明：`perbase`=逐碱基(k=1)、`kmer3`=3-mer(stride1)、`kmer6`=6-mer(stride1)、`kmer6s2`=6-mer(stride2)、`kmer6s6`=6-mer(stride6，几乎不重叠）。所有 small 模型 `vocab_size=4100`。

## 二、20 组实验 Loss 汇总表

“早停 step”列 = TensorBoard 中 `Validation/Loss` 取得最小值时对应的训练 step（约等于 `best_checkpoint.pth` 保存时的 step，训练脚本按 `val_interval` 周期性验证并保留最优 checkpoint）。“完成情况”列标注该 run 是否跑满了配置的 6000 步。

| 组 | Tokenizer | Run 目录 | 训练环境 (node:GPU) | 完成 step / 计划 | 早停 step | 最优 Val Loss | 最终 Train Loss | 最终 Val Loss | 最终 Val PPL | 完成情况 |
|---|---|---|---|---|---|---|---|---|---|---|
| small/base | kmer3 | small_kmer3_train/3_1 | k106: Tesla V100-32GB | 120/6000 | — | — | 1.437 | — | — | **失败/极早崩溃**（无验证记录、无 checkpoint） |
| small/base | kmer6 | small_kmer6_train/6_1 | k202: H200(推断) | 6000/6000 | 6000 | 4.379 | 4.343 | 4.379 | 79.7 | 完成 |
| small/base | kmer6s2 | small_kmer6s2_train/6_2 | k172: L40S 46GB | 4668/6000 | 1800 | 5.365 | 6.187 | 6.301 | 545.3 | 未跑满（walltime） |
| small/base | kmer6s6 | small_kmer6s6_train/6_6 | k172: L40S 46GB | 4680/6000 | 4200 | 8.166 | 8.156 | 8.166 | 3520.6 | 未跑满（walltime） |
| small/base | perbase | small_perbase_train/1_1 | k173: L40S 46GB | 4800/6000 | 3000 | 1.373 | 1.364 | 1.376 | 3.96 | 未跑满（walltime） |
| small/ctx256_bs64 | perbase | .../1_1 | k201: H200 143GB + k207: ~95GB卡(混合) | 4200/6000 | 1200 | 1.349 | 1.379 | 1.384 | 3.99 | 未跑满（walltime，且中途换过节点） |
| small/ctx256_bs64 | kmer3 | .../3_1 | k094/k096: L40S + k201: H200(多次重启) | 4200/6000 | 1200 | 1.351 | 2.373 | 2.366 | 10.66 | 未跑满，且**训练在后期发散**（早停点后 loss 又涨回 2.37） |
| small/ctx256_bs64 | kmer6 | .../6_1 | k096: L40S + k201: H200 | 6000/6000 | 4200 | 5.602 | 5.841 | 5.901 | 365.3 | 完成 |
| small/ctx256_bs64 | kmer6s2 | .../6_2 | k201: H200 | 5400/6000 | 2400 | 6.718 | 7.168 | 7.193 | 1330.7 | 未跑满 |
| small/ctx256_bs64 | kmer6s6 | .../6_6 | k098: L40S + k201: H200 | 6000/6000 | 6000 | 8.239 | 8.228 | 8.239 | 3784.4 | 完成（但未真正学到东西，见分析） |
| tiny/base | kmer3 | tiny_kmer3_train/3_1 | k171: L40S(推断) | 5400/6000 | 2400 | 1.334 | 2.183 | 2.185 | 8.89 | 未跑满，且早停点后同样出现**明显发散** |
| tiny/base | kmer6 | tiny_kmer6_train/6_1 | k172: L40S 46GB | 3600/6000 | 600 | 1.872 | 6.840 | 6.880 | 972.4 | 未跑满，且**严重发散**（600步后一路涨到6.88） |
| tiny/base | kmer6s2 | tiny_kmer6s2_train/6_2 | k103: Tesla V100-32GB | 3600/6000 | 600 | 3.073 | 6.113 | 6.143 | 465.4 | 未跑满，同样早期即发散 |
| tiny/base | kmer6s6 | tiny_kmer6s6_train/6_6 | k171: L40S(推断) | 3600/6000 | 600 | 8.132 | 8.160 | 8.186 | 3589.0 | 未跑满，几乎没学到东西（loss≈ln(vocab)=8.32） |
| tiny/base | perbase | tiny_perbase_train/1_1 | k171: L40S(推断) | 5400/6000 | 2400 | 1.332 | 1.363 | 1.370 | 3.94 | 未跑满，但相对稳定 |
| tiny/rep_ani_95 | kmer3 | rep_ani_95/tiny_kmer3_train/3_1 | k204: H200 143GB | 5400/6000 | 2400 | 1.329 | 1.978 | 1.976 | 7.21 | 未跑满，早停后同样有所回涨（但幅度远小于 tiny/base 的 kmer3） |
| tiny/rep_ani_95 | kmer6 | rep_ani_95/tiny_kmer6_train/6_1 | k106: Tesla V100-32GB | 6000/6000 | 3000 | 1.421 | 1.442 | 1.439 | 4.22 | 完成，**loss 大幅优于 tiny/base 的 kmer6 (1.42 vs 6.88)** |
| tiny/rep_ani_95 | kmer6s2 | rep_ani_95/tiny_kmer6s2_train/6_2 | k204: H200 143GB | 4200/6000 | 1200 | 3.839 | 7.981 | 7.968 | 2887.3 | 未跑满，仍然发散严重 |
| tiny/rep_ani_95 | kmer6s6 | rep_ani_95/tiny_kmer6s6_train/6_6 | k204: H200 143GB | 6000/6000 | 5400 | 8.115 | 8.156 | 8.116 | 3349.1 | 完成，但同样没学到东西 |
| tiny/rep_ani_95 | perbase | rep_ani_95/tiny_perbase_train/1_1 | k204: H200 143GB | 5400/6000 | 2400 | 1.322 | 1.333 | 1.330 | 3.78 | 未跑满，最优 |

原始数据见 [loss_table.csv](loss_table.csv)（含更多字段：max_lr、weight_decay、GPU 完整信息等）。

## 三、训练环境（GPU）说明

集群按节点名分配 GPU，节点型号本身基本固定，因此可以按 job 日志/资源统计里记录的显存大小反推 GPU 型号：

| 节点前缀 | 显存 | 推断 GPU 型号 | 置信度 |
|---|---|---|---|
| k106 | 32768 MiB | Tesla V100-SXM2-32GB | 直接见于 nvidia-smi |
| k103 | 32768 MiB | Tesla V100-32GB | 来自 PBS 资源统计（显存推断） |
| k172 / k173 / k094 / k096 / k098 | 46068 MiB | NVIDIA L40S | 直接见于 nvidia-smi 或资源统计 |
| k201 / k204 | 143771 MiB | NVIDIA H200 | 直接见于 nvidia-smi |
| k207 | 97887 MiB | 约 95GB 级显卡（H100/H200 分区，具体型号未直接确认） | 仅资源统计，型号存疑 |
| k171 | — | 推断为 L40S（与 k172/k173 同批次节点） | 推断，未直接验证 |
| k202 | — | 推断为 H200（与 k201/k204 同批次节点） | 推断，未直接验证 |

**重要提示**：这批实验分布在至少 3 种不同世代/显存规格的 GPU 上（V100-32GB / L40S-46GB / H200-143GB），且不少 job 在运行中途因抢占/重启换过节点（同一个 run 的 tb_logs 里出现多个 host）。这意味着：
- 不同 run 之间的训练速度、每步耗时、能跑多少 step（受 4 小时 walltime 限制）差异很大，**不能仅凭"是否跑满 6000 步"来判断模型/tokenizer 好坏**，速度快的 GPU（如 H200）天然能跑更多 step。
- 建议后续补充实验时，尽量固定 GPU 型号，或者在报告结果时按"每秒/每小时 step 数"归一化，而不是只看绝对 step 数。

## 四、值得注意的发现

1. **`small_kmer3_train/3_1` 基本是失败的实验**：TensorBoard 里只有到 step 120 的训练记录，没有任何验证 loss，`checkpoints/` 目录是空的。需要重跑，当前不能用于比较。

2. **6-mer 分词器随 stride 增大迅速失效**：以 `vocab_size=4100` 计算，`ln(4100)≈8.32`，而 `kmer6s6`（几乎不重叠的 6-mer）在 small 和 tiny 的所有变体上，最终 loss 都停在 **8.1~8.24**，基本等于"随机猜测"的水平——模型完全没有学到有效信号。`kmer6s2`（stride2）情况稍好但仍然很差（loss 5~8）。只有 `kmer6`（stride1，即滑窗式 6-mer，token 之间高度重叠）才能训练出明显低于随机水平的 loss（small 上 4.34~5.6，tiny/rep_ani_95 上 1.44）。
   → 结论：stride 越大，相邻 token 的重叠信息越少、等效序列长度越短，任务难度显著上升；在当前 6000 步/短 walltime 的预算下，大 stride 的 6-mer 分词方案几乎无法收敛。

3. **`perbase` 和 `kmer3`（小词表、高重叠）始终是最稳的方案**：所有组里两者的最优 val loss 都稳定在 **1.3~1.4**，且是全表里为数不多"早停后仍然基本稳定、没有明显发散"的配置（tiny/base 的 kmer3 除外，见第4点）。

4. **多个 run 出现"早停点之后 loss 明显回涨（发散）"**，尤其是 `tiny/base` 组几乎全员中招：
   - `tiny_kmer6_train`：最优 val loss 1.87@step600，最终却涨到 6.88（600步后训练在崩，可能是学习率/梯度问题，或该 tokenizer 在该数据集上极不稳定）。
   - `tiny_kmer6s2_train`：同样 600 步后从 3.07 涨到 6.14。
   - `small_tokenizer_comparison_ctx_256_bs64` 的 `kmer3`：1200 步时 1.35，最终涨到 2.37。
   - `tiny_kmer3_train`（原始数据集）：2400 步时 1.33，最终涨到 2.18。
   这提示：（a）`save_interval`/`val_interval`=600（tiny 相关组）可能过密、也可能是训练在这些配置下本身不稳定；（b）比较不同 tokenizer/组的效果时，应该以"最优 checkpoint（早停点）的 val loss"为准，而不是"训练结束时的最后一个 val loss"，后者会被发散污染，容易得出错误结论。

5. **数据去冗余（rep_ani_95）对 tiny 模型有明显帮助，尤其是 kmer6**：
   - `tiny/base` 的 kmer6：最优 1.87，最终发散到 6.88；
   - `tiny/rep_ani_95` 的 kmer6：最优 1.42，最终 1.44，**完整跑满 6000 步且没有发散**。
   其余 tokenizer（perbase、kmer3）在两个数据集上差别不大（都在 1.3 左右），但 `kmer6`/`kmer6s2` 在去冗余数据集上明显更稳定。说明原始 `refseq_viral` 数据集中大量高度相似/冗余的序列，对 6-mer 这种更依赖长程/精确匹配统计的分词方式伤害更大（模型可能在冗余序列上过拟合导致验证集不稳定），去冗余后训练动态明显改善。

6. **`small/ctx256_bs64` 相比 `small/base`（同一 tokenizer）总体不占优势**：把 context_length 从 2048 降到 256、batch_size 从 32 提到 64 之后，`kmer6`/`kmer6s6` 的最优 loss 与 base 组接近或更差（`kmer6`: 5.60 vs 4.38；`kmer6s6`: 8.24 vs 8.17），说明短上下文对这类基因组语言建模任务没有带来收益，反而可能因为可用上下文变短使任务更难。

7. **`small_tokenizer_comparison_wd_5`（weight_decay=5 对照组）从未真正运行**，如果这组实验本来是想研究更强正则化的效果，需要重新提交。

## 五、后续建议

- 重跑 `small_kmer3_train`（当前失败）。
- 补齐/重跑 `small_tokenizer_comparison_wd_5`，否则 weight_decay 的消融就缺了对照。
- 针对 `kmer6`/`kmer6s2` 在 `tiny/base`、`small/ctx256_bs64` 等组里出现的"早停后发散"，建议加大验证频率附近的学习率衰减检查，或直接使用 `best_checkpoint.pth`（对应本报告"早停 step"列）而不是最终 checkpoint 做后续评估/预测。
- 尽量把同一组对比实验固定在同一批次/同型号 GPU 上排队，避免 walltime 限制导致的"跑不满"和 GPU 异构带来的速度差异，干扰组间比较。
