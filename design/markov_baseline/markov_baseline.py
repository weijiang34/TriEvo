#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TriEvo 高阶 Markov 基线：实测 unigram 频率零模型 + 1–8 阶条件熵基线
====================================================================

目的
----
1. 把分析脚本里硬编码的 ZERO = 1.370（"经验值"）替换为实测值：
   unigram（1 阶）在 train→val 上的交叉熵，即严格的"频率零模型"；
2. 补齐 09-27 模拟面试指出的缺口：2–8 阶 Markov 条件熵基线，
   用于判定模型 1.33 nats/base 到底超出了几阶统计结构。

方法学定位（防止过度解读，务必读）
----------------------------------
- 所有基线均为 train 上拟合、val 上评估的交叉熵。
  任意预测器的交叉熵 CE_q = H_true + KL(p||q) >= H_true，
  因此这些基线是任务条件熵的【上界参照】，不是下界。
- 若模型 loss < Markov_k 的 CE，结论是"模型提取了超出 k 阶的结构"；
  不能反推出"熵受限/熵已耗尽"——下界需要别的证据。
- k=0 行即实测的频率零模型（对应简历中的 1.370）。
- val 自拟合熵（self_entropy_val 列）是在 val 上拟合又在 val 上评估的
  plug-in 熵，高阶时因上下文过拟合而系统性偏低，仅作参考，不要引用。

与模型 val loss 的可比性口径
----------------------------
- 模型 loss 在 EOS 拼接的打包流上计算（含对 EOS 的预测）；
  本脚本按 EOS 切分为独立序列、不预测 EOS。二者差异上界 =
  EOS 占比（脚本会打印，预计 ~1e-4 量级，可忽略）。
- token 流中可能存在 UNK 等非碱基符号：脚本保留全部非 EOS 符号
  （动态字母表），与模型实际预测的流一致。
- train 文件若已含反向互补（RC）扩增，直接使用即可——
  基线与模型见到同一分布，这正是可比性要求。

使用方法
--------
1. 填好下方 CONFIG 中标 TODO 的三项（两个路径 + EOS_ID）；
2. python markov_baseline.py            （或带参数覆盖，见 argparse）
3. 产物：终端报告 + markov_baseline_results.csv（与脚本同目录）

依赖：numpy（torch 仅在 FILE_FORMAT="pt" 时需要）
预计运行时间：CPU 数分钟到 ~20 分钟（train ~680M tokens × 8 阶）
"""

import os
import sys
import csv
import time
import argparse

import numpy as np

# ============================================================
# CONFIG —— 上服务器后填这里（三项 TODO）
# ============================================================
CONFIG = {
    # TODO(1): 训练集 uint16 打包流路径（含 RC 扩增后的实际训练文件）
    "TRAIN_PATH": r"",

    # TODO(2): 验证集 uint16 打包流路径。
    #          注意：应使用 per-base tokenizer 的打包流（模型 val loss 用的
    #          同一份文件），k-mer 流的字母表为 64/4096，高阶表会爆炸；
    #          k-mer 基线用 unigram×stride 的算术即可（见 README F2）。
    "VAL_PATH": r"",

    # TODO(3): EOS 的 token id（核对你的 tokenizer 实现；其余符号如 UNK
    #          无需配置，脚本会自动纳入动态字母表）
    "EOS_ID": 4,

    # 文件格式：auto（按扩展名推断）| npy | bin | pt | fasta
    "FILE_FORMAT": "auto",

    # 二进制格式（npy 自动读 dtype；bin/pt 需要此设置）
    "DTYPE": np.uint16,

    # 最大 Markov 阶数（k=0 即 unigram 频率零模型）
    "MAX_ORDER": 8,

    # 输出 CSV 路径
    "OUT_CSV": "markov_baseline_results.csv",

    # 模型参照数字（仅用于报告中的 gap 计算；来源：master_experiments.csv）
    "MODEL_REF": {
        "tiny_best": 1.3220,   # C1a 协议 @3e-4
        "small_best": 1.3311,  # C2 协议 @1e-5
        "large_best": 1.3302,  # 全项目 per-base 最优
        "assumed_zero": 1.370, # 此前硬编码的经验值（本次要检验的对象）
    },
}

# 计数表内存上限（S^(k+1) 个 int64），超过则自动降阶
MAX_TABLE_ENTRIES = 2**28


# ============================================================
# 数据加载
# ============================================================
def load_flat_stream(path: str, fmt: str, dtype) -> np.ndarray:
    """加载打包的 token 流，返回一维 int 数组。"""
    if not path:
        raise ValueError("TRAIN_PATH / VAL_PATH 尚未填写（CONFIG 中标 TODO 处）")
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")

    ext = os.path.splitext(path)[1].lower()
    if fmt == "auto":
        if ext in (".npy",):
            fmt = "npy"
        elif ext in (".bin", ".dat", ".uint16"):
            fmt = "bin"
        elif ext in (".pt", ".pth"):
            fmt = "pt"
        elif ext in (".fasta", ".fa", ".fna", ".fas") or path.endswith(".gz"):
            fmt = "fasta"
        else:
            raise ValueError(
                f"无法从扩展名 {ext} 推断格式，请在 CONFIG['FILE_FORMAT'] "
                f"中显式指定 npy/bin/pt/fasta"
            )

    if fmt == "npy":
        arr = np.load(path)
        return np.asarray(arr).ravel()
    if fmt == "bin":
        return np.fromfile(path, dtype=dtype)
    if fmt == "pt":
        import torch  # 延迟导入，非必需不加载
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            return obj.numpy().ravel()
        raise TypeError(f".pt 文件内容是 {type(obj)}，非 tensor；"
                        f"请检查是否为打包流文件，或改用 npy/bin 格式路径")
    if fmt == "fasta":
        ids = []
        table = {}
        cur = 0
        import gzip
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as fh:
            for line in fh:
                if line.startswith(">"):
                    continue
                for ch in line.strip():
                    if ch not in table:
                        table[ch] = cur
                        cur += 1
                    ids.append(table[ch])
        print(f"  [fasta] 检测到字母表: {sorted(table.items())}")
        return np.asarray(ids, dtype=np.int64)
    raise ValueError(f"未知格式: {fmt}")


def split_by_eos(stream: np.ndarray, eos_id: int):
    """按 EOS 切分为独立序列；返回 (序列列表, EOS 个数, 总 token 数)。"""
    flat = stream.astype(np.int64, copy=False)
    eos_pos = np.where(flat == eos_id)[0]
    n_eos = len(eos_pos)
    n_tok = len(flat) - n_eos  # 非 EOS token 总数

    seqs = []
    start = 0
    for p in eos_pos:
        if p > start:
            seqs.append(flat[start:p])
        start = p + 1
    if start < len(flat):
        seqs.append(flat[start:])
    return seqs, n_eos, n_tok


def build_alphabet(seqs):
    """把出现的 token id 映射为紧凑编码 0..S-1。"""
    ids = np.unique(np.concatenate(seqs)) if seqs else np.array([], dtype=np.int64)
    remap = {int(v): i for i, v in enumerate(ids)}
    S = len(ids)
    return remap, S, ids


def remap_seqs(seqs, remap):
    """把原始 id 序列批量映射为紧凑编码（向量化）。"""
    if not remap:
        return list(seqs)
    lookup = np.full(max(remap.keys()) + 1, -1, dtype=np.int64)
    for k, v in remap.items():
        lookup[k] = v
    return [lookup[s] for s in seqs]


# ============================================================
# 拟合与评估
# ============================================================
def fit_counts(seqs, k: int, S: int) -> np.ndarray:
    """拟合 k 阶转移计数表，形状 (S^k, S)。k=0 即 unigram。"""
    size = S ** (k + 1)
    counts = np.zeros(size, dtype=np.int64)
    for s in seqs:
        L = len(s)
        if L <= k:
            continue
        ctx = np.zeros(L - k, dtype=np.int64)
        for j in range(k):
            ctx = ctx * S + s[j: L - k + j]
        nxt = s[k:]
        counts += np.bincount(ctx * S + nxt, minlength=size)
    return counts.reshape(S ** k, S)


def eval_cross_entropy(seqs, models, kmax: int, S: int):
    """
    在 seqs 上评估带 backoff 的交叉熵（nats/token）。
    models: dict {k: (counts_k, totals_k)}，k=0..kmax。
    返回 (平均 CE, 各阶解决的 token 数数组, token 总数)。
    优先用最高阶的已见上下文；上下文未见时逐级 backoff 到低阶，
    k=0（unigram）恒可用（train 非空即覆盖全部剩余位置）。
    """
    n_tok = 0
    nll = 0.0
    resolved_at = np.zeros(kmax + 1, dtype=np.int64)
    n_oov = 0  # train 中完全未见过的 val 符号（均匀分布兜底）

    for s in seqs:
        L = len(s)
        if L == 0:
            continue
        lp = np.zeros(L, dtype=np.float64)
        unresolved = np.ones(L, dtype=bool)
        for k in range(kmax, -1, -1):
            if not unresolved.any():
                break
            counts, totals = models[k]
            if L <= k:
                continue
            ctx = np.zeros(L - k, dtype=np.int64)
            for j in range(k):
                ctx = ctx * S + s[j: L - k + j]
            nxt = s[k:]
            # 转移对级 seen：上下文已见且该 (ctx, next) 对也已见，
            # 否则 backoff 到低阶（防止 p=0 → log(0) = -inf）
            pair_seen = counts[ctx, nxt] > 0
            m = unresolved[k:] & pair_seen
            if m.any():
                c = ctx[m]
                p = counts[c, nxt[m]] / totals[c]
                lp[k:][m] = np.log(p)
                unresolved[k:] &= ~m
                resolved_at[k] += int(m.sum())
        if unresolved.any():
            # train 中未见过的符号：均匀分布兜底（正常应仅 UNK 等罕见符号）
            n_oov += int(unresolved.sum())
            lp[unresolved] = np.log(1.0 / S)
        nll += -lp.sum()
        n_tok += L

    if n_oov:
        print(f"      注意：{n_oov:,} 个 val 符号在 train 未见（均匀兜底）")
    return nll / n_tok, resolved_at, n_tok


# ============================================================
# 主流程
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="TriEvo 高阶 Markov 基线")
    ap.add_argument("--train", default=CONFIG["TRAIN_PATH"],
                    help="训练集打包流路径（覆盖 CONFIG）")
    ap.add_argument("--val", default=CONFIG["VAL_PATH"],
                    help="验证集打包流路径（覆盖 CONFIG）")
    ap.add_argument("--eos", type=int, default=CONFIG["EOS_ID"],
                    help="EOS token id（覆盖 CONFIG）")
    ap.add_argument("--format", default=CONFIG["FILE_FORMAT"])
    ap.add_argument("--max-order", type=int, default=CONFIG["MAX_ORDER"])
    ap.add_argument("--out", default=CONFIG["OUT_CSV"])
    args = ap.parse_args()

    t0 = time.time()
    ref = CONFIG["MODEL_REF"]

    print("=" * 64)
    print("TriEvo 高阶 Markov 基线 — unigram 实测 + 1–{} 阶条件熵".format(args.max_order))
    print("=" * 64)

    # ---------- 加载 ----------
    print(f"\n[1/5] 加载 train: {args.train}")
    train_stream = load_flat_stream(args.train, args.format, CONFIG["DTYPE"])
    print(f"      总 token 数 {len(train_stream):,}")
    print(f"\n      加载 val:   {args.val}")
    val_stream = load_flat_stream(args.val, args.format, CONFIG["DTYPE"])
    print(f"      总 token 数 {len(val_stream):,}")

    # ---------- 切分与字母表 ----------
    print("\n[2/5] 按 EOS 切分序列")
    train_seqs, n_eos_tr, n_tok_tr = split_by_eos(train_stream, args.eos)
    val_seqs, n_eos_va, n_tok_va = split_by_eos(val_stream, args.eos)
    print(f"      train: {len(train_seqs):,} 条序列, 非 EOS token {n_tok_tr:,}, "
          f"EOS {n_eos_tr:,} (占比 {n_eos_tr / max(len(train_stream),1):.2e})")
    print(f"      val:   {len(val_seqs):,} 条序列, 非 EOS token {n_tok_va:,}, "
          f"EOS {n_eos_va:,} (占比 {n_eos_va / max(len(val_stream),1):.2e})")

    remap, S, ids = build_alphabet(train_seqs + val_seqs)
    sym_names = [f"id={int(v)}" for v in ids]
    print(f"      动态字母表大小 S = {S} ({', '.join(sym_names)})")
    if S < 4:
        print("      警告：字母表小于 4，请检查 BASE token id 是否符合预期")

    train_seq_r = remap_seqs(train_seqs, remap)
    val_seq_r = remap_seqs(val_seqs, remap)

    # ---------- 自动降阶（内存保护） ----------
    kmax = args.max_order
    while kmax > 0 and S ** (kmax + 1) > MAX_TABLE_ENTRIES:
        kmax -= 1
    if kmax != args.max_order:
        print(f"      内存保护：MAX_ORDER 由 {args.max_order} 降至 {kmax}")

    # ---------- 拟合 ----------
    print(f"\n[3/5] 在 train 上拟合计数表（k = 0..{kmax}）")
    models_tr = {}
    for k in range(kmax + 1):
        models_tr[k] = None  # 占位
    for k in range(kmax + 1):
        c = fit_counts(train_seq_r, k, S)
        models_tr[k] = (c, c.sum(axis=1))
        print(f"      k={k}: 上下文数 {c.shape[0]:,}, "
              f"非空上下文 {(c.sum(axis=1) > 0).sum():,}")

    # unigram 频率（k=0）
    uni = models_tr[0][0].ravel()
    freq = uni / uni.sum()
    H_unigram_train2val = None  # 占位，稍后由评估填

    print(f"\n      train 上的符号频率: "
          + ", ".join(f"{n}={p:.4f}" for n, p in zip(sym_names, freq)))

    # ---------- 评估 ----------
    print(f"\n[4/5] 在 val 上评估（train→val 交叉熵 + backoff）")
    results = []
    for k in range(kmax + 1):
        ce, resolved, n = eval_cross_entropy(
            val_seq_r, {j: models_tr[j] for j in range(k + 1)}, k, S)
        resolved_frac = resolved / n
        H_unigram_train2val = ce if k == 0 else H_unigram_train2val
        results.append({
            "order": k,
            "ce_train2val_nats_per_tok": ce,
            "resolved_frac_at_order": resolved_frac.tolist(),
            "self_entropy_val": np.nan,  # 稍后填
        })
        print(f"      k={k}: CE = {ce:.4f} nats/token"
              + (f"  (backoff 比例 {1 - resolved_frac[k]:.3f})" if k > 0 else ""))

    # val 自拟合熵（参考列，plug-in 偏低，勿引用）
    print(f"\n[5/5] val 自拟合熵（仅参考，plug-in 偏低）")
    models_va = {}
    for k in range(kmax + 1):
        c = fit_counts(val_seq_r, k, S)
        models_va[k] = (c, c.sum(axis=1))
    for k in range(kmax + 1):
        ce_self, _, _ = eval_cross_entropy(
            val_seq_r, {j: models_va[j] for j in range(k + 1)}, k, S)
        results[k]["self_entropy_val"] = ce_self

    # ---------- 报告 ----------
    elapsed = time.time() - t0
    uni_ce = results[0]["ce_train2val_nats_per_tok"]
    print("\n" + "=" * 64)
    print("核心结果")
    print("=" * 64)
    print(f"\n  实测频率零模型 (unigram, train→val): {uni_ce:.4f} nats/token")
    print(f"  此前硬编码经验值:                    {ref['assumed_zero']:.4f}")
    print(f"  差异: {uni_ce - ref['assumed_zero']:+.4f} nats/token")

    print(f"\n  {'k':>2} | {'train→val CE':>13} | {'与 large_best 差':>15} | {'val 自拟合(参考)':>16}")
    for r in results:
        gap = r["ce_train2val_nats_per_tok"] - ref["large_best"]
        print(f"  {r['order']:>2} | {r['ce_train2val_nats_per_tok']:>13.4f} | "
              f"{gap:>+15.4f} | {r['self_entropy_val']:>16.4f}")

    print(f"\n  模型参照: tiny {ref['tiny_best']} / small {ref['small_best']} / "
          f"large {ref['large_best']} nats/token (per-base)")
    best_k = max((r for r in results), key=lambda r: -r["ce_train2val_nats_per_tok"])
    below = [r["order"] for r in results
             if ref["large_best"] < r["ce_train2val_nats_per_tok"]]
    if below:
        print(f"\n  解读：large 最优 loss {ref['large_best']} 低于 "
              f"0–{max(below)} 阶全部 Markov 基线 → 模型提取了超出 "
              f"{max(below)} 阶的统计结构。")
    else:
        print("\n  解读：模型 loss 不低于任何 Markov 基线 —— 需人工复核口径。")

    print("\n  注意：以上基线均为交叉熵 = 熵上界参照（CE = H + KL >= H）。")
    print("  模型低于 k 阶基线 ⇒ 超出 k 阶结构；不能据此断言『熵已耗尽』。")
    print(f"\n  EOS 占比（val）: {n_eos_va / max(len(val_stream),1):.2e}"
          f" —— 与模型打包流口径的差异上界，可忽略。" if n_eos_va else "")
    print(f"\n  运行耗时: {elapsed:.1f}s")

    # ---------- CSV ----------
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["order", "ce_train2val_nats_per_tok",
                    "self_entropy_val", "model_ref_large_best"])
        for r in results:
            w.writerow([r["order"],
                        f"{r['ce_train2val_nats_per_tok']:.6f}",
                        f"{r['self_entropy_val']:.6f}",
                        ref["large_best"]])
    print(f"\n  结果已写入: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
