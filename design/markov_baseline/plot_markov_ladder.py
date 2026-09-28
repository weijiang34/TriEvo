# -*- coding: utf-8 -*-
"""0–8 阶 Markov 基线阶梯图（README 5.6 配图）
输入: markov_baseline_results.csv (阶梯) + ../unified_analysis/master_experiments.csv (C2 三规模模型)
输出: fig5_markov_ladder.png
全部数值直接读自 CSV，无硬编码（uniform = ln 4 除外）。
"""
import os, math
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..", "..")
UNI = os.path.join(BASE, "design", "unified_analysis")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 数据 ----
mb = pd.read_csv(os.path.join(HERE, "markov_baseline_results.csv"))
mb = mb.sort_values("order")
k = mb["order"].to_numpy()
ce = mb["ce_train2val_nats_per_tok"].to_numpy()
model_best = float(mb["model_ref_large_best"].iloc[0])  # large 全局最优 1.3302 (C3 @3e-5)

m = pd.read_csv(os.path.join(UNI, "master_experiments.csv"))
c2 = m[(m["cohort"] == "C2-scaling") & (m["tokenizer"] == "perbase")]
val = {r["model"]: float(r["nats_per_base"]) for _, r in c2.iterrows()}
tiny_c2, small_c2 = val["tiny"], val["small"]

UNIFORM = math.log(4)  # 1.3863 nats/base

# ---- 增量（密码子周期跳变检测）----
steps = ce[:-1] - ce[1:]  # steps[i] = k=i → i+1 的增益
jump3, jump6 = float(steps[2]), float(steps[5])

# ---- 绘图 ----
fig, ax = plt.subplots(figsize=(7.2, 4.6))

ax.plot(k, ce, "o-", color="#08519c", lw=1.8, ms=5.5, label="0–8 阶 Markov（train→val）", zorder=3)
ax.axhline(UNIFORM, color="#888888", ls=":", lw=1.2, label=f"均匀分布 ln4 = {UNIFORM:.4f}")

ax.axhline(tiny_c2, color="#92c5de", ls="--", lw=1.5,
           label=f"tiny 8.4M（C2, 1e-5）{tiny_c2:.4f}")
ax.axhline(small_c2, color="#f4a582", ls="--", lw=1.5,
           label=f"small 85M（C2, 1e-5）{small_c2:.4f}")
ax.axhline(model_best, color="#b2182b", ls="--", lw=1.8,
           label=f"large 0.95B 最优（3e-5）{model_best:.4f}")

# 密码子周期跳变标注（k=3、k=6）
for kk, gain in [(3, jump3), (6, jump6)]:
    ax.axvline(kk, color="#bbbbbb", ls="--", lw=0.9, zorder=1)
    yv = float(ce[kk])
    ax.annotate(f"k={kk}: −{gain:.4f}", xy=(kk, yv), xytext=(kk + 0.15, yv + 0.0035),
                fontsize=8.5, color="#333333")
ax.text(4.5, UNIFORM - 0.006, "密码子周期跳变 (k=3, 6)", fontsize=8.5,
        color="#666666", ha="center")

# 超出 8 阶的结构信号（双箭头）
gap = float(ce[-1]) - model_best
ax.annotate("", xy=(8.35, float(ce[-1])), xytext=(8.35, model_best),
            arrowprops=dict(arrowstyle="<->", color="#b2182b", lw=1.4))
ax.text(7.2, (float(ce[-1]) + model_best) / 2,
        f"{gap:.4f} nats/base\n超出 0–8 阶的结构信号", fontsize=9, color="#b2182b",
        ha="right", va="center", fontweight="bold")

ax.set_xlim(-0.5, 9.2)
ax.set_ylim(1.322, 1.391)
ax.set_xticks(range(9))
ax.set_xlabel("Markov 阶数 k")
ax.set_ylabel("val 交叉熵 (nats/base)")
ax.set_title("0–8 阶 Markov 基线阶梯 vs. TriEvo per-base 模型\n"
             "三规模模型全部低于 8 阶基线；总增益近半来自 8 阶窗口之外", fontsize=11)
ax.grid(alpha=0.3)
ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9)

fig.tight_layout()
out = os.path.join(HERE, "fig5_markov_ladder.png")
fig.savefig(out, dpi=200)
plt.close(fig)

print(f"saved: {out}")
print(f"ladder: k=0 {ce[0]:.4f} -> k=8 {ce[-1]:.4f} | uniform {UNIFORM:.4f}")
print(f"models: tiny(C2) {tiny_c2:.4f} | small(C2) {small_c2:.4f} | large-best {model_best:.4f}")
print(f"codon jumps: k=3 -{jump3:.4f}, k=6 -{jump6:.4f} | beyond-8th-order: {gap:.4f}")
