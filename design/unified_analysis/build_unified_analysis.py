# -*- coding: utf-8 -*-
"""TriEvo 实验结果统一整编与系统分析
输入: 5 个 CSV (2 轮实验, 4 类协议, 52 runs)
输出: master_experiments.csv + 4 张图 + 控制台摘要
"""
import os, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"D:\WorkBuddySpace\26fall\TriEvo"
OUT = os.path.join(BASE, "design", "unified_analysis")
os.makedirs(OUT, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ZERO = 1.370  # per-base 频率零模型 (nats/base, 经验值)

STRIDE = {"perbase": 1, "kmer3": 1, "kmer6": 1, "kmer6s2": 2, "kmer6s6": 6}
VOCAB = {"perbase": 4, "kmer3": 64, "kmer6": 4096, "kmer6s2": 4096, "kmer6s6": 4096}
# 参数量 (M): 已知精确值来自代码计算; 其余按 embedding+head (untied) 差值推算
PARAMS = {
    ("tiny", "perbase"): 8.40, ("tiny", "kmer3"): 8.43, ("tiny", "kmer6"): 10.49,
    ("tiny", "kmer6s2"): 10.49, ("tiny", "kmer6s6"): 10.49,
    ("small", "perbase"): 84.97, ("small", "kmer3"): 85.07, ("small", "kmer6"): 91.25,
    ("small", "kmer6s2"): 91.25, ("small", "kmer6s6"): 91.25,
    ("large", "perbase"): 943.8, ("large", "kmer3"): 944.0, ("large", "kmer6"): 954.3,
    ("large", "kmer6s2"): 954.3, ("large", "kmer6s6"): 954.3,
}
TOK_COLOR = {"perbase": "#2166ac", "kmer3": "#92c5de", "kmer6": "#f4a582",
             "kmer6s2": "#d6604d", "kmer6s6": "#b2182b"}
TOK_LABEL = {"perbase": "per-base (s=1)", "kmer3": "3-mer (s=1)", "kmer6": "6-mer (s=1)",
             "kmer6s2": "6-mer (s=2)", "kmer6s6": "6-mer (s=6)"}
TOK_ORDER = ["perbase", "kmer3", "kmer6", "kmer6s2", "kmer6s6"]

rows = []

# ---------- Round 2: large LR sweep (rep_ani_95, rc, 事件文件均在单节点 gpu3) ----------
dfL = pd.read_csv(os.path.join(BASE, "design", "large_experiment_summary", "large_loss_summary.csv"))
for _, r in dfL.iterrows():
    lr_s = str(r["learning_rate"])
    lr = 3e-4 if "baseline" in lr_s else float(lr_s)
    tok = r["tokenizer"]
    best = r["best_val_loss"]
    rows.append(dict(
        run_id=r["experiment"], round=2, cohort="C3-large-LRsweep", model="large",
        params_M=PARAMS[("large", tok)], tokenizer=tok, stride=STRIDE[tok], lr=lr,
        batch_size=8, ctx=2048, dataset="rep95", rc=True, n_gpu="未记录(事件文件均在 gpu3 节点)",
        protocol="early-stop", status="ok" if pd.notna(best) else "no-val",
        best_val_loss=best, best_val_step=r["best_val_step"], final_step=r["final_step"],
        peak_mem_mb=r["peak_mem_mb"], tokens_per_s=r["mean_tokens_per_second"],
        grad_norm_min=r["min_grad_norm"], grad_norm_max=r["max_grad_norm"],
        source="large_loss_summary.csv"))

# ---------- Round 2: small (rep_ani_95) ----------
dfS = pd.read_csv(os.path.join(BASE, "design", "large_experiment_summary", "small_loss_summary.csv"))
for _, r in dfS.iterrows():
    tok = r["tokenizer"]; best = r["best_val_loss"]; rc = bool(r["reverse_complement"])
    lr = float(r["learning_rate"])
    cohort = "C2-scaling" if lr == 1e-5 else "C4-rcAB"
    rows.append(dict(
        run_id=r["experiment"], round=2, cohort=cohort, model="small",
        params_M=PARAMS[("small", tok)], tokenizer=tok, stride=STRIDE[tok], lr=lr,
        batch_size=int(r["batch_size"]), ctx=2048, dataset="rep95", rc=rc,
        n_gpu="未记录(事件文件均在 gpu3 节点)", protocol="early-stop", status="ok",
        best_val_loss=best, best_val_step=r["best_val_step"], final_step=r["final_step"],
        peak_mem_mb=r["peak_mem_mb"], tokens_per_s=r["mean_tokens_per_second"],
        grad_norm_min=r["min_grad_norm"], grad_norm_max=r["max_grad_norm"],
        source="small_loss_summary.csv"))

# ---------- Round 2: tiny (rep_ani_95) ----------
dfT = pd.read_csv(os.path.join(BASE, "design", "large_experiment_summary", "tiny_loss_summary.csv"))
for _, r in dfT.iterrows():
    tok = r["tokenizer"]; best = r["best_val_loss"]
    rows.append(dict(
        run_id=r["experiment"], round=2, cohort="C2-scaling", model="tiny",
        params_M=PARAMS[("tiny", tok)], tokenizer=tok, stride=STRIDE[tok], lr=float(r["learning_rate"]),
        batch_size=int(r["batch_size"]), ctx=2048, dataset="rep95", rc=bool(r["reverse_complement"]),
        n_gpu="未记录(事件文件均在 gpu3 节点)", protocol="early-stop", status="ok",
        best_val_loss=best, best_val_step=r["best_val_step"], final_step=r["final_step"],
        peak_mem_mb=r["peak_mem_mb"], tokens_per_s=r["mean_tokens_per_second"],
        grad_norm_min=r["min_grad_norm"], grad_norm_max=r["max_grad_norm"],
        source="tiny_loss_summary.csv"))

# ---------- Round 1: loss_table ----------
df1 = pd.read_csv(os.path.join(BASE, "design", "experiment_summary", "loss_table.csv"))
for _, r in df1.iterrows():
    tok = r["tokenizer"]; model = r["model"]
    ds = "full" if "full" in str(r["dataset"]) else "rep95"
    group = r["group"]
    cohort = "C1b-r1-ctx256-bs64" if "ctx256" in group else "C1a-r1-ctx2048-bs32"
    n_gpu = str(r["gpu_nodes"]).count(";") + 1
    best = r["best_val_loss"]
    rows.append(dict(
        run_id=str(r["run_dir"]).split("/")[0], round=1, cohort=cohort, model=model,
        params_M=PARAMS[(model, tok)], tokenizer=tok, stride=STRIDE[tok], lr=3e-4,
        batch_size=int(r["batch_size"]), ctx=int(r["ctx_len"]), dataset=ds, rc=False,
        n_gpu=n_gpu, protocol="fixed-6000iters", status=str(r["status"]),
        best_val_loss=best, best_val_step=r["best_val_step(early_stop)"],
        final_step=r["final_step_reached"], peak_mem_mb=np.nan, tokens_per_s=np.nan,
        grad_norm_min=np.nan, grad_norm_max=np.nan, source="loss_table.csv"))

# ---------- 派生度量 ----------
m = pd.DataFrame(rows)
m["nats_per_base"] = m["best_val_loss"] / m["stride"]
m["uniform_per_token"] = m["tokenizer"].map(lambda t: math.log(VOCAB[t]))
m["uniform_per_base"] = m["uniform_per_token"] / m["stride"]
m["gain_vs_freq_model"] = ZERO - m["nats_per_base"]
m = m.sort_values(["round", "cohort", "model", "tokenizer", "lr"]).reset_index(drop=True)
m.to_csv(os.path.join(OUT, "master_experiments.csv"), index=False, encoding="utf-8-sig")

MODELS = ["tiny", "small", "large"]
LR_ORDER = [1e-5, 3e-5, 1e-4, 3e-4]

def pick(df, **kw):
    q = df
    for k, v in kw.items():
        q = q[q[k] == v]
    return q

# ================= Fig 1: large LR 敏感性 (双面板) =================
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
axA, axB = axes
L = pick(m, cohort="C3-large-LRsweep")
for tok in ["perbase", "kmer3"]:
    sub = pick(L, tokenizer=tok).sort_values("lr")
    axA.plot(sub["lr"], sub["nats_per_base"], "o-", color=TOK_COLOR[tok],
             label=TOK_LABEL[tok], lw=1.8, ms=5)
axA.axhline(ZERO, color="#888888", ls="--", lw=1.2)
axA.text(1.05e-5, ZERO + 0.001, "频率零模型 1.370", va="bottom", fontsize=8.5, color="#555555")
axA.set_xscale("log"); axA.set_ylim(1.30, 1.55)
axA.set_xticks(LR_ORDER); axA.set_xticklabels(["1e-5", "3e-5", "1e-4", "3e-4"])
axA.set_xlabel("学习率"); axA.set_ylabel("val loss (nats / new-base)")
axA.set_title("(a) 高重叠词表 (per-base / 3-mer)", fontsize=10.5)
axA.grid(alpha=0.3); axA.legend(fontsize=8.5, loc="upper left")
for tok in ["kmer6", "kmer6s2", "kmer6s6"]:
    sub = pick(L, tokenizer=tok).sort_values("lr")
    axB.plot(sub["lr"], sub["nats_per_base"], "o-", color=TOK_COLOR[tok],
             label=TOK_LABEL[tok], lw=1.8, ms=5)
axB.axhline(ZERO, color="#888888", ls="--", lw=1.2)
axB.axhline(math.log(4096) / 6, color="#b2182b", ls=":", lw=1.2)
axB.text(1.05e-5, math.log(4096) / 6 + 0.03, "6-mer 均匀基线 1.386", fontsize=8, color="#b2182b")
axB.annotate("kmer6/kmer6s2 远高于零模型\n且对 LR 不敏感 → 优化异常", xy=(1e-4, 7.27), xytext=(3e-5, 5.2),
             fontsize=8, color="#d6604d", arrowprops=dict(arrowstyle="->", color="#d6604d", lw=0.9))
axB.set_xscale("log"); axB.set_ylim(1.2, 8.8)
axB.set_xticks(LR_ORDER); axB.set_xticklabels(["1e-5", "3e-5", "1e-4", "3e-4"])
axB.set_xlabel("学习率"); axB.set_ylabel("val loss (nats / new-base)")
axB.set_title("(b) 压缩词表 (6-mer 家族)", fontsize=10.5)
axB.grid(alpha=0.3); axB.legend(fontsize=8.5, loc="center right")
fig.suptitle("Large (≈0.95B) tokenizer × 学习率敏感性 (rep_ani_95, RC 扩增, 5 tokenizer × 4 LR)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(OUT, "fig1_lr_sensitivity_large.png"), dpi=200); plt.close(fig)

# ================= Fig 2: 受控队列 scaling =================
# C2 = tiny/small @1e-5; large 无 C2 标签, 取 C3 中同协议 (lr=1e-5) 的 run 并入
C2 = pd.concat([pick(m, cohort="C2-scaling"),
                pick(m, cohort="C3-large-LRsweep", lr=1e-5)], ignore_index=True)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
axA, axB = axes
for tok in ["perbase", "kmer3"]:
    sub = pick(C2, tokenizer=tok).set_index("model")
    xs = [PARAMS[(mm, tok)] for mm in MODELS]
    ys = [sub.loc[mm, "nats_per_base"] for mm in MODELS]
    axA.plot(xs, ys, "o-", color=TOK_COLOR[tok], label=TOK_LABEL[tok], lw=1.8, ms=5)
axA.axhline(ZERO, color="#888888", ls="--", lw=1.2)
axA.text(9.5, ZERO + 0.001, "频率零模型 1.370", fontsize=8.5, color="#555555")
axA.set_xscale("log"); axA.set_ylim(1.28, 1.40)
axA.set_xticks([10, 100, 1000]); axA.set_xticklabels(["10M", "100M", "1B"])
axA.set_xlabel("参数量"); axA.set_ylabel("val loss (nats / new-base)")
axA.set_title("(a) stride=1 tokenizer: 饱和", fontsize=10.5)
axA.grid(alpha=0.3); axA.legend(fontsize=8.5, loc="upper right")
for tok in ["kmer6", "kmer6s2", "kmer6s6"]:
    sub = pick(C2, tokenizer=tok).set_index("model")
    xs = [PARAMS[(mm, tok)] for mm in MODELS]
    ys = [sub.loc[mm, "nats_per_base"] for mm in MODELS]
    axB.plot(xs, ys, "o-", color=TOK_COLOR[tok], label=TOK_LABEL[tok], lw=1.8, ms=5)
axB.axhline(ZERO, color="#888888", ls="--", lw=1.2)
axB.axhline(math.log(4096) / 6, color="#b2182b", ls=":", lw=1.2)
axB.text(11, math.log(4096) / 6 + 0.03, "6-mer 均匀基线 1.386", fontsize=8, color="#b2182b")
axB.annotate("非单调异常\n(梯度范数 1e2–1e6)", xy=(954.3, 6.77), xytext=(60, 5.6),
             fontsize=8, color="#d6604d", arrowprops=dict(arrowstyle="->", color="#d6604d", lw=0.9))
axB.set_xscale("log"); axB.set_ylim(0, 7.6)
axB.set_xticks([10, 100, 1000]); axB.set_xticklabels(["10M", "100M", "1B"])
axB.set_xlabel("参数量"); axB.set_ylabel("val loss (nats / new-base)")
axB.set_title("(b) 压缩/重叠词表 tokenizer: 异常", fontsize=10.5)
axB.grid(alpha=0.3); axB.legend(fontsize=8.5, loc="center left")
fig.suptitle("受控队列 C2 的三点 scaling (lr=1e-5, RC, rep_ani_95, bs=8, ctx=2048)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(OUT, "fig2_scaling_controlled_cohort.png"), dpi=200); plt.close(fig)

# ================= Fig 3: best-config 矩阵 (仅 rep95) =================
rep = m[(m["dataset"] == "rep95") & m["nats_per_base"].notna()]
mat = np.full((5, 3), np.nan); lrs = {}
for i, tok in enumerate(TOK_ORDER):
    for j, mm in enumerate(MODELS):
        sub = pick(rep, tokenizer=tok, model=mm)
        if len(sub) == 0:
            continue
        k = sub["nats_per_base"].idxmin()
        mat[i, j] = sub.loc[k, "nats_per_base"]; lrs[(i, j)] = sub.loc[k, "lr"]
fig, ax = plt.subplots(figsize=(7.2, 5.2))
norm = matplotlib.colors.PowerNorm(gamma=0.3, vmin=1.31, vmax=mat.max())
im = ax.imshow(mat, cmap="viridis_r", norm=norm)
for i in range(5):
    for j in range(3):
        v = mat[i, j]
        if np.isnan(v):
            continue
        lr = lrs[(i, j)]
        lrs_txt = {3e-4: "3e-4", 1e-4: "1e-4", 3e-5: "3e-5", 1e-5: "1e-5"}[lr]
        nrm = norm(v)
        good = v < ZERO
        ax.text(j, i - 0.05, f"{v:.4f}", ha="center", va="center", fontsize=10,
                color="white" if nrm > 0.55 else ("#08306b" if good else "#333333"),
                fontweight="bold" if good else "normal")
        ax.text(j, i + 0.28, f"@{lrs_txt}", ha="center", va="center", fontsize=7.5,
                color="white" if nrm > 0.55 else "#555555")
ax.set_xticks(range(3))
ax.set_xticklabels(["tiny\n(8–10M)", "small\n(85–91M)", "large\n(≈0.95B)"], fontsize=9.5)
ax.set_yticks(range(5)); ax.set_yticklabels([TOK_LABEL[t] for t in TOK_ORDER], fontsize=9)
ax.set_title("rep_ani_95 上各 (规模×tokenizer) 的最优 val loss (nats/new-base)\n粗体=优于频率零模型 1.370; @标注为对应学习率", fontsize=10)
cb = fig.colorbar(im, ax=ax, shrink=0.85); cb.set_label("nats / new-base (非线性色标, γ=0.3)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_best_config_matrix.png"), dpi=200); plt.close(fig)

# ================= Fig 4: 数据/LR 干预 (tiny) =================
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
axA, axB = axes
x = np.arange(5); w = 0.36
full_v, rep_v = [], []
r1t = pick(m, round=1, model="tiny")
for tok in TOK_ORDER:
    full_v.append(pick(r1t, tokenizer=tok, dataset="full")["nats_per_base"].iloc[0])
    rep_v.append(pick(r1t, tokenizer=tok, dataset="rep95")["nats_per_base"].iloc[0])
axA.bar(x - w / 2, full_v, w, color="#bdbdbd", label="refseq_viral 全集 (19,575)")
axA.bar(x + w / 2, rep_v, w, color="#2166ac", label="95% ANI 去冗余 (17,339)")
axA.axhline(ZERO, color="#888888", ls="--", lw=1.1)
axA.set_xticks(x); axA.set_xticklabels([TOK_LABEL[t].replace(" (", "\n(") for t in TOK_ORDER], fontsize=8)
axA.set_ylabel("nats / new-base"); axA.set_title("(a) 去冗余干预 (tiny, 3e-4, bs=32, 无RC)", fontsize=10.5)
axA.grid(alpha=0.3, axis="y"); axA.legend(fontsize=8.5)
lr3, lr1 = [], []
r2t = pick(m, round=2, model="tiny")
for tok in TOK_ORDER:
    lr3.append(pick(r1t, tokenizer=tok, dataset="rep95")["nats_per_base"].iloc[0])
    lr1.append(pick(r2t, tokenizer=tok)["nats_per_base"].iloc[0])
axB.bar(x - w / 2, lr3, w, color="#f4a582", label="lr=3e-4 (R1, bs=32, 无RC)")
axB.bar(x + w / 2, lr1, w, color="#b2182b", label="lr=1e-5 (R2, bs=8, RC)")
axB.axhline(ZERO, color="#888888", ls="--", lw=1.1)
axB.set_xticks(x); axB.set_xticklabels([TOK_LABEL[t].replace(" (", "\n(") for t in TOK_ORDER], fontsize=8)
axB.set_ylabel("nats / new-base"); axB.set_title("(b) 学习率干预 (tiny, rep_ani_95; 注: bs/RC 亦不同)", fontsize=10.5)
axB.grid(alpha=0.3, axis="y"); axB.legend(fontsize=8.5)
fig.suptitle("tiny 规模上的两项干预对比", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(OUT, "fig4_data_interventions.png"), dpi=200); plt.close(fig)

# ================= 控制台摘要 =================
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
print("=" * 80)
print(f"总 run 数: {len(m)}  (R1={int((m['round']==1).sum())}, R2={int((m['round']==2).sum())})")
print("\n--- C2 受控队列 (1e-5, RC, rep95, bs8): nats/new-base ---")
print(C2.pivot_table(index="tokenizer", columns="model", values="nats_per_base").reindex(TOK_ORDER)[MODELS].round(4))
print("\n--- C3 large LR 扫描: nats/new-base ---")
print(L.pivot_table(index="tokenizer", columns="lr", values="nats_per_base").reindex(TOK_ORDER)[LR_ORDER].round(4))
print("\n--- 梯度范数 (min/max) 按 规模×tokenizer (C2/C3 的 1e-5 runs) ---")
g = m[m["lr"] == 1e-5].dropna(subset=["grad_norm_min"])
print(g.assign(gn=lambda d: d.apply(lambda r: f"{r['grad_norm_min']:.2g}/{r['grad_norm_max']:.2g}", axis=1))
      .pivot_table(index="tokenizer", columns="model", values="gn", aggfunc="first").reindex(TOK_ORDER)[MODELS])
print("\n--- 吞吐/显存 (R2 单节点 gpu3) ---")
print(m.dropna(subset=["tokens_per_s"]).groupby("model")[["tokens_per_s", "peak_mem_mb"]].mean().round(0))
print("\n--- best-config 矩阵 (rep95) ---")
print(pd.DataFrame(mat, index=TOK_ORDER, columns=MODELS).round(4))
print("\n--- RC A/B (small kmer6s6, 3e-4, bs32, rep95) ---")
ab = pick(m, cohort="C4-rcAB")[["run_id", "rc", "best_val_loss", "nats_per_base"]]
print(ab.to_string(index=False))
print("\n--- 未收敛 run (best_val_step == final_step, 仍在下降) ---")
nc = m[m["best_val_step"] == m["final_step"]][["run_id", "tokenizer", "lr", "best_val_step"]]
print(nc.to_string(index=False) if len(nc) else "无")
print("\n输出目录:", OUT)
