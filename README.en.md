# TriEvo: A DNA Language Model Pretrained from Scratch on Viral Genomes

[简体中文（主版）](README.md) | English

> An independent, from-scratch pretraining study: corpus construction, a 5-scheme tokenizer study, model implementation, and a **52-run controlled experiment campaign** across three model scales (8.4M → 0.95B) — built following the methodology of Stanford CS336 (Assignments 1–2).

**One-line summary**: On a 425 Mbp viral genome corpus, parameter scaling saturates at ~85M parameters (an 11× scale-up improves val loss by only 0.0002 nats/new-base), and the 6×-compression 6-mer tokenizer collapses exactly onto a base-frequency null model — evidence that the extractable signal in this corpus is locked by data scale, not model capacity.

---

## 1. What & Why (Motivation)

Genome foundation models (DNABERT-2, Nucleotide Transformer, HyenaDNA, Evo) all face two design questions that are usually answered by intuition rather than controlled experiments:

1. **How should DNA be tokenized?** Per-base (no compression, highest resolution), k-mer (fixed compression, larger vocab), or learned BPE? Compression buys context length — but what does it *cost* in learnable signal?
2. **Does scaling parameters help at small-corpus scale?** Viral genomes are abundant (RefSeq holds ~20k complete genomes) but tiny compared to web text. Where is the parameter–data balance point?

TriEvo answers both with a leakage-free corpus and a controlled, cohort-structured experiment campaign — deliberately small enough to be run and understood by one person, but rigorous enough that every claim traces to a specific run.

## 2. Data

| Stage | Detail |
|---|---|
| Source | NCBI RefSeq viral genomes: **19,575** complete genomes |
| Deduplication | BLAST all-vs-all, **95% ANI** clustering → **17,339** representative genomes |
| Corpus size | ~425 Mbp; **339,868,143** training tokens (per-base tokenization) |
| Split | **Genome-level** 8:1:1 train/val/test — clustering *before* splitting, so no homolog of a test genome appears in training |
| Augmentation | Reverse-complement (RC) expansion (used in the large-scale runs) |
| Storage | uint16 pre-tokenized arrays; EOS concatenation + random-start fixed-length window sampling; context length 2,048 (4,096 also evaluated) |

## 3. Method & Tech Stack

**Tokenizer study** — 5 schemes under identical model configurations:

| Scheme | Vocab | Stride | Compression |
|---|---:|---|---|
| per-base | 8 | 1 | none (1 token / base) |
| 3-mer | 68 | 1 | none (1 token / base) |
| 6-mer | 4,100 | 1 / 2 / 6 | 1× / 2× / 6× |

**Architecture** — decoder-only Transformer, implemented from scratch (no framework-generated model code):

- RoPE positional encoding; RMSNorm (pre-norm); SwiGLU FFN (3 matrices)
- No bias terms; untied embeddings; standard multi-head attention
- Context length 2,048

| Config | d_model | d_ff | Layers | Heads | Params |
|---|---:|---:|---:|---:|---:|
| tiny | 256 | 1,024 | 8 | 8 | 8.4M |
| small | 768 | 2,048 | 12 | 12 | 85M |
| large | 1,280 | 5,120 | 36 | 20 | ≈0.95B |

**Hand-written training stack**:

- AdamW, cosine LR schedule, gradient clipping, mixed precision; multi-GPU DDP
- **Triton FlashAttention-2 (forward + backward)**, fp32 accumulation, validated against the reference implementation (numerical error < 1e-2, fp16 standard tolerance)
- Trained on a heterogeneous cluster (V100-32GB / L40S-46GB / H200-143GB); large runs on 4×GPU DDP; throughput 215k / 100k / 14k tokens/s at tiny / small / large scale

**Triton FA2 system benchmark** (single RTX 4090, torch 2.3.1; 3 dtypes × 4 head_dims × 5 seq lens; PyTorch SDPA backends forced and verified via profiler):

- vs naive attention (bf16): **5–36× faster at long sequences** (16k), speedup growing with sequence length — consistent with IO-complexity theory
- vs SDPA mem-efficient backend: consistently faster for d ≤ 32 (+1.1–2.7×) and at d=128 long sequences (+1.5×)
- vs official FlashAttention-2 kernel: **1.1–1.2× faster at d∈{16,64}, seq=4096; 1.3–2.8× slower at d≥32 long sequences**, with the gap attributed to pipeline depth (num_stages) and tiling — a known, bounded optimization gap
- fp16 and bf16 measurements agree pointwise within ±2% (internal reproducibility check)
- Resolved a d=128 shared-memory overflow by hand-computing the SMEM budget (96/99 KB) and pruning autotune configs by head_dim

## 4. Experimental Protocol

**52 runs with training data**, organized into cohorts so that comparisons are only made *within* a controlled protocol:

| Cohort | Protocol | Runs | Question answered |
|---|---|---:|---|
| C1a | ctx2048, bs32, lr 3e-4, no RC, fixed 6000 steps | 15 | Dataset intervention (full vs 95%-ANI-reduced) |
| C1b | ctx256, bs64 | 5 | Context-length intervention |
| C2 | lr 1e-5, RC, reduced data, bs8, early-stop | 15 | **Three-point scaling curve** (the only cross-scale controlled cohort) |
| C3 | large only: 5 tokenizers × 4 LRs | 20 | **Tokenizer × LR sensitivity grid** |
| C4 | RC on/off paired (small, 6-mer s6) | 2 | RC augmentation intervention |

**Unified metric**: per-token losses are not comparable across tokenizers with different strides/vocabularies. All cross-tokenizer comparisons use **nats/new-base = best_val_loss ÷ stride** (≈ bits-per-base ÷ ln 2), against two baselines:

- Uniform-distribution baseline: ln(vocab) per token (1.386 / 4.159 / 8.318 nats)
- **Base-frequency null model: 1.370 nats/base** (empirical unigram entropy). *Gain* = 1.370 − nats/new-base — the signal extracted beyond base composition alone.

**Red line**: cross-cohort loss differences are reported but never used as evidence (batch size, LR, stopping rule, and RC change simultaneously across cohorts).

## 5. Results

*Figures below use Chinese axis labels; the numbers are as reported here.*

### 5.1 Parameter scaling saturates — the bottleneck is data, not model size

![Scaling](design/unified_analysis/fig2_scaling_controlled_cohort.png)

Within the controlled C2 cohort (identical protocol across scales), per-base validation loss:

| | tiny 8.4M | small 85M | large ≈0.95B |
|---|---:|---:|---:|
| nats/new-base | 1.3481 | **1.3311** | **1.3309** |
| tokens consumed to best | 16.7M | 23.6M | 94.4M |
| epoch fraction | 0.05 | 0.07 | 0.28 |

Three mutually reinforcing pieces of evidence:

1. **11× parameters (85M → 0.95B) bought 0.0002 nats/new-base** — while 8.4M → 85M bought 0.017. The curve is flat, not slowly declining.
2. **It is not undertraining**: the 0.95B model consumed 4× more tokens than 85M to reach *the same* loss, and all large runs across 4 learning rates hit their best point and then rose (early-stop triggered by overfitting, not budget truncation).
3. **Chinchilla consistency**: with D = 340M unique tokens, the compute-optimal parameter count ≈ D/20 ≈ **17M**. tiny (8.4M) sits below optimum; small exceeds it 5×; large exceeds it 56×. Flat scaling above 85M is exactly what data-constrained scaling theory predicts.

Total signal extracted over the frequency null model: **≈ 0.040 nats/base (~2.9% of 1.370)** — exhausted by ~85M parameters.

### 5.2 The stride-6 tokenizer collapses exactly onto the null model

![Best config matrix](design/unified_analysis/fig3_best_config_matrix.png)

Across **all 20 large runs (4 learning rates)**, 6-mer stride-6 validation loss stays within **8.233–8.244** — an extreme range of only 0.010 across learning rates. Divided by stride 6: **1.372–1.374 nats/base, vs the 1.370 frequency null model**.

The mechanism, step by step:

1. At stride 1, adjacent 6-mer tokens share 5/6 bases, so next-token prediction degenerates to "copy 5 known bases + predict 1 new base" — the same difficulty as per-base (tiny 6-mer s=1 reaches 1.42, near this floor).
2. Stride 6 removes the overlap: each token now contains 6 genuinely unknown bases, lifting the per-token conditional-entropy floor to ≥ 6 × 1.370 = 8.22.
3. The measured 8.233 ≈ 6 × 1.3722 — the model learned *base composition only* (0.085 below uniform ln(4096) = 8.318) and **no cross-base structure**: it saw the context but used marginal frequencies.

**Interpretation**: the 6× compression bought 12,288 bp of effective context (2,048 tokens), but the extractable signal (~0.04 nats/base) is almost entirely local — the context was paid for and never used. (Boundary: Evo at 7B params / ~100× data shows big tokenizers can work at larger budgets; this finding is scoped to ≤0.95B params / 340M tokens.)

### 5.3 Tokenizer rankings are a function of learning rate

![LR sensitivity](design/unified_analysis/fig1_lr_sensitivity_large.png)

The large-scale LR sweep (C3) reveals a three-way interaction:

- **per-base is LR-robust** at 0.95B: 1.3302–1.3500 across all 4 LRs (range 0.02), best at 3e-5
- **3-mer is LR-sensitive**: 1.3422 (1e-5) → 1.5198 (3e-4), range 0.18
- **6-mer family fails at all 4 LRs** (≥ 6.77)

At tiny scale the ordering **reverses**: 3e-4 is better for *every* tokenizer (per-base 1.322 vs 1.348). **Any tokenizer comparison run at a single fixed LR measures that LR's ranking, not the tokenizer's** — a methodological caveat for tokenizer studies generally, including published ones.

### 5.4 Data interventions

![Data interventions](design/unified_analysis/fig4_data_interventions.png)

- **95% ANI deduplication (tiny @3e-4)**: per-base 1.332 → 1.322; 6-mer s=1 **1.872 → 1.421** with divergence eliminated (the only 6-mer run to complete all 6000 steps without collapsing). Data hygiene helps overlapping-vocabulary tokenizers most — but 6-mer s=2 *worsened* (1.536 → 1.920), so the effect does not generalize to all k-mer configs.
- **RC augmentation**: the only clean paired comparison (small, 6-mer s6) shows Δ = 0.0005 nats/base — **not interpretable**, because the landing point sits in the stride-6 collapsed regime. We report no controlled evidence for or against RC augmentation (a three-arm decoupled experiment — verbatim-repeat vs RC-vs-verbatim, Muennighoff-style — is designed in `design/rc_experiment/DESIGN.md` but not yet run).

### 5.5 An unresolved optimization anomaly, reported as such

6-mer s=1 and s=2 degrade monotonically with scale (s=1: 1.42 → 2.25 → 6.77), with gradient norms exploding across scales (max grad norm at lr 1e-5: tiny 1.8 → small 58 → large **1.9×10⁶**). We explicitly do **not** interpret this as "6-mer tokenization is worse at scale" — it is an unresolved mix of optimization instability, LR-scale mismatch, and budget shortage, and we flag it as an open item rather than fitting it into a narrative.

## 6. Core Conclusions

1. **Parameter scaling saturates; the corpus is the constraint.** 8.4M → 85M → 0.95B yields 1.3481 → 1.3311 → 1.3309 nats/new-base under identical protocol; 11× scale-up buys 0.0002. Chinchilla-optimal for D = 340M is ~17M params.
2. **Compression-only tokenizers (6-mer s=6) converge to a context-independent base-frequency predictor** at this budget: 8.233–8.244 nats/token ≈ 6 × 1.372 across 4 LRs — compression bought 12,288 bp of context that the local-only signal could not use.
3. **Tokenizer rankings are LR-dependent and scale-dependent** (per-base robust / 3-mer sensitive / 6-mer unstable at 0.95B; reversed ordering at 8.4M) — single-LR tokenizer comparisons are confounded.
4. **Data hygiene (ANI deduplication) mainly stabilizes overlapping-vocabulary tokenizers** (6-mer s=1: 1.872 → 1.421, divergence eliminated).
5. **A hand-written Triton FA2 reaches fused-kernel territory**: 5–36× over naive attention at long sequences; between the official flash and mem-efficient backends; gaps localized and attributed.

## 7. Limitations (stated plainly)

- **No downstream-task evaluation** is included in this release; conclusions are restricted to held-out language-modeling loss.
- The "parameter saturation" claim is strictly **"parameters cannot extract more signal from this corpus under a near-single-epoch budget"** — it is not a claim about irreducible entropy; a higher-order Markov baseline is needed to bound the task entropy precisely (the frequency null model is an upper reference, not a lower bound).
- All runs consumed **< 0.5 epoch**; hardware was heterogeneous (V100/L40S/H200) and some runs were interrupted by node preemption; the RC effect has **no controlled estimate**.
- Cross-cohort comparisons are reported but not interpreted, by design.

## 8. Repository & Reproducibility

Code release is in progress. The planned layout follows `configs/ src/ scripts/ tests/ results/`; the analysis artifacts referenced above are already in-repo:

```
design/unified_analysis/        # 52-run master table + 4 figures + full analysis (ANALYSIS.md)
design/experiment_summary/      # R1 loss tables (20 runs)
design/large_experiment_summary/# R2 large/small/tiny summaries (32 runs)
design/rc_experiment/           # three-arm RC intervention design (not yet run)
benchmark_results/              # Triton FA2 benchmark suite (fp16/bf16/fp32) + summary
```

Every number in this README traces to `design/unified_analysis/master_experiments.csv` or `benchmark_results/benchmark_summary.md`.

## 9. Roadmap

- [x] Corpus construction & leakage-free split (17,339 genomes)
- [x] 5-scheme tokenizer study + nats/new-base normalization
- [x] tiny (8.4M) / small (85M) / large (0.95B) training — 52 runs
- [x] Triton FA2 + system benchmark (3 dtypes × 4 head_dims × 5 seq lens)
- [ ] Three-arm RC augmentation experiment (designed, `design/rc_experiment/DESIGN.md`)
- [ ] Higher-order Markov baseline (entropy lower bound)
- [ ] Code release & downstream-task evaluation

## Acknowledgments

The training-stack design follows the methodology of [Stanford CS336 (Language Modeling from Scratch)](https://stanford-cs336.github.io/) Assignments 1–2, implemented independently.

## Contact

Jiang Wei — [GitHub](https://github.com/weijiang34) · wjiang34-c@my.cityu.edu.hk
