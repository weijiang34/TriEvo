# Large model experiment summary

Generated from `out_experiments/rep_ani_95/large_*` by `summarize_large.py`.
The primary comparison metric is the minimum recorded `Validation/Loss`; `final_val_loss` is the last validation point in the selected event log.

## 20-group loss table

| Rank | Experiment | Tokenizer | Max LR | Best val loss | Best step | Final val loss | Final train loss | Final PPL | Stop step | Checkpoints | Event runs |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `large_perbase_3e-5_rc_train` | `perbase` | 3e-5 | 1.3302 | 960 | 1.3370 | 1.3456 | 3.8076 | 1260 | 2 | 1 |
| 2 | `large_perbase_1e-5_rc_train` | `perbase` | 1e-5 | 1.3309 | 1440 | 1.3330 | 1.3644 | 3.7923 | 1740 | 2 | 1 |
| 3 | `large_perbase_1e-4_rc_train` | `perbase` | 1e-4 | 1.3309 | 900 | 1.3326 | 1.3186 | 3.7908 | 1200 | 1 | 1 |
| 4 | `large_kmer3_1e-5_rc_train` | `kmer3` | 1e-5 | 1.3422 | 3360 | 1.3461 | 1.3440 | 3.8423 | 3660 | 6 | 1 |
| 5 | `large_kmer3_3e-5_rc_train` | `kmer3` | 3e-5 | 1.3461 | 1620 | 1.3546 | 1.3601 | 3.8751 | 1920 | 3 | 1 |
| 6 | `large_perbase_rc_train` | `perbase` | 0.0003 (baseline) | 1.3500 | 300 | 1.3734 | 1.3934 | 3.9488 | 600 | 0 | 1 |
| 7 | `large_kmer3_1e-4_rc_train` | `kmer3` | 1e-4 | 1.4119 | 540 | 2.9270 | 3.0260 | 18.6716 | 840 | 1 | 1 |
| 8 | `large_kmer3_rc_train` | `kmer3` | 0.0003 (baseline) | 1.5198 | 180 | 4.2697 | 5.8719 | 71.4982 | 480 | 0 | 1 |
| 9 | `large_kmer6_1e-5_rc_train` | `kmer6` | 1e-5 | 6.7676 | 4080 | 6.7934 | 6.6726 | 891.9200 | 4380 | 7 | 1 |
| 10 | `large_kmer6s2_3e-5_rc_train` | `kmer6s2` | 3e-5 | 6.8262 | 2820 | 7.3985 | 7.2038 | 1633.5237 | 3120 | 5 | 1 |
| 11 | `large_kmer6_1e-4_rc_train` | `kmer6` | 1e-4 | 7.2666 | 840 | 7.7405 | 7.8433 | 2299.6812 | 1140 | 1 | 1 |
| 12 | `large_kmer6_3e-5_rc_train` | `kmer6` | 3e-5 | 7.2754 | 900 | 7.3526 | 7.2886 | 1560.3046 | 1200 | 1 | 1 |
| 13 | `large_kmer6_rc_train` | `kmer6` | 0.0003 (baseline) | 7.5918 | 240 | 7.9934 | 8.0212 | 2961.2217 | 540 | 0 | 1 |
| 14 | `large_kmer6s2_1e-4_rc_train` | `kmer6s2` | 1e-4 | 7.6621 | 1260 | 8.2188 | 8.2933 | 3709.9541 | 1560 | 2 | 1 |
| 15 | `large_kmer6s2_rc_train` | `kmer6s2` | 0.0003 (baseline) | 7.8755 | 480 | 8.1371 | 8.1388 | 3419.0806 | 780 | 1 | 1 |
| 16 | `large_kmer6s2_1e-5_rc_train` | `kmer6s2` | 1e-5 | 8.0576 | 3000 | 8.0759 | 8.1249 | 3216.1802 | 3300 | 5 | 1 |
| 17 | `large_kmer6s6_rc_train` | `kmer6s6` | 0.0003 (baseline) | 8.2337 | 420 | 8.2543 | 8.2612 | 3844.1279 | 720 | 1 | 2 |
| 18 | `large_kmer6s6_1e-5_rc_train` | `kmer6s6` | 1e-5 | 8.2339 | 1680 | 8.2402 | 8.2535 | 3790.3562 | 1980 | 3 | 1 |
| 19 | `large_kmer6s6_3e-5_rc_train` | `kmer6s6` | 3e-5 | 8.2406 | 780 | 8.2461 | 8.2495 | 3812.8342 | 1080 | 1 | 1 |
| 20 | `large_kmer6s6_1e-4_rc_train` | `kmer6s6` | 1e-4 | 8.2441 | 420 | 8.2526 | 8.2592 | 3837.6406 | 720 | 1 | 1 |

## Notes and analysis

- All 20 expected large-model directories were found and all have usable `Validation/Loss` and `Train/Loss` records.
- The tokenizer dominates the result: per-base runs reach approximately 1.33 validation loss, k-mer3 reaches approximately 1.35-1.52, while k-mer6/k-mer6s2/k-mer6s6 remain substantially higher.
- Among the four per-base runs, the 3e-5 run has the lowest recorded validation loss, followed very closely by 1e-5 and 1e-4. The baseline learning rate is already usable but stops earlier and has slightly worse final validation loss.
- Among k-mer3 runs, 1e-5 is strongest, followed by 3e-5; the 1e-4 and baseline runs deteriorate sharply by their final logged points. Among k-mer6 and k-mer6s2, 1e-5 or 3e-5 improves with longer training, but neither reaches the k-mer3/per-base regime.
- k-mer6s6 is nearly flat around validation loss 8.24 across all learning rates; this suggests tokenization/optimization or data alignment is limiting learning rather than only the learning rate.
- `final_val_loss` is not always the best point because some curves fluctuate or continue after their minimum. Use `best_val_loss` for model selection and inspect the corresponding checkpoint/log step before deployment.
- The baseline k-mer3, k-mer6, k-mer6s2, and per-base experiments contain an empty 88-byte TensorBoard event file plus a usable event file. The summary ignores empty files. k-mer6s6 contains two usable event files with the same experiment name; the row selects the run with the lower recorded best validation loss and reports `event_runs=2`.
- Peak GPU memory is stable within tokenizer families: about 21.4 GB for per-base/k-mer3 and about 21.6 GB for k-mer6 variants. Throughput reaches roughly 14.8-15.0k tokens/s after warm-up, so the large loss differences are not explained by a major throughput difference.
- Several runs show very large late gradient norms, especially k-mer6 and k-mer6s6. This is worth checking together with gradient clipping, scheduler behavior, and the saved best checkpoint; it may explain unstable or slow validation improvement.

## Reproduction

Run `python design/large_experiment_summary/summarize_large.py` from the TriEvo repository root. The CSV contains the full numeric summary and source event paths.
