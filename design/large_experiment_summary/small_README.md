# Small model experiment summary

Generated from `out_experiments/rep_ani_95/small_*` by `summarize_small.py`.
The primary comparison metric is the minimum recorded `Validation/Loss`; `final_val_loss` is the last validation point in the selected usable event log.

## Small model loss table

| Rank | Experiment | Tokenizer | LR | Batch | RC | Best val loss | Best step | Final val loss | Final PPL | Stop step | Checkpoints |
|---:|---|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `small_perbase_1e-5_rc_train` | `perbase` | 1.0e-05 | 8 | yes | 1.3311 | 1440 | 1.3335 | 3.7942 | 1740 | 2 |
| 2 | `small_kmer3_1e-5_rc_train` | `kmer3` | 1.0e-05 | 8 | yes | 1.3395 | 2400 | 1.3410 | 3.8228 | 2700 | 4 |
| 3 | `small_kmer6_1e-5_rc_train` | `kmer6` | 1.0e-05 | 8 | yes | 2.2528 | 5340 | 2.2610 | 9.5927 | 5640 | 9 |
| 4 | `small_kmer6s2_1e-5_rc_train` | `kmer6s2` | 1.0e-05 | 8 | yes | 4.5991 | 5460 | 4.6230 | 101.8014 | 5760 | 9 |
| 5 | `small_kmer6s6_train` | `kmer6s6` | 0.0003 | 32 | no | 8.2389 | 240 | 8.2394 | 3787.3716 | 468 | 0 |
| 6 | `small_kmer6s6_1e-5_rc_train` | `kmer6s6` | 1.0e-05 | 8 | yes | 8.2390 | 1080 | 8.2486 | 3822.2307 | 1380 | 2 |
| 7 | `small_kmer6s6_rc_train` | `kmer6s6` | 0.0003 | 32 | yes | 8.2422 | 420 | 8.2443 | 3806.0442 | 582 | 0 |

## Notes and analysis

- Found 7 small-model experiment directories under `rep_ani_95`; these are the complete small results currently present in the output directory.
- The strongest recorded run is `small_perbase_1e-5_rc_train`, with best validation loss approximately 1.3311. `small_kmer3_1e-5_rc_train` is close at approximately 1.3395.
- k-mer6 learns substantially further than the large-model k-mer6 runs in this output set, reaching best validation loss approximately 2.2528, but it remains behind per-base and k-mer3.
- k-mer6s2 reaches approximately 4.5991, while both k-mer6s6 variants remain near 8.24. This preserves the same broad tokenizer ordering seen in the large-model results.
- `small_kmer6s6_train` is the non-RC comparison and `small_kmer6s6_rc_train` is the RC baseline. Their best validation losses are nearly identical, so reverse-complement augmentation has little visible effect for this tokenizer in the short runs.
- `small_kmer6s6_rc_train` has three event files, including an empty file and a short event containing no validation loss; only the event with both train and validation loss is used. The source event paths are retained in the CSV.
- Peak memory is approximately 3.3-3.4 GB for the completed small runs, with much higher throughput after warm-up than the large runs. The k-mer6s2 run shows a very large late gradient norm, worth checking with clipping and scheduler behavior.

## Reproduction

Run `python design/large_experiment_summary/summarize_small.py` from the TriEvo repository root. The CSV contains the full numeric summary and source event paths.
