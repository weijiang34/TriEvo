# Tiny model experiment summary

Generated from `out_experiments/rep_ani_95/tiny_*` by `summarize_tiny.py`.
The primary comparison metric is the minimum recorded `Validation/Loss`; `final_val_loss` is the last validation point in the selected event log.

## Tiny model loss table

| Rank | Experiment | Tokenizer | LR | Batch | RC | Best val loss | Best step | Final val loss | Final PPL | Stop step | Checkpoints | Final ckpt |
|---:|---|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | `tiny_perbase_1e-5_rc_train` | `perbase` | 1.0e-05 | 8 | yes | 1.3481 | 1020 | 1.3508 | 3.8604 | 1320 | 2 | no |
| 2 | `tiny_kmer3_1e-5_rc_train` | `kmer3` | 1.0e-05 | 8 | yes | 1.3519 | 3060 | 1.3543 | 3.8739 | 3360 | 5 | no |
| 3 | `tiny_kmer6_1e-5_rc_train` | `kmer6` | 1.0e-05 | 8 | yes | 5.2771 | 6000 | 5.2771 | 195.8068 | 6000 | 9 | yes |
| 4 | `tiny_kmer6s2_1e-5_rc_train` | `kmer6s2` | 1.0e-05 | 8 | yes | 6.7077 | 5460 | 6.7293 | 836.5814 | 5760 | 9 | no |
| 5 | `tiny_kmer6s6_1e-5_rc_train` | `kmer6s6` | 1.0e-05 | 8 | yes | 8.2331 | 1860 | 8.2411 | 3793.7051 | 2160 | 3 | no |

## Notes and analysis

- Found 5 tiny-model experiment directories under `rep_ani_95`; these are the complete tiny results currently present.
- The best recorded run is `tiny_perbase_1e-5_rc_train`, with validation loss approximately 1.3481; `tiny_kmer3_1e-5_rc_train` follows at approximately 1.3519.
- k-mer6 reaches validation loss approximately 5.2771 after 6000 steps and has a final checkpoint, but remains behind per-base and k-mer3. k-mer6s2 reaches approximately 6.7077, while k-mer6s6 remains near 8.2331.
- The tokenizer ordering is consistent with the small and large summaries: per-base and k-mer3 are strongest, followed by k-mer6, k-mer6s2, and k-mer6s6.
- k-mer6 is the only tiny run with a `final_checkpoint_iter_6000.pth`; its validation loss is still improving at the end of the logged run, so its result is not directly comparable to the shorter runs as a fully converged endpoint.
- Peak memory is approximately 0.9 GB for per-base/k-mer3 and 1.6 GB for k-mer6 variants. The k-mer6s2 run has the largest late gradient norm and merits checking for optimization instability.

## Reproduction

Run `python design/large_experiment_summary/summarize_tiny.py` from the TriEvo repository root. The CSV contains the full numeric summary and source event paths.
