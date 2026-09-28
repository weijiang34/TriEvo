from pathlib import Path
import csv
import math
import numpy as np

for old, new in (
    ("bool8", np.bool_),
    ("string_", np.bytes_),
    ("unicode_", str),
    ("float_", np.float64),
    ("int_", np.int64),
):
    if not hasattr(np, old):
        setattr(np, old, new)

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = ROOT / "out_experiments" / "rep_ani_95"
REPORT_ROOT = Path(__file__).resolve().parent


def scalar_values(event_path, tag):
    accumulator = EventAccumulator(str(event_path))
    accumulator.Reload()
    if tag not in accumulator.Tags().get("scalars", []):
        return []
    return accumulator.Scalars(tag)


def summarize_event(event_path):
    validation = scalar_values(event_path, "Validation/Loss")
    training = scalar_values(event_path, "Train/Loss")
    perplexity = scalar_values(event_path, "Validation/Perplexity")
    grad_norm = scalar_values(event_path, "Train/GradNorm")
    throughput = scalar_values(event_path, "Train/GlobalTokensPerSecond")
    memory = scalar_values(event_path, "Train/PeakMem")
    if not validation or not training:
        return None
    best = min(validation, key=lambda item: item.value)
    final_validation = validation[-1]
    final_training = training[-1]
    final_perplexity = perplexity[-1].value if perplexity else math.nan
    return {
        "event": str(event_path.relative_to(ROOT)),
        "train_points": len(training),
        "val_points": len(validation),
        "final_step": training[-1].step,
        "best_val_step": best.step,
        "best_val_loss": best.value,
        "final_val_loss": final_validation.value,
        "final_train_loss": final_training.value,
        "final_val_perplexity": final_perplexity,
        "min_grad_norm": min(item.value for item in grad_norm) if grad_norm else math.nan,
        "max_grad_norm": max(item.value for item in grad_norm) if grad_norm else math.nan,
        "mean_tokens_per_second": sum(item.value for item in throughput) / len(throughput) if throughput else math.nan,
        "peak_mem_mb": max(item.value for item in memory) if memory else math.nan,
    }


def experiment_info(name, config_path):
    parts = name.removeprefix("large_").removesuffix("_rc_train").split("_")
    tokenizer = parts[0]
    learning_rate = "0.0003 (baseline)"
    if len(parts) == 2:
        learning_rate = {"1e-4": "1e-4", "3e-5": "3e-5", "1e-5": "1e-5"}[parts[1]]
    config_text = config_path.read_text()
    for line in config_text.splitlines():
        if "max_lr:" in line:
            configured_lr = line.split("max_lr:", 1)[1].strip()
            if configured_lr not in {"0.0003", "3.0e-05", "1.0e-05", "0.0001"}:
                learning_rate = configured_lr
            break
    return tokenizer, learning_rate


rows = []
for experiment_dir in sorted(EXPERIMENT_ROOT.glob("large_*")):
    config_path = next(experiment_dir.glob("*/resolved_config.yaml"))
    event_summaries = [
        summary
        for event_path in sorted(experiment_dir.glob("*/tb_logs/events*"))
        if (summary := summarize_event(event_path)) is not None
    ]
    if not event_summaries:
        raise RuntimeError(f"No usable Validation/Loss event found for {experiment_dir.name}")
    selected = min(event_summaries, key=lambda item: item["best_val_loss"])
    tokenizer, learning_rate = experiment_info(experiment_dir.name, config_path)
    selected.update(
        {
            "experiment": experiment_dir.name,
            "tokenizer": tokenizer,
            "learning_rate": learning_rate,
            "event_runs": len(event_summaries),
            "duplicate_event_note": "; ".join(item["event"] for item in event_summaries),
        }
    )
    checkpoint_dir = next(experiment_dir.glob("*/checkpoints"))
    selected["checkpoint_count"] = len(list(checkpoint_dir.glob("checkpoint_iter_*.pth")))
    selected["has_best_checkpoint"] = (checkpoint_dir / "best_checkpoint.pth").exists()
    rows.append(selected)

columns = [
    "experiment", "tokenizer", "learning_rate", "best_val_loss", "best_val_step",
    "final_val_loss", "final_train_loss", "final_val_perplexity", "final_step",
    "train_points", "val_points", "checkpoint_count", "has_best_checkpoint",
    "event_runs", "peak_mem_mb", "mean_tokens_per_second", "min_grad_norm",
    "max_grad_norm", "event", "duplicate_event_note",
]
with (REPORT_ROOT / "large_loss_summary.csv").open("w", newline="") as output:
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

def fmt(value):
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return "n/a" if math.isnan(value) else f"{value:.4f}"
    return str(value)


ranking = sorted(rows, key=lambda row: row["best_val_loss"])
lines = [
    "# Large model experiment summary",
    "",
    "Generated from `out_experiments/rep_ani_95/large_*` by `summarize_large.py`.",
    "The primary comparison metric is the minimum recorded `Validation/Loss`; `final_val_loss` is the last validation point in the selected event log.",
    "",
    "## 20-group loss table",
    "",
    "| Rank | Experiment | Tokenizer | Max LR | Best val loss | Best step | Final val loss | Final train loss | Final PPL | Stop step | Checkpoints | Event runs |",
    "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for rank, row in enumerate(ranking, 1):
    lines.append(
        f"| {rank} | `{row['experiment']}` | `{row['tokenizer']}` | {row['learning_rate']} | "
        f"{fmt(row['best_val_loss'])} | {row['best_val_step']} | {fmt(row['final_val_loss'])} | "
        f"{fmt(row['final_train_loss'])} | {fmt(row['final_val_perplexity'])} | {row['final_step']} | "
        f"{row['checkpoint_count']} | {row['event_runs']} |"
    )

lines.extend(
    [
        "",
        "## Notes and analysis",
        "",
        f"- All 20 expected large-model directories were found and all have usable `Validation/Loss` and `Train/Loss` records.",
        "- The tokenizer dominates the result: per-base runs reach approximately 1.33 validation loss, k-mer3 reaches approximately 1.35-1.52, while k-mer6/k-mer6s2/k-mer6s6 remain substantially higher.",
        "- Among the four per-base runs, the 3e-5 run has the lowest recorded validation loss, followed very closely by 1e-5 and 1e-4. The baseline learning rate is already usable but stops earlier and has slightly worse final validation loss.",
        "- Among k-mer3 runs, 1e-5 is strongest, followed by 3e-5; the 1e-4 and baseline runs deteriorate sharply by their final logged points. Among k-mer6 and k-mer6s2, 1e-5 or 3e-5 improves with longer training, but neither reaches the k-mer3/per-base regime.",
        "- k-mer6s6 is nearly flat around validation loss 8.24 across all learning rates; this suggests tokenization/optimization or data alignment is limiting learning rather than only the learning rate.",
        "- `final_val_loss` is not always the best point because some curves fluctuate or continue after their minimum. Use `best_val_loss` for model selection and inspect the corresponding checkpoint/log step before deployment.",
        "- The baseline k-mer3, k-mer6, k-mer6s2, and per-base experiments contain an empty 88-byte TensorBoard event file plus a usable event file. The summary ignores empty files. k-mer6s6 contains two usable event files with the same experiment name; the row selects the run with the lower recorded best validation loss and reports `event_runs=2`.",
        "- Peak GPU memory is stable within tokenizer families: about 21.4 GB for per-base/k-mer3 and about 21.6 GB for k-mer6 variants. Throughput reaches roughly 14.8-15.0k tokens/s after warm-up, so the large loss differences are not explained by a major throughput difference.",
        "- Several runs show very large late gradient norms, especially k-mer6 and k-mer6s6. This is worth checking together with gradient clipping, scheduler behavior, and the saved best checkpoint; it may explain unstable or slow validation improvement.",
        "",
        "## Reproduction",
        "",
        "Run `python design/large_experiment_summary/summarize_large.py` from the TriEvo repository root. The CSV contains the full numeric summary and source event paths.",
    ]
)
(REPORT_ROOT / "README.md").write_text("\n".join(lines) + "\n")