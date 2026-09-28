from pathlib import Path
import csv
import math
import re
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
    return {
        "event": str(event_path.relative_to(ROOT)),
        "train_points": len(training),
        "val_points": len(validation),
        "final_step": training[-1].step,
        "best_val_step": best.step,
        "best_val_loss": best.value,
        "final_val_loss": validation[-1].value,
        "final_train_loss": training[-1].value,
        "final_val_perplexity": perplexity[-1].value if perplexity else math.nan,
        "min_grad_norm": min(item.value for item in grad_norm) if grad_norm else math.nan,
        "max_grad_norm": max(item.value for item in grad_norm) if grad_norm else math.nan,
        "mean_tokens_per_second": sum(item.value for item in throughput) / len(throughput) if throughput else math.nan,
        "peak_mem_mb": max(item.value for item in memory) if memory else math.nan,
    }


def experiment_info(name, config_path):
    config_text = config_path.read_text()
    tokenizer = name.removeprefix("small_").removesuffix("_train").removesuffix("_rc").split("_")[0]
    if tokenizer == name.removeprefix("small_").split("_")[0]:
        tokenizer = name.removeprefix("small_").split("_")[0]
    match = re.search(r"max_lr:\s*([^\n]+)", config_text)
    learning_rate = match.group(1).strip() if match else "n/a"
    batch_match = re.search(r"batch_size:\s*(\d+)", config_text)
    batch_size = batch_match.group(1) if batch_match else "n/a"
    reverse_complement = "_rc_" in name or name.endswith("_rc_train")
    return tokenizer, learning_rate, batch_size, reverse_complement


rows = []
for experiment_dir in sorted(EXPERIMENT_ROOT.glob("small_*")):
    config_path = next(experiment_dir.glob("*/resolved_config.yaml"))
    all_events = sorted(experiment_dir.glob("*/tb_logs/events*"))
    event_summaries = [
        summary
        for event_path in all_events
        if (summary := summarize_event(event_path)) is not None
    ]
    if not event_summaries:
        raise RuntimeError(f"No usable Validation/Loss event found for {experiment_dir.name}")
    selected = min(event_summaries, key=lambda item: item["best_val_loss"])
    tokenizer, learning_rate, batch_size, reverse_complement = experiment_info(experiment_dir.name, config_path)
    selected.update(
        {
            "experiment": experiment_dir.name,
            "tokenizer": tokenizer,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "reverse_complement": reverse_complement,
            "event_runs": len(event_summaries),
            "all_event_files": len(all_events),
            "duplicate_event_note": "; ".join(item["event"] for item in event_summaries),
        }
    )
    checkpoint_dir = next(experiment_dir.glob("*/checkpoints"))
    selected["checkpoint_count"] = len(list(checkpoint_dir.glob("checkpoint_iter_*.pth")))
    selected["has_best_checkpoint"] = (checkpoint_dir / "best_checkpoint.pth").exists()
    rows.append(selected)

columns = [
    "experiment", "tokenizer", "learning_rate", "batch_size", "reverse_complement",
    "best_val_loss", "best_val_step", "final_val_loss",
    "final_train_loss", "final_val_perplexity", "final_step", "train_points", "val_points",
    "checkpoint_count", "has_best_checkpoint", "event_runs", "all_event_files",
    "peak_mem_mb", "mean_tokens_per_second", "min_grad_norm", "max_grad_norm",
    "event", "duplicate_event_note",
]
with (REPORT_ROOT / "small_loss_summary.csv").open("w", newline="") as output:
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)


def fmt(value):
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return "n/a" if math.isnan(value) else f"{value:.4f}"
    return str(value)


ranking = sorted(rows, key=lambda row: row["best_val_loss"])
lines = [
    "# Small model experiment summary",
    "",
    "Generated from `out_experiments/rep_ani_95/small_*` by `summarize_small.py`.",
    "The primary comparison metric is the minimum recorded `Validation/Loss`; `final_val_loss` is the last validation point in the selected usable event log.",
    "",
    "## Small model loss table",
    "",
    "| Rank | Experiment | Tokenizer | LR | Batch | RC | Best val loss | Best step | Final val loss | Final PPL | Stop step | Checkpoints |",
    "|---:|---|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
]
for rank, row in enumerate(ranking, 1):
    lines.append(
        f"| {rank} | `{row['experiment']}` | `{row['tokenizer']}` | {row['learning_rate']} | {row['batch_size']} | "
        f"{'yes' if row['reverse_complement'] else 'no'} | {fmt(row['best_val_loss'])} | {row['best_val_step']} | "
        f"{fmt(row['final_val_loss'])} | {fmt(row['final_val_perplexity'])} | "
        f"{row['final_step']} | {row['checkpoint_count']} |"
    )

lines.extend(
    [
        "",
        "## Notes and analysis",
        "",
        f"- Found {len(rows)} small-model experiment directories under `rep_ani_95`; these are the complete small results currently present in the output directory.",
        "- The strongest recorded run is `small_perbase_1e-5_rc_train`, with best validation loss approximately 1.3311. `small_kmer3_1e-5_rc_train` is close at approximately 1.3395.",
        "- k-mer6 learns substantially further than the large-model k-mer6 runs in this output set, reaching best validation loss approximately 2.2528, but it remains behind per-base and k-mer3.",
        "- k-mer6s2 reaches approximately 4.5991, while both k-mer6s6 variants remain near 8.24. This preserves the same broad tokenizer ordering seen in the large-model results.",
        "- `small_kmer6s6_train` is the non-RC comparison and `small_kmer6s6_rc_train` is the RC baseline. Their best validation losses are nearly identical, so reverse-complement augmentation has little visible effect for this tokenizer in the short runs.",
        "- `small_kmer6s6_rc_train` has three event files, including an empty file and a short event containing no validation loss; only the event with both train and validation loss is used. The source event paths are retained in the CSV.",
        "- Peak memory is approximately 3.3-3.4 GB for the completed small runs, with much higher throughput after warm-up than the large runs. The k-mer6s2 run shows a very large late gradient norm, worth checking with clipping and scheduler behavior.",
        "",
        "## Reproduction",
        "",
        "Run `python design/large_experiment_summary/summarize_small.py` from the TriEvo repository root. The CSV contains the full numeric summary and source event paths.",
    ]
)
(REPORT_ROOT / "small_README.md").write_text("\n".join(lines) + "\n")