"""Legacy dataclass configuration schema.

The training entry point now uses Hydra configs under ``configs/``. This module
is retained for callers that still need the previous JSON schema.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    context_length: int = 256
    d_model: int = 512
    num_heads: int = 8
    d_ff: int = 1344
    num_layers: int = 6
    theta: float = 10000.0
    checkpointing_blocksize: int = 0

    def __post_init__(self):
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")


@dataclass
class TrainingConfig:
    resume_from: int = 0
    batch_size: int = 64
    max_lr: float = 3e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01
    train_num_iterations: int = 6000
    max_l2_norm: float = 1.0
    eval_num_iterations: int = 300
    min_lr: Optional[float] = None
    warm_up_steps: Optional[int] = None
    anneal_steps: Optional[int] = None
    log_interval: Optional[int] = None
    val_interval: Optional[int] = None
    save_interval: Optional[int] = None
    val_num_iterations: Optional[int] = None

    def __post_init__(self):
        if self.min_lr is None:
            self.min_lr = self.max_lr * 0.01
        if self.log_interval is None:
            self.log_interval = max(1, self.train_num_iterations // 1000)
        if self.val_interval is None:
            self.val_interval = max(1, self.train_num_iterations // 10)
        if self.save_interval is None:
            self.save_interval = max(1, self.train_num_iterations // 10)
        if self.val_num_iterations is None:
            self.val_num_iterations = max(1, self.train_num_iterations // 20)
        if self.warm_up_steps is None:
            self.warm_up_steps = max(1, self.train_num_iterations // 10)
        if self.anneal_steps is None:
            self.anneal_steps = self.train_num_iterations


@dataclass
class TokenizerConfig:
    vocab_filepath: str = "vocab.pkl"
    merges_filepath: str = "merges.pkl"
    special_tokens: list = field(default_factory=lambda: ["<|endoftext|>"])


@dataclass
class PredictingConfig:
    max_length: int = 256
    temperature: float = 1.0
    top_p: float = 0.9


@dataclass
class ExperimentConfig:
    experiment_name: str = "default_experiment"
    train_dataset_path: str = ""
    val_dataset_path: str = ""
    test_dataset_path: str = ""
    output_dir: str = "out_experiments"
    checkpoint_dir: str = "out_experiments/checkpoints"
    log_dir: str = "out_experiments/logs"
    tb_log_dir: str = "out_experiments/tb_logs"
    do_accounting: bool = True
    do_train: bool = True
    do_eval: bool = True
    do_predict: bool = True
    device: str = "cuda"
    dtype: str = "float32"
    model_config: ModelConfig = field(default_factory=ModelConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    tokenizer_config: TokenizerConfig = field(default_factory=TokenizerConfig)
    predicting_config: PredictingConfig = field(default_factory=PredictingConfig)

    def __post_init__(self):
        if self.predicting_config.max_length > self.model_config.context_length:
            self.predicting_config.max_length = self.model_config.context_length

    @classmethod
    def load_from_json(cls, json_path: os.PathLike):
        with open(json_path, "r") as file:
            data = json.load(file)
        return cls(
            experiment_name=data.get("experiment_name", "default_experiment"),
            train_dataset_path=data.get("train_dataset_path", ""),
            val_dataset_path=data.get("val_dataset_path", ""),
            test_dataset_path=data.get("test_dataset_path", ""),
            output_dir=data.get("output_dir", "out_experiments"),
            checkpoint_dir=data.get("checkpoint_dir", ""),
            log_dir=data.get("log_dir", ""),
            tb_log_dir=data.get("tb_log_dir", ""),
            do_accounting=data.get("do_accounting", True),
            do_train=data.get("do_train", True),
            do_eval=data.get("do_eval", True),
            do_predict=data.get("do_predict", True),
            device=data.get("device", "cuda"),
            dtype=data.get("dtype", "float32"),
            model_config=ModelConfig(**data.get("model_config", {})),
            training_config=TrainingConfig(**data.get("training_config", {})),
            tokenizer_config=TokenizerConfig(**data.get("tokenizer_config", {})),
            predicting_config=PredictingConfig(**data.get("predicting_config", {})),
        )

    def save_to_json(self, json_path: os.PathLike) -> None:
        with open(json_path, "w") as file:
            json.dump(asdict(self), file, indent=4)


def write_default_config(save_path: os.PathLike) -> None:
    ExperimentConfig().save_to_json(save_path)
