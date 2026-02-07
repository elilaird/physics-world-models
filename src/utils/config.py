"""Configuration management for IMU world modeling experiments."""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Any


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    data_dir: str = "data/kaist"
    processed_dir: str = "data/processed"
    splits_dir: str = "data/splits"
    window_size: float = 5.0
    stride: float = 0.5
    sampling_rate: int = 100
    normalize: bool = True
    train_sequences: List[str] = field(default_factory=lambda: ["urban01", "urban02", "urban03"])
    val_sequences: List[str] = field(default_factory=lambda: ["urban04"])
    test_sequences: List[str] = field(default_factory=lambda: ["urban05"])


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    name: str = "tiny_tcn"
    input_channels: int = 6
    window_samples: int = 500
    hidden_channels: int = 16
    num_blocks: int = 3
    output_dim: int = 6
    dropout: float = 0.1
    activation: str = "relu6"
    norm_type: str = "layer"


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 150
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    position_loss_weight: float = 1.0
    orientation_loss_weight: float = 0.5
    velocity_loss_weight: float = 0.3
    regularization_weight: float = 1e-5
    quantize_aware: bool = False
    distill_from: Optional[str] = None
    distill_temperature: float = 3.0
    distill_alpha: float = 0.3
    gradient_clip: float = 1.0
    num_workers: int = 4


@dataclass
class MobileConfig:
    """Mobile export configuration."""
    target_format: str = "coreml"
    quantize: bool = True
    quantize_mode: str = "int8"
    optimize_for_ane: bool = True
    input_shape: List[int] = field(default_factory=lambda: [1, 500, 6])
    max_model_size_mb: float = 5.0


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    name: str = "default"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    mobile: MobileConfig = field(default_factory=MobileConfig)
    output_dir: str = "results"
    checkpoint_dir: str = "checkpoints"
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None


def load_config(config_path: str) -> ExperimentConfig:
    """Load experiment configuration from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    config = ExperimentConfig()

    for key in ("name", "seed", "output_dir", "checkpoint_dir", "wandb_project", "wandb_entity"):
        if key in raw:
            setattr(config, key, raw[key])

    if "data" in raw:
        config.data = _update_dataclass(DataConfig(), raw["data"])
    if "model" in raw:
        config.model = _update_dataclass(ModelConfig(), raw["model"])
    if "training" in raw:
        config.training = _update_dataclass(TrainingConfig(), raw["training"])
    if "mobile" in raw:
        config.mobile = _update_dataclass(MobileConfig(), raw["mobile"])

    return config


def _update_dataclass(instance: Any, updates: dict) -> Any:
    """Update dataclass fields from a dictionary."""
    for key, value in updates.items():
        if hasattr(instance, key):
            setattr(instance, key, value)
    return instance


def save_config(config: ExperimentConfig, save_path: str) -> None:
    """Save experiment configuration to a YAML file."""
    import dataclasses

    def _to_dict(obj: Any) -> Any:
        if dataclasses.is_dataclass(obj):
            return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
        return obj

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(_to_dict(config), f, default_flow_style=False, sort_keys=False)
