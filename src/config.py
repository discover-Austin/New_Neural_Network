"""
Production Configuration System
================================

Centralized configuration management for production deployments.
Supports environment variables, config files, and runtime overrides.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    model_type: str = "gpt"  # gpt, bert, t5, llama
    vocab_size: int = 50257
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    num_kv_heads: Optional[int] = None  # For GQA (LLaMA)
    d_ff: int = 3072
    max_seq_len: int = 2048
    dropout: float = 0.1
    activation: str = "gelu"
    norm_type: str = "layer_norm"  # layer_norm, rms_norm
    use_glu: bool = False
    tie_weights: bool = True
    attention_type: str = "multi_head"  # multi_head, multi_query, grouped_query, flash
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    max_grad_norm: float = 1.0

    # Schedule
    warmup_steps: int = 2000
    lr_schedule: str = "cosine"  # linear, cosine, inverse_sqrt
    min_lr_ratio: float = 0.1

    # Batch size and steps
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_steps: int = 100000
    num_epochs: Optional[int] = None

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # float16, bfloat16

    # Checkpointing
    save_steps: int = 1000
    save_total_limit: int = 5
    checkpoint_dir: str = "./checkpoints"
    resume_from_checkpoint: Optional[str] = None

    # Logging
    log_steps: int = 10
    eval_steps: int = 500

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9999

    # Gradient checkpointing
    gradient_checkpointing: bool = False


@dataclass
class DistributedConfig:
    """Distributed training configuration."""
    enabled: bool = False
    backend: str = "nccl"  # nccl, gloo
    init_method: str = "env://"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0

    # Data parallelism
    use_ddp: bool = True
    find_unused_parameters: bool = False

    # Model parallelism
    use_fsdp: bool = False
    fsdp_sharding_strategy: str = "full"  # full, grad_op, no_shard
    fsdp_cpu_offload: bool = False

    # Pipeline parallelism
    use_pipeline: bool = False
    pipeline_chunks: int = 1


@dataclass
class DataConfig:
    """Data configuration."""
    train_data_path: str = ""
    eval_data_path: str = ""
    tokenizer_path: str = ""

    # Preprocessing
    max_length: int = 2048
    truncation: bool = True
    padding: str = "max_length"  # max_length, longest, do_not_pad

    # DataLoader
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True

    # Streaming
    streaming: bool = False
    buffer_size: int = 10000


@dataclass
class ProductionConfig:
    """Complete production configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Runtime
    seed: int = 42
    device: str = "cuda"
    compile: bool = False  # torch.compile

    # Monitoring
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    tensorboard_dir: Optional[str] = None

    # Safety
    max_memory_gb: Optional[float] = None
    timeout_minutes: Optional[int] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ProductionConfig":
        """Create config from dictionary."""
        model_config = ModelConfig(**config_dict.get("model", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        distributed_config = DistributedConfig(**config_dict.get("distributed", {}))
        data_config = DataConfig(**config_dict.get("data", {}))

        return cls(
            model=model_config,
            training=training_config,
            distributed=distributed_config,
            data=data_config,
            **{k: v for k, v in config_dict.items()
               if k not in ["model", "training", "distributed", "data"]}
        )

    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> "ProductionConfig":
        """Load config from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "ProductionConfig":
        """Load config from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_env(cls) -> "ProductionConfig":
        """Load config from environment variables."""
        config_dict = {}

        # Parse environment variables with prefix CONFIG_
        for key, value in os.environ.items():
            if key.startswith("CONFIG_"):
                # Convert CONFIG_MODEL_D_MODEL to model.d_model
                parts = key[7:].lower().split("_")

                # Try to parse as number or bool
                try:
                    value = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    pass

                # Build nested dict
                current = config_dict
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value

        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model": asdict(self.model),
            "training": asdict(self.training),
            "distributed": asdict(self.distributed),
            "data": asdict(self.data),
            "seed": self.seed,
            "device": self.device,
            "compile": self.compile,
            "wandb_project": self.wandb_project,
            "wandb_entity": self.wandb_entity,
            "tensorboard_dir": self.tensorboard_dir,
            "max_memory_gb": self.max_memory_gb,
            "timeout_minutes": self.timeout_minutes,
        }

    def save(self, path: Union[str, Path], format: str = "yaml"):
        """Save config to file."""
        path = Path(path)
        config_dict = self.to_dict()

        with open(path, 'w') as f:
            if format == "json":
                json.dump(config_dict, f, indent=2)
            elif format == "yaml":
                yaml.dump(config_dict, f, default_flow_style=False)
            else:
                raise ValueError(f"Unknown format: {format}")

        logger.info(f"Config saved to {path}")

    def validate(self) -> None:
        """Validate configuration values."""
        errors = []

        # Model validation
        if self.model.d_model % self.model.num_heads != 0:
            errors.append(f"d_model ({self.model.d_model}) must be divisible by num_heads ({self.model.num_heads})")

        if self.model.num_kv_heads is not None:
            if self.model.num_heads % self.model.num_kv_heads != 0:
                errors.append(f"num_heads ({self.model.num_heads}) must be divisible by num_kv_heads ({self.model.num_kv_heads})")

        # Training validation
        if self.training.learning_rate <= 0:
            errors.append(f"learning_rate must be positive, got {self.training.learning_rate}")

        if self.training.batch_size <= 0:
            errors.append(f"batch_size must be positive, got {self.training.batch_size}")

        # Distributed validation
        if self.distributed.enabled:
            if self.distributed.world_size < 1:
                errors.append(f"world_size must be >= 1, got {self.distributed.world_size}")

            if self.distributed.rank >= self.distributed.world_size:
                errors.append(f"rank ({self.distributed.rank}) must be < world_size ({self.distributed.world_size})")

        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

        logger.info("Configuration validation passed")


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    use_env: bool = True,
    overrides: Optional[Dict[str, Any]] = None,
) -> ProductionConfig:
    """
    Load production configuration with priority: overrides > file > env > defaults.

    Args:
        config_path: Path to config file (JSON or YAML)
        use_env: Whether to load from environment variables
        overrides: Dictionary of override values

    Returns:
        Validated ProductionConfig
    """
    # Start with defaults
    config_dict = {}

    # Load from environment if enabled
    if use_env:
        env_config = ProductionConfig.from_env()
        config_dict.update(env_config.to_dict())

    # Load from file if provided
    if config_path is not None:
        config_path = Path(config_path)
        if config_path.suffix == ".json":
            file_config = ProductionConfig.from_json(config_path)
        elif config_path.suffix in [".yaml", ".yml"]:
            file_config = ProductionConfig.from_yaml(config_path)
        else:
            raise ValueError(f"Unknown config format: {config_path.suffix}")

        config_dict.update(file_config.to_dict())

    # Apply overrides
    if overrides:
        config_dict.update(overrides)

    # Create and validate config
    config = ProductionConfig.from_dict(config_dict)
    config.validate()

    return config
