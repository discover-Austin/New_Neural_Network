"""
Production Training System
===========================

Enterprise-grade training orchestration with comprehensive fault tolerance,
monitoring, distributed training, and optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import autocast, GradScaler
from typing import Optional, Dict, Any, Callable
import logging
import time
import os
from pathlib import Path
from dataclasses import dataclass

from .config import ProductionConfig
from .logging_utils import setup_logging, MetricsLogger, PerformanceMonitor, log_duration
from .distributed import (
    setup_distributed, cleanup_distributed, is_main_process,
    get_rank, get_world_size, setup_ddp, setup_fsdp,
    GradientAccumulator, CheckpointManager, all_reduce_dict,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """Encapsulates complete training state."""
    global_step: int = 0
    epoch: int = 0
    best_loss: float = float('inf')
    best_metric: float = float('-inf')
    total_tokens: int = 0
    start_time: float = 0.0


class ProductionTrainer:
    """
    Production-grade trainer with enterprise features:

    - Distributed training (DDP/FSDP)
    - Mixed precision (AMP)
    - Gradient accumulation
    - Checkpointing with fault tolerance
    - Comprehensive logging and monitoring
    - EMA (Exponential Moving Average)
    - Learning rate scheduling
    - Gradient clipping
    - Early stopping
    - Performance profiling
    """

    def __init__(
        self,
        config: ProductionConfig,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ):
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # Setup distributed training
        if config.distributed.enabled:
            self.rank, self.world_size, self.local_rank = setup_distributed(
                backend=config.distributed.backend,
                init_method=config.distributed.init_method,
            )
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0

        # Setup device
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        # Move model to device
        model = model.to(self.device)

        # Setup distributed model
        if config.distributed.enabled:
            if config.distributed.use_fsdp:
                model = setup_fsdp(
                    model,
                    sharding_strategy=config.distributed.fsdp_sharding_strategy,
                    cpu_offload=config.distributed.fsdp_cpu_offload,
                    mixed_precision=config.training.use_amp,
                )
            elif config.distributed.use_ddp:
                model = setup_ddp(
                    model,
                    find_unused_parameters=config.distributed.find_unused_parameters,
                )

        self.model = model

        # Setup optimizer
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.training.learning_rate,
                betas=(config.training.beta1, config.training.beta2),
                eps=config.training.eps,
                weight_decay=config.training.weight_decay,
            )
        self.optimizer = optimizer

        # Setup scheduler
        if scheduler is None and config.training.lr_schedule == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=config.training.max_steps,
                eta_min=config.training.learning_rate * config.training.min_lr_ratio,
            )
        self.scheduler = scheduler

        # Setup mixed precision
        self.use_amp = config.training.use_amp and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None

        # Setup gradient accumulation
        self.gradient_accumulator = GradientAccumulator(
            config.training.gradient_accumulation_steps
        )

        # Setup checkpointing
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.training.checkpoint_dir,
            max_checkpoints=config.training.save_total_limit,
        )

        # Setup logging
        if is_main_process():
            setup_logging(
                name="production_trainer",
                level=logging.INFO,
                log_file=Path(config.training.checkpoint_dir) / "training.log",
                structured=True,
                distributed=config.distributed.enabled,
            )

            self.metrics_logger = MetricsLogger(
                use_wandb=config.wandb_project is not None,
                use_tensorboard=config.tensorboard_dir is not None,
                wandb_project=config.wandb_project,
                wandb_entity=config.wandb_entity,
                tensorboard_dir=config.tensorboard_dir,
                log_interval=config.training.log_steps,
            )
        else:
            self.metrics_logger = None

        # Performance monitoring
        self.perf_monitor = PerformanceMonitor(log_interval=config.training.log_steps)

        # Training state
        self.state = TrainingState()

        # EMA model
        if config.training.use_ema:
            from torch_ema import ExponentialMovingAverage
            self.ema = ExponentialMovingAverage(
                model.parameters(),
                decay=config.training.ema_decay,
            )
        else:
            self.ema = None

        # Load checkpoint if resuming
        if config.training.resume_from_checkpoint:
            self.resume_from_checkpoint(config.training.resume_from_checkpoint)

        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"World size: {self.world_size}, Rank: {self.rank}")
        logger.info(f"Mixed precision: {self.use_amp}")
        logger.info(f"Gradient accumulation steps: {config.training.gradient_accumulation_steps}")

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Execute single training step.

        Args:
            batch: Batch of data

        Returns:
            Dictionary of metrics
        """
        self.model.train()

        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Forward pass with AMP
        if self.use_amp:
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = self.model(**batch)
                loss = outputs[1] if isinstance(outputs, tuple) else outputs["loss"]
        else:
            outputs = self.model(**batch)
            loss = outputs[1] if isinstance(outputs, tuple) else outputs["loss"]

        # Scale loss for gradient accumulation
        loss = self.gradient_accumulator.scale_loss(loss)

        # Backward pass with AMP
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step if accumulation complete
        if self.gradient_accumulator.should_step():
            # Gradient clipping
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.max_grad_norm,
            )

            # Optimizer step
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)

            # EMA update
            if self.ema is not None:
                self.ema.update()

        else:
            grad_norm = 0.0

        metrics = {
            "loss": loss.item() * self.config.training.gradient_accumulation_steps,
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

        return metrics

    @torch.no_grad()
    def evaluation_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Execute single evaluation step.

        Args:
            batch: Batch of data

        Returns:
            Dictionary of metrics
        """
        self.model.eval()

        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Forward pass
        if self.use_amp:
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = self.model(**batch)
                loss = outputs[1] if isinstance(outputs, tuple) else outputs["loss"]
        else:
            outputs = self.model(**batch)
            loss = outputs[1] if isinstance(outputs, tuple) else outputs["loss"]

        metrics = {
            "eval_loss": loss.item(),
        }

        return metrics

    def train(self) -> None:
        """
        Main training loop with comprehensive error handling and monitoring.
        """
        logger.info("Starting training...")
        self.state.start_time = time.time()

        try:
            for epoch in range(self.state.epoch, 1000000):  # Effectively unlimited
                self.state.epoch = epoch

                # Setup distributed sampler
                if hasattr(self.train_dataloader.sampler, 'set_epoch'):
                    self.train_dataloader.sampler.set_epoch(epoch)

                # Training epoch
                with log_duration(logger, f"epoch_{epoch}"):
                    self.train_epoch()

                # Evaluation
                if self.eval_dataloader is not None and \
                   self.state.global_step % self.config.training.eval_steps == 0:
                    self.evaluate()

                # Check stopping conditions
                if self.state.global_step >= self.config.training.max_steps:
                    logger.info(f"Reached max steps: {self.config.training.max_steps}")
                    break

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed with exception: {e}", exc_info=True)
            raise
        finally:
            self.cleanup()

        logger.info("Training completed")

    def train_epoch(self) -> None:
        """Train for one epoch."""
        for batch in self.train_dataloader:
            # Training step
            metrics = self.training_step(batch)

            self.state.global_step += 1

            # Reduce metrics across processes
            if self.config.distributed.enabled:
                metrics = all_reduce_dict(metrics, average=True)

            # Log metrics
            if is_main_process() and self.state.global_step % self.config.training.log_steps == 0:
                self.metrics_logger.log_metrics(metrics, step=self.state.global_step)

                # Log to console
                logger.info(
                    f"Step {self.state.global_step}: "
                    f"loss={metrics['loss']:.4f}, "
                    f"lr={metrics['learning_rate']:.2e}"
                )

            # Performance monitoring
            self.perf_monitor.log_step(
                batch_size=len(batch[list(batch.keys())[0]]),
                logger=self.metrics_logger,
                step=self.state.global_step,
            )

            # Checkpointing
            if self.state.global_step % self.config.training.save_steps == 0:
                self.save_checkpoint()

            # Check if max steps reached
            if self.state.global_step >= self.config.training.max_steps:
                break

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation on validation set.

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Running evaluation...")

        self.model.eval()

        all_metrics = []

        for batch in self.eval_dataloader:
            metrics = self.evaluation_step(batch)
            all_metrics.append(metrics)

        # Average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = sum(values) / len(values)

        # Reduce across processes
        if self.config.distributed.enabled:
            avg_metrics = all_reduce_dict(avg_metrics, average=True)

        # Log metrics
        if is_main_process():
            self.metrics_logger.log_metrics(avg_metrics, step=self.state.global_step)

            logger.info(
                f"Evaluation at step {self.state.global_step}: "
                f"eval_loss={avg_metrics['eval_loss']:.4f}"
            )

        return avg_metrics

    def save_checkpoint(self) -> None:
        """Save training checkpoint."""
        if not is_main_process():
            return

        logger.info(f"Saving checkpoint at step {self.state.global_step}")

        self.checkpoint_manager.save_checkpoint(
            step=self.state.global_step,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            metrics={
                "global_step": self.state.global_step,
                "epoch": self.state.epoch,
                "best_loss": self.state.best_loss,
            },
        )

    def resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """Resume training from checkpoint."""
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")

        metadata = self.checkpoint_manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )

        self.state.global_step = metadata.get("step", 0)
        self.state.epoch = metadata.get("epoch", 0)

        logger.info(f"Resumed from step {self.state.global_step}, epoch {self.state.epoch}")

    def cleanup(self) -> None:
        """Cleanup resources."""
        if is_main_process() and self.metrics_logger is not None:
            self.metrics_logger.close()

        if self.config.distributed.enabled:
            cleanup_distributed()

        logger.info("Trainer cleanup completed")
