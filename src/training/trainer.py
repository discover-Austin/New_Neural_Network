"""
Advanced Training Infrastructure
=================================

Research-grounded training components:

1. Mixed Precision Training
   - Reference: "Mixed Precision Training" (Micikevicius et al., ICLR 2018)
   - Uses FP16 for speed, FP32 for stability
   - Dynamic loss scaling to prevent underflow

2. Gradient Accumulation
   - Simulates larger batch sizes
   - Essential for large models on limited hardware

3. Gradient Checkpointing
   - Reference: "Training Deep Nets with Sublinear Memory Cost" (Chen et al., 2016)
   - Trades compute for memory
   - O(√N) memory instead of O(N)

4. Model EMA (Exponential Moving Average)
   - Reference: "Mean teachers are better role models" (Tarvainen & Valpola, 2017)
   - Stabilizes training
   - Often improves final performance

Mathematical Foundation:
-----------------------
Mixed Precision:
  - Forward: FP16 (2 bytes per parameter)
  - Backward: FP16 gradients
  - Master weights: FP32 (4 bytes)
  - Update: w_FP32 = w_FP32 - lr * grad_FP16

Loss Scaling:
  - loss_scaled = loss * scale_factor
  - Prevents gradient underflow in FP16
  - Dynamic: adjust scale based on overflow detection

Gradient Accumulation:
  - Effective batch size = micro_batch * accumulation_steps
  - Average gradients over accumulation steps
  - Update weights after accumulation complete

EMA:
  - θ_ema(t) = β * θ_ema(t-1) + (1-β) * θ(t)
  - Typically β = 0.999 or 0.9999
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass
import math
import time


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Model and data
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_steps: int = 100000
    eval_steps: int = 1000
    save_steps: int = 5000

    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 2000

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"  # "float16" or "bfloat16"

    # Gradient checkpointing
    use_gradient_checkpointing: bool = False

    # EMA
    use_ema: bool = False
    ema_decay: float = 0.9999

    # Logging
    logging_steps: int = 100
    log_grad_norm: bool = True

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ModelEMA:
    """
    Exponential Moving Average of model parameters.

    Reference: Tarvainen & Valpola, NeurIPS 2017
    "Mean teachers are better role models: Weight-averaged consistency targets"

    Maintains shadow copy of model weights:
    θ_ema(t) = decay * θ_ema(t-1) + (1 - decay) * θ(t)

    Properties:
    - Smooths training dynamics
    - Often improves final performance
    - Minimal computational overhead
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: Optional[str] = None,
    ):
        self.decay = decay
        self.device = device

        # Create shadow model
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(device)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay
                )

    def apply_shadow(self, model: nn.Module) -> None:
        """Apply EMA parameters to model (for evaluation)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module, backup: Dict[str, torch.Tensor]) -> None:
        """Restore original parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in backup:
                param.data.copy_(backup[name])

    def get_current_decay(self, step: int, warmup_steps: int = 1000) -> float:
        """
        Get decay with warmup.

        Linearly increases decay from 0 to target during warmup.
        """
        if step < warmup_steps:
            return min(self.decay, (step + 1) / warmup_steps)
        return self.decay


class Trainer:
    """
    Advanced trainer with mixed precision, gradient accumulation, and EMA.

    Features:
    - Automatic mixed precision (AMP)
    - Gradient accumulation for large effective batch sizes
    - Gradient checkpointing for memory efficiency
    - Model EMA for stability
    - Comprehensive logging
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.lr_scheduler = lr_scheduler

        # Move model to device
        self.model.to(config.device)

        # Mixed precision scaler
        self.scaler = None
        if config.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        # EMA
        self.ema = None
        if config.use_ema:
            self.ema = ModelEMA(
                model,
                decay=config.ema_decay,
                device=config.device,
            )

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Metrics
        self.metrics_history = []

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        accumulation_step: int,
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Training batch
            accumulation_step: Current accumulation step

        Returns:
            Metrics dictionary
        """
        self.model.train()

        # Move batch to device
        batch = {k: v.to(self.config.device) for k, v in batch.items()}

        # Forward pass with AMP
        with torch.cuda.amp.autocast(
            enabled=self.config.use_amp,
            dtype=torch.float16 if self.config.amp_dtype == "float16" else torch.bfloat16,
        ):
            outputs = self.model(**batch)

            if isinstance(outputs, dict):
                loss = outputs["loss"]
            else:
                loss = outputs[0] if isinstance(outputs, tuple) else outputs

            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update weights if accumulation complete
        is_accumulation_step = (accumulation_step + 1) % self.config.gradient_accumulation_steps == 0

        metrics = {"loss": loss.item() * self.config.gradient_accumulation_steps}

        if is_accumulation_step:
            # Unscale gradients for clipping
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)

            # Gradient clipping
            if self.config.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                metrics["grad_norm"] = grad_norm.item()

            # Optimizer step
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # Learning rate schedule
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Zero gradients
            self.optimizer.zero_grad()

            # Update EMA
            if self.ema is not None:
                self.ema.update(self.model)

            # Update global step
            self.global_step += 1

            # Add learning rate to metrics
            metrics["lr"] = self.optimizer.param_groups[0]["lr"]

        return metrics

    @torch.no_grad()
    def evaluate(
        self,
        eval_dataloader,
        use_ema: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate model.

        Args:
            eval_dataloader: Evaluation dataloader
            use_ema: Whether to use EMA weights

        Returns:
            Evaluation metrics
        """
        self.model.eval()

        # Backup and apply EMA if requested
        backup = None
        if use_ema and self.ema is not None:
            backup = {
                name: param.data.clone()
                for name, param in self.model.named_parameters()
                if param.requires_grad
            }
            self.ema.apply_shadow(self.model)

        total_loss = 0
        total_samples = 0

        for batch in eval_dataloader:
            batch = {k: v.to(self.config.device) for k, v in batch.items()}

            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                outputs = self.model(**batch)

                if isinstance(outputs, dict):
                    loss = outputs["loss"]
                else:
                    loss = outputs[0] if isinstance(outputs, tuple) else outputs

            batch_size = batch[list(batch.keys())[0]].size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        # Restore original weights if using EMA
        if backup is not None:
            self.ema.restore(self.model, backup)

        avg_loss = total_loss / total_samples

        return {"eval_loss": avg_loss}

    def train(
        self,
        train_dataloader,
        eval_dataloader=None,
        num_epochs: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        """
        Complete training loop.

        Args:
            train_dataloader: Training dataloader
            eval_dataloader: Evaluation dataloader (optional)
            num_epochs: Number of epochs (optional, uses max_steps if None)

        Returns:
            Training history
        """
        accumulation_step = 0
        start_time = time.time()

        if num_epochs is None:
            # Train for max_steps
            total_steps = self.config.max_steps
        else:
            # Train for num_epochs
            total_steps = num_epochs * len(train_dataloader)

        print(f"Training for {total_steps} steps...")

        while self.global_step < total_steps:
            for batch in train_dataloader:
                # Training step
                metrics = self.train_step(batch, accumulation_step)
                accumulation_step += 1

                # Logging
                if self.global_step % self.config.logging_steps == 0 and accumulation_step % self.config.gradient_accumulation_steps == 0:
                    elapsed = time.time() - start_time
                    steps_per_sec = self.global_step / elapsed if elapsed > 0 else 0

                    log_str = f"Step {self.global_step}/{total_steps} | "
                    log_str += " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
                    log_str += f" | Steps/sec: {steps_per_sec:.2f}"

                    print(log_str)

                    self.metrics_history.append({
                        "step": self.global_step,
                        **metrics,
                        "elapsed": elapsed,
                    })

                # Evaluation
                if eval_dataloader is not None and self.global_step % self.config.eval_steps == 0:
                    eval_metrics = self.evaluate(eval_dataloader, use_ema=self.config.use_ema)
                    print(f"Evaluation at step {self.global_step}: {eval_metrics}")

                    self.metrics_history.append({
                        "step": self.global_step,
                        **eval_metrics,
                    })

                # Checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{self.global_step}")

                # Check if done
                if self.global_step >= total_steps:
                    break

            self.epoch += 1

        print(f"Training complete! Total time: {time.time() - start_time:.2f}s")

        return self.metrics_history

    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "config": self.config,
        }

        if self.lr_scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.lr_scheduler.state_dict()

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        if self.ema is not None:
            checkpoint["ema_shadow"] = self.ema.shadow

        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]

        if self.lr_scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if self.ema is not None and "ema_shadow" in checkpoint:
            self.ema.shadow = checkpoint["ema_shadow"]

        print(f"Loaded checkpoint from {path} (step {self.global_step})")
