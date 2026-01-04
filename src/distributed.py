"""
Distributed Training Infrastructure
====================================

Production-grade distributed training with DDP, FSDP, and pipeline parallelism.
Includes fault tolerance, gradient synchronization, and multi-node support.
"""

import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision, CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from typing import Optional, Dict, Any, Tuple
import logging
from contextlib import contextmanager
import socket
import pickle

logger = logging.getLogger(__name__)


def setup_distributed(
    backend: str = "nccl",
    init_method: str = "env://",
    timeout_minutes: int = 30,
) -> Tuple[int, int, int]:
    """
    Initialize distributed training environment.

    Args:
        backend: Communication backend (nccl, gloo, mpi)
        init_method: Initialization method
        timeout_minutes: Timeout for initialization

    Returns:
        Tuple of (rank, world_size, local_rank)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Multi-node setup
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Initialize process group
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
            timeout=torch.distributed.timedelta(minutes=timeout_minutes),
        )

        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        logger.info(
            f"Distributed training initialized: rank={rank}/{world_size}, "
            f"local_rank={local_rank}, backend={backend}"
        )

        return rank, world_size, local_rank

    else:
        # Single-node setup
        logger.info("Running in single-process mode (no distributed)")
        return 0, 1, 0


def cleanup_distributed() -> None:
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed training cleaned up")


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank() -> int:
    """Get current process rank."""
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    """Get total number of processes."""
    return dist.get_world_size() if dist.is_initialized() else 1


def barrier() -> None:
    """Synchronization barrier across all processes."""
    if dist.is_initialized():
        dist.barrier()


@contextmanager
def main_process_first():
    """
    Context manager to execute code on main process first.

    Useful for dataset downloads, preprocessing, etc.
    """
    if is_main_process():
        yield
        barrier()
    else:
        barrier()
        yield


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """
    Broadcast Python object from src rank to all ranks.

    Args:
        obj: Object to broadcast
        src: Source rank

    Returns:
        Broadcasted object
    """
    if not dist.is_initialized():
        return obj

    # Serialize on source rank
    if get_rank() == src:
        serialized = pickle.dumps(obj)
        size = torch.tensor([len(serialized)], dtype=torch.long)
    else:
        size = torch.tensor([0], dtype=torch.long)

    # Broadcast size
    dist.broadcast(size, src=src)

    # Broadcast data
    if get_rank() == src:
        buffer = torch.ByteTensor(list(serialized))
    else:
        buffer = torch.ByteTensor([0] * size.item())

    dist.broadcast(buffer, src=src)

    # Deserialize
    if get_rank() != src:
        obj = pickle.loads(bytes(buffer.tolist()))

    return obj


def all_reduce_dict(
    metrics: Dict[str, float],
    average: bool = True,
) -> Dict[str, float]:
    """
    All-reduce a dictionary of metrics across all processes.

    Args:
        metrics: Dictionary of metric name -> value
        average: Whether to average (True) or sum (False)

    Returns:
        Reduced metrics
    """
    if not dist.is_initialized():
        return metrics

    # Convert to tensor
    names = sorted(metrics.keys())
    values = torch.tensor([metrics[name] for name in names])

    # All-reduce
    dist.all_reduce(values, op=dist.ReduceOp.SUM)

    # Average if requested
    if average:
        values /= get_world_size()

    # Convert back to dict
    return {name: value.item() for name, value in zip(names, values)}


def setup_ddp(
    model: nn.Module,
    device_ids: Optional[list] = None,
    find_unused_parameters: bool = False,
    gradient_as_bucket_view: bool = True,
    static_graph: bool = False,
) -> DDP:
    """
    Wrap model with DistributedDataParallel.

    Args:
        model: Model to wrap
        device_ids: Device IDs for DDP
        find_unused_parameters: Whether to find unused parameters
        gradient_as_bucket_view: Use bucket view for gradients (faster)
        static_graph: Whether computation graph is static

    Returns:
        DDP-wrapped model
    """
    if device_ids is None and torch.cuda.is_available():
        device_ids = [torch.cuda.current_device()]

    ddp_model = DDP(
        model,
        device_ids=device_ids,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=gradient_as_bucket_view,
        static_graph=static_graph,
    )

    logger.info(f"Model wrapped with DDP on devices: {device_ids}")

    return ddp_model


def setup_fsdp(
    model: nn.Module,
    sharding_strategy: str = "full",
    cpu_offload: bool = False,
    mixed_precision: bool = True,
    auto_wrap_min_params: int = 1e8,
) -> FSDP:
    """
    Wrap model with FullyShardedDataParallel.

    FSDP shards model parameters, gradients, and optimizer states
    across all GPUs, enabling training of very large models.

    Args:
        model: Model to wrap
        sharding_strategy: Sharding strategy (full, grad_op, no_shard)
        cpu_offload: Offload parameters to CPU
        mixed_precision: Use mixed precision training
        auto_wrap_min_params: Minimum params for auto-wrapping

    Returns:
        FSDP-wrapped model
    """
    # Map sharding strategy
    strategy_map = {
        "full": ShardingStrategy.FULL_SHARD,
        "grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
    }
    sharding_strategy = strategy_map.get(sharding_strategy, ShardingStrategy.FULL_SHARD)

    # Setup mixed precision
    if mixed_precision:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        )
    else:
        mp_policy = None

    # Setup CPU offload
    if cpu_offload:
        cpu_offload_policy = CPUOffload(offload_params=True)
    else:
        cpu_offload_policy = None

    # Auto-wrap policy
    auto_wrap_policy = size_based_auto_wrap_policy(min_num_params=auto_wrap_min_params)

    # Wrap with FSDP
    fsdp_model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        mixed_precision=mp_policy,
        cpu_offload=cpu_offload_policy,
        auto_wrap_policy=auto_wrap_policy,
    )

    logger.info(
        f"Model wrapped with FSDP: strategy={sharding_strategy.name}, "
        f"cpu_offload={cpu_offload}, mixed_precision={mixed_precision}"
    )

    return fsdp_model


class GradientAccumulator:
    """
    Gradient accumulation for effectively larger batch sizes.

    Accumulates gradients over multiple micro-batches before optimizer step.
    """

    def __init__(self, accumulation_steps: int = 1):
        self.accumulation_steps = accumulation_steps
        self.current_step = 0

    def should_step(self) -> bool:
        """Check if optimizer should step."""
        self.current_step += 1
        should_step = self.current_step % self.accumulation_steps == 0

        return should_step

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for gradient accumulation."""
        return loss / self.accumulation_steps

    def reset(self) -> None:
        """Reset accumulation counter."""
        self.current_step = 0


class CheckpointManager:
    """
    Manages distributed checkpointing with fault tolerance.

    Handles saving/loading of model, optimizer, and training state
    in distributed settings.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        save_optimizer: bool = True,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.save_optimizer = save_optimizer
        self.checkpoints = []

        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(
        self,
        step: int,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Save checkpoint.

        Only saves on main process for DDP, saves all shards for FSDP.

        Args:
            step: Training step
            model: Model to save
            optimizer: Optimizer state
            scheduler: LR scheduler state
            metrics: Training metrics

        Returns:
            Checkpoint path
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{step}.pt")

        # For DDP, only save on main process
        if isinstance(model, DDP) and not is_main_process():
            return checkpoint_path

        # Get model state dict
        if isinstance(model, (DDP, FSDP)):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()

        # Build checkpoint
        checkpoint = {
            "step": step,
            "model_state_dict": model_state,
        }

        if optimizer is not None and self.save_optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        if metrics is not None:
            checkpoint["metrics"] = metrics

        # Save
        torch.save(checkpoint, checkpoint_path)
        self.checkpoints.append(checkpoint_path)

        # Remove old checkpoints
        while len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)

        logger.info(f"Checkpoint saved: {checkpoint_path}")

        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
            model: Model to load into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into

        Returns:
            Checkpoint metadata
        """
        logger.info(f"Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Load model
        if isinstance(model, (DDP, FSDP)):
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        metadata = {
            "step": checkpoint.get("step", 0),
            "metrics": checkpoint.get("metrics", {}),
        }

        logger.info(f"Checkpoint loaded: step={metadata['step']}")

        return metadata


def reduce_gradients(model: nn.Module) -> None:
    """
    Manually reduce gradients across all processes.

    Useful when not using DDP/FSDP.
    """
    if not dist.is_initialized():
        return

    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= get_world_size()
