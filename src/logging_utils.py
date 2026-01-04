"""
Production Logging Infrastructure
==================================

Comprehensive logging system for production deployments with structured
logging, metrics, and integration with monitoring systems.
"""

import logging
import sys
import time
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import traceback
from contextlib import contextmanager
import torch
import torch.distributed as dist


class StructuredFormatter(logging.Formatter):
    """
    Structured JSON formatter for production logging.

    Outputs logs in JSON format for easy parsing by log aggregation systems.
    """

    def __init__(self, include_extras: bool = True):
        super().__init__()
        self.include_extras = include_extras

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add extra fields
        if self.include_extras:
            for key, value in record.__dict__.items():
                if key not in [
                    "name", "msg", "args", "created", "filename", "funcName",
                    "levelname", "levelno", "lineno", "module", "msecs",
                    "message", "pathname", "process", "processName", "relativeCreated",
                    "thread", "threadName", "exc_info", "exc_text", "stack_info"
                ]:
                    log_data[key] = value

        return json.dumps(log_data)


class DistributedFilter(logging.Filter):
    """Filter to add distributed training context to logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add rank information if distributed."""
        if dist.is_initialized():
            record.rank = dist.get_rank()
            record.world_size = dist.get_world_size()
            record.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        else:
            record.rank = 0
            record.world_size = 1
            record.local_rank = 0

        return True


def setup_logging(
    name: str = "nlp_system",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    structured: bool = True,
    distributed: bool = False,
) -> logging.Logger:
    """
    Setup production logging with structured output and distributed support.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file path for logging
        structured: Whether to use JSON structured logging
        distributed: Whether to add distributed training context

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if structured:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    console_handler.setFormatter(formatter)

    if distributed:
        console_handler.addFilter(DistributedFilter())

    logger.addHandler(console_handler)

    # File handler if specified
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)

        if distributed:
            file_handler.addFilter(DistributedFilter())

        logger.addHandler(file_handler)

    return logger


class MetricsLogger:
    """
    Production metrics logger for tracking training and inference metrics.

    Integrates with Weights & Biases, TensorBoard, and custom backends.
    """

    def __init__(
        self,
        use_wandb: bool = False,
        use_tensorboard: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        tensorboard_dir: Optional[Path] = None,
        log_interval: int = 10,
    ):
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self.log_interval = log_interval
        self.step = 0

        # Initialize W&B
        if self.use_wandb:
            try:
                import wandb
                self.wandb = wandb
                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    config={},
                )
            except ImportError:
                logging.warning("wandb not installed, disabling W&B logging")
                self.use_wandb = False

        # Initialize TensorBoard
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tensorboard_dir = Path(tensorboard_dir or "./tensorboard")
                self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
                self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir))
            except ImportError:
                logging.warning("tensorboard not installed, disabling TensorBoard logging")
                self.use_tensorboard = False

        # In-memory metrics buffer
        self.metrics_buffer: Dict[str, List[float]] = {}

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        commit: bool = True,
    ) -> None:
        """
        Log metrics to all enabled backends.

        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step number (auto-increments if None)
            commit: Whether to immediately flush to backends
        """
        if step is None:
            step = self.step
            self.step += 1

        # Log to W&B
        if self.use_wandb and commit:
            self.wandb.log(metrics, step=step)

        # Log to TensorBoard
        if self.use_tensorboard:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)
            if commit:
                self.writer.flush()

        # Add to buffer
        for key, value in metrics.items():
            if key not in self.metrics_buffer:
                self.metrics_buffer[key] = []
            self.metrics_buffer[key].append(value)

    def log_histograms(
        self,
        tensors: Dict[str, torch.Tensor],
        step: Optional[int] = None,
    ) -> None:
        """Log tensor histograms for weight/gradient analysis."""
        if step is None:
            step = self.step

        if self.use_tensorboard:
            for name, tensor in tensors.items():
                self.writer.add_histogram(name, tensor.detach().cpu(), step)

    def log_text(
        self,
        tag: str,
        text: str,
        step: Optional[int] = None,
    ) -> None:
        """Log text output (e.g., generated samples)."""
        if step is None:
            step = self.step

        if self.use_tensorboard:
            self.writer.add_text(tag, text, step)

        if self.use_wandb:
            self.wandb.log({tag: self.wandb.Html(text)}, step=step)

    def get_average(self, key: str, last_n: Optional[int] = None) -> float:
        """Get average of logged metrics."""
        if key not in self.metrics_buffer:
            return 0.0

        values = self.metrics_buffer[key]
        if last_n is not None:
            values = values[-last_n:]

        return sum(values) / len(values) if values else 0.0

    def clear_buffer(self) -> None:
        """Clear metrics buffer."""
        self.metrics_buffer.clear()

    def close(self) -> None:
        """Close all logging backends."""
        if self.use_tensorboard:
            self.writer.close()

        if self.use_wandb:
            self.wandb.finish()


@contextmanager
def log_duration(logger: logging.Logger, operation: str, level: int = logging.INFO):
    """
    Context manager to log operation duration.

    Usage:
        with log_duration(logger, "training_step"):
            # ... training code ...
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        logger.log(
            level,
            f"{operation} completed",
            extra={"duration_seconds": duration, "operation": operation}
        )


class PerformanceMonitor:
    """
    Monitor system performance metrics during training/inference.

    Tracks GPU memory, throughput, and system resources.
    """

    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.step_times: List[float] = []
        self.last_log_time = time.time()

    def log_step(
        self,
        batch_size: int,
        logger: Optional[MetricsLogger] = None,
        step: Optional[int] = None,
    ) -> None:
        """Log performance metrics for a training/inference step."""
        current_time = time.time()
        self.step_times.append(current_time)

        # Log periodically
        if len(self.step_times) % self.log_interval == 0:
            # Calculate throughput
            time_window = current_time - self.last_log_time
            steps_in_window = min(len(self.step_times), self.log_interval)
            throughput = (steps_in_window * batch_size) / time_window

            metrics = {
                "throughput_samples_per_second": throughput,
                "step_time_ms": (time_window / steps_in_window) * 1000,
            }

            # GPU metrics
            if torch.cuda.is_available():
                metrics["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
                metrics["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
                metrics["gpu_utilization"] = torch.cuda.utilization()

            if logger is not None:
                logger.log_metrics(metrics, step=step)

            self.last_log_time = current_time
            self.step_times = self.step_times[-self.log_interval:]

    def reset(self) -> None:
        """Reset monitoring state."""
        self.step_times.clear()
        self.last_log_time = time.time()


import os
