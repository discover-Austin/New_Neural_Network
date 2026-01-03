"""
Learning Rate Schedulers
=========================

Research-grounded learning rate schedules:

1. Linear Warmup with Cosine Decay
   - Reference: "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2019)
   - Used in: BERT, GPT, T5, most modern LLMs

2. Linear Warmup with Linear Decay
   - Simple and effective
   - Used in: Original Transformer

3. Inverse Square Root Schedule
   - Reference: "Attention is All You Need" (Vaswani et al., 2017)
   - lr(t) = d_model^(-0.5) * min(t^(-0.5), t * warmup_steps^(-1.5))

4. OneCycle
   - Reference: "Super-Convergence" (Smith, 2018)
   - Cyclical learning rate with momentum

Mathematical Foundation:
-----------------------
Warmup prevents instability in early training:
  - Large lr + random initialization → gradient explosion
  - Warmup: lr(t) = lr_max * min(t/warmup_steps, 1)

Cosine Decay:
  - lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T))
  - Smooth decay to minimum learning rate
  - Better than step decay (fewer hyperparameters)

Inverse Square Root:
  - lr(t) ∝ 1/√t after warmup
  - Theoretical justification from SGD convergence theory
"""

import math
from torch.optim.lr_scheduler import _LRScheduler
from typing import List


class LinearWarmupCosineDecayScheduler(_LRScheduler):
    """
    Linear warmup followed by cosine decay.

    Most popular schedule for transformer training.

    Schedule:
    - Warmup: lr increases linearly from 0 to max_lr
    - Decay: lr decreases following cosine curve to min_lr

    Reference: Used in BERT, GPT-3, T5, LLaMA
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate for current step."""
        step = self.last_epoch + 1

        if step < self.warmup_steps:
            # Linear warmup
            warmup_factor = step / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)

            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]


class LinearWarmupLinearDecayScheduler(_LRScheduler):
    """
    Linear warmup followed by linear decay.

    Simpler alternative to cosine decay.

    Schedule:
    - Warmup: lr increases linearly from 0 to max_lr
    - Decay: lr decreases linearly to min_lr
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate for current step."""
        step = self.last_epoch + 1

        if step < self.warmup_steps:
            # Linear warmup
            warmup_factor = step / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        else:
            # Linear decay
            decay_steps = self.total_steps - self.warmup_steps
            steps_remaining = self.total_steps - step
            decay_factor = steps_remaining / decay_steps

            return [
                self.min_lr + (base_lr - self.min_lr) * decay_factor
                for base_lr in self.base_lrs
            ]


class InverseSqrtScheduler(_LRScheduler):
    """
    Inverse square root learning rate schedule.

    Reference: "Attention is All You Need" (Vaswani et al., 2017)

    Schedule:
    lr(t) = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))

    Properties:
    - Increases during warmup
    - Decreases as 1/√t after warmup
    - No need to specify total steps
    """

    def __init__(
        self,
        optimizer,
        d_model: int,
        warmup_steps: int = 4000,
        last_epoch: int = -1,
    ):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.scale = d_model ** (-0.5)

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate for current step."""
        step = max(self.last_epoch + 1, 1)  # Avoid division by zero

        # Inverse square root schedule
        lr_scale = self.scale * min(
            step ** (-0.5),
            step * (self.warmup_steps ** (-1.5))
        )

        return [base_lr * lr_scale for base_lr in self.base_lrs]


class CyclicCosineScheduler(_LRScheduler):
    """
    Cyclic cosine annealing.

    Reference: "SGDR: Stochastic Gradient Descent with Warm Restarts"
               (Loshchilov & Hutter, ICLR 2017)

    Periodically resets learning rate to explore different basins.

    Schedule:
    lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t_cur / t_i))

    where t_cur is current step in cycle, t_i is cycle length
    """

    def __init__(
        self,
        optimizer,
        cycle_steps: int,
        min_lr: float = 0.0,
        cycle_mult: float = 1.0,
        last_epoch: int = -1,
    ):
        self.cycle_steps = cycle_steps
        self.min_lr = min_lr
        self.cycle_mult = cycle_mult

        self.current_cycle = 0
        self.current_cycle_steps = cycle_steps
        self.steps_in_cycle = 0

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate for current step."""
        # Cosine annealing within current cycle
        progress = self.steps_in_cycle / self.current_cycle_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

        lr_list = [
            self.min_lr + (base_lr - self.min_lr) * cosine_decay
            for base_lr in self.base_lrs
        ]

        # Update cycle tracking
        self.steps_in_cycle += 1

        if self.steps_in_cycle >= self.current_cycle_steps:
            # Start new cycle
            self.current_cycle += 1
            self.steps_in_cycle = 0
            self.current_cycle_steps = int(self.current_cycle_steps * self.cycle_mult)

        return lr_list


class ConstantWithWarmup(_LRScheduler):
    """
    Constant learning rate with initial warmup.

    Simple baseline schedule.

    Schedule:
    - Warmup: lr increases linearly to target
    - Constant: lr stays at target value
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate for current step."""
        step = self.last_epoch + 1

        if step < self.warmup_steps:
            # Linear warmup
            warmup_factor = step / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Constant
            return self.base_lrs


class PolynomialDecayScheduler(_LRScheduler):
    """
    Polynomial decay with warmup.

    Reference: Used in ELECTRA and other models

    Schedule:
    lr(t) = (lr_max - lr_min) * ((T - t) / T)^power + lr_min

    Common choice: power = 1.0 (linear decay)
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        power: float = 1.0,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.power = power
        self.min_lr = min_lr

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate for current step."""
        step = self.last_epoch + 1

        if step < self.warmup_steps:
            # Linear warmup
            warmup_factor = step / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        else:
            # Polynomial decay
            decay_steps = self.total_steps - self.warmup_steps
            steps_remaining = max(self.total_steps - step, 0)

            decay_factor = (steps_remaining / decay_steps) ** self.power

            return [
                self.min_lr + (base_lr - self.min_lr) * decay_factor
                for base_lr in self.base_lrs
            ]
