from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn.functional as F

StepOutput = dict[str,Any]

class Task(ABC):
    """Task-specific training logic consumed by the generic Trainer."""

    @abstractmethod
    def training_step(self, model: torch.nn.Module, batch: dict[str, Any]) -> StepOutput:
        raise NotImplementedError

    @abstractmethod
    def validation_step(self, model: torch.nn.Module, batch: dict[str, Any]) -> StepOutput:
        raise NotImplementedError
