import json
import time
from pathlib import Path
from typing import Any

from src.training.utils import ensure_dir
from src.utils.logger import get_logger


class BaseLogger:
    def log(self, metrics: dict[str, float], step: int, epoch: int, split: str) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return None


class ConsoleLogger(BaseLogger):
    def __init__(self, log_every_steps: int = 1) -> None:
        self.log_every_steps = max(1, int(log_every_steps))
        self.logger = get_logger("training")

    def log(self, metrics: dict[str, float], step: int, epoch: int, split: str) -> None:
        if split == "train_step" and step % self.log_every_steps != 0:
            return
        metric_text = " | ".join(f"{key}={value:.4f}" for key, value in sorted(metrics.items()))
        self.logger.info("epoch=%s step=%s split=%s | %s", epoch, step, split, metric_text)


class JsonlLogger(BaseLogger):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        ensure_dir(self.path.parent)
        self._file = self.path.open("a", encoding="utf-8")

    def log(self, metrics: dict[str, float], step: int, epoch: int, split: str) -> None:
        record: dict[str, Any] = {
            "time": time.time(),
            "step": int(step),
            "epoch": int(epoch),
            "split": split,
            "metrics": metrics,
        }
        self._file.write(json.dumps(record, sort_keys=True) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class WandbLogger(BaseLogger):
    def __init__(self, project: str, name: str, config: dict[str, Any] | None = None) -> None:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("wandb logging was enabled, but the wandb package is not installed") from exc

        self.wandb = wandb
        self.run = wandb.init(project=project, name=name, config=config)

    def log(self, metrics: dict[str, float], step: int, epoch: int, split: str) -> None:
        payload = dict(metrics)
        payload["epoch"] = epoch
        payload["split"] = split
        self.wandb.log(payload, step=step)

    def close(self) -> None:
        self.run.finish()


class CompositeLogger(BaseLogger):
    def __init__(self, loggers: list[BaseLogger] | None = None) -> None:
        self.loggers = loggers or []

    def log(self, metrics: dict[str, float], step: int, epoch: int, split: str) -> None:
        for logger in self.loggers:
            logger.log(metrics, step=step, epoch=epoch, split=split)

    def close(self) -> None:
        for logger in self.loggers:
            logger.close()
