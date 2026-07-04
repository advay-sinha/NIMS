"""Shared PyTorch training engine for Engine B models.

Purpose
-------
One reusable fit/predict implementation for every deep-learning model, so the
concrete models (MLP, CNN, LSTM, Transformer) only define their network
architecture. Implements the training loop, validation loop, early stopping
with best-model restore, learning-rate scheduling, mixed precision, gradient
accumulation/clipping, optional checkpointing and per-epoch logging.

Integration
-----------
Subclasses of :class:`TorchModelBase` satisfy the Engine A
:class:`src.models.base.BaseModel` contract, so the existing trainer,
metric suite and experiment manifests are reused without duplication.

Configuration
-------------
``params`` (from ``configs/deep_learning.yaml > models.<name>.params``) holds
the architecture hyperparameters plus a ``training`` sub-mapping:
``batch_size``, ``epochs``, ``learning_rate``, ``optimizer``, ``scheduler``,
``weight_decay``, ``early_stopping`` (``enabled``/``patience``/``min_delta``),
``mixed_precision``, ``gradient_clipping``, ``gradient_accumulation_steps``,
``checkpointing`` (``enabled``/``dir``), ``progress_bar``, plus GPU-throughput
keys ``tf32``, ``cudnn_benchmark``, ``num_workers``, ``prefetch_factor`` and
``persistent_workers``.

Limitations
-----------
Inputs are tabular feature matrices (rows = samples); sequence models treat
the feature vector as a synthetic sequence. Device selection is centralized in
:mod:`src.utils.hardware` (CLAUDE.md > GPU Rules).
"""

from __future__ import annotations

import logging
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any, Mapping

from src.models.base import BaseModel

logger = logging.getLogger(__name__)

# Training defaults; every value is overridable from configuration.
DEFAULT_TRAINING: dict[str, Any] = {
    "batch_size": 512,
    "epochs": 50,
    "learning_rate": 1e-3,
    "optimizer": "adamw",
    "scheduler": "plateau",
    "weight_decay": 0.0,
    "early_stopping": {"enabled": True, "patience": 5, "min_delta": 0.0},
    "mixed_precision": True,
    "gradient_clipping": 0.0,
    "gradient_accumulation_steps": 1,
    "checkpointing": {"enabled": False, "dir": None},
    "progress_bar": False,
    # GPU throughput (all no-ops on CPU). TF32 and cuDNN autotuning trade
    # nothing measurable for large kernel speedups on RTX-class GPUs.
    "tf32": True,
    "cudnn_benchmark": True,
    # DataLoader parallelism. Datasets are in-memory tensors, so workers add
    # process-spawn/IPC cost on Windows; keep 0 unless batches are huge.
    "num_workers": 0,
    "prefetch_factor": 2,
    "persistent_workers": False,
}


def _merged_training_config(params: Mapping[str, Any]) -> dict[str, Any]:
    """Overlay the configured ``training`` block onto the defaults."""
    from src.utils.config import deep_merge

    return deep_merge(DEFAULT_TRAINING, dict(params.get("training") or {}))


def _writable_array(x: "Any", dtype: "Any") -> "Any":
    """Return ``x`` as a writable, contiguous ndarray of ``dtype``.

    ``np.asarray`` zero-copies when the input already holds the target dtype
    (e.g. Arrow-backed frames read from parquet), which yields a *read-only*
    view; ``torch.as_tensor`` then warns that non-writable tensors are
    undefined behaviour. Copy only in that case so the common path stays
    zero-copy.
    """
    import numpy as np

    array = np.ascontiguousarray(x, dtype=dtype)
    if not array.flags.writeable:
        array = array.copy()
    return array


def _float32_matrix(x: "Any") -> "Any":
    """Return ``x`` as a writable, contiguous float32 ndarray."""
    import numpy as np

    return _writable_array(x, np.float32)


class TorchModelBase(BaseModel):
    """Base class for all PyTorch Engine B models.

    Subclasses implement :meth:`build_network` only; fitting, prediction,
    device placement and bookkeeping are shared here.
    """

    name = "torch_base"
    is_supervised = True

    @abstractmethod
    def build_network(self, input_dim: int, n_classes: int) -> "Any":
        """Return the ``torch.nn.Module`` for this architecture.

        Parameters
        ----------
        input_dim:
            Number of input features.
        n_classes:
            Number of target classes (output logits).
        """
        raise NotImplementedError

    @property
    def arch_params(self) -> dict[str, Any]:
        """Architecture hyperparameters (``params`` minus the training block)."""
        return {k: v for k, v in self.params.items() if k != "training"}

    @property
    def classes_(self) -> "Any | None":
        """The fitted class labels (set during :meth:`fit`)."""
        return getattr(self, "_classes", None)

    # ------------------------------------------------------------------ fit

    def fit(
        self,
        x_train: "Any",
        y_train: "Any | None" = None,
        x_val: "Any | None" = None,
        y_val: "Any | None" = None,
    ) -> "TorchModelBase":
        """Train the network; validation data drives early stopping/scheduling."""
        import numpy as np
        import torch

        from src.utils.hardware import get_device

        if y_train is None:
            raise ValueError(f"{self.name} is supervised; y_train is required.")

        torch.manual_seed(self.seed)
        training = _merged_training_config(self.params)
        self.device = get_device(prefer_gpu=self.use_gpu)
        if self.use_gpu and self.device == "cpu":
            logger.warning("%s: CUDA unavailable; falling back to CPU.", self.name)
        if self.device == "cuda":
            from src.utils.hardware import configure_cuda_performance

            configure_cuda_performance(
                tf32=bool(training["tf32"]),
                cudnn_benchmark=bool(training["cudnn_benchmark"]),
            )

        x = _float32_matrix(x_train)
        self._classes = np.sort(np.unique(np.asarray(y_train)))
        n_classes = len(self._classes)
        y_idx = np.searchsorted(self._classes, np.asarray(y_train))

        self.model = self.build_network(x.shape[1], n_classes).to(self.device)
        self.fitted_params = {
            "architecture": self.arch_params,
            "training": training,
            "input_dim": int(x.shape[1]),
            "n_classes": int(n_classes),
            "device": self.device,
        }
        logger.info("%s final parameters: %s", self.name, self.fitted_params)

        train_loader = self._loader(x, y_idx, training, shuffle=True)
        val_loader = None
        if x_val is not None and y_val is not None:
            xv = _float32_matrix(x_val)
            yv = np.searchsorted(self._classes, np.asarray(y_val))
            val_loader = self._loader(xv, yv, training, shuffle=False)

        self._train_loop(train_loader, val_loader, training)
        return self

    def _loader(
        self, x: "Any", y_idx: "Any", training: Mapping[str, Any], shuffle: bool
    ) -> "Any":
        """Build a (optionally pinned, seeded-shuffle, parallel) DataLoader."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        import numpy as np

        dataset = TensorDataset(
            torch.as_tensor(_float32_matrix(x), dtype=torch.float32),
            torch.as_tensor(_writable_array(y_idx, np.int64), dtype=torch.long),
        )
        generator = torch.Generator().manual_seed(self.seed) if shuffle else None
        num_workers = max(0, int(training["num_workers"]))
        # prefetch_factor / persistent_workers are only valid with workers.
        worker_kwargs: dict[str, Any] = {}
        if num_workers > 0:
            worker_kwargs = {
                "prefetch_factor": int(training["prefetch_factor"] or 2),
                "persistent_workers": bool(training["persistent_workers"]),
            }
        return DataLoader(
            dataset,
            batch_size=int(training["batch_size"]),
            shuffle=shuffle,
            generator=generator,
            pin_memory=(self.device == "cuda"),
            num_workers=num_workers,
            **worker_kwargs,
        )

    def _train_loop(
        self, train_loader: "Any", val_loader: "Any | None", training: Mapping[str, Any]
    ) -> None:
        """Run the epoch loop with early stopping and best-model restore."""
        import torch

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = self._build_optimizer(training)
        scheduler = self._build_scheduler(optimizer, training)
        use_amp = bool(training["mixed_precision"]) and self.device == "cuda"
        scaler = torch.amp.GradScaler(self.device, enabled=use_amp)

        stopping = dict(training["early_stopping"] or {})
        patience = int(stopping.get("patience", 5))
        min_delta = float(stopping.get("min_delta", 0.0))
        stop_enabled = bool(stopping.get("enabled", True))

        epochs = int(training["epochs"])
        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        bad_epochs = 0
        self.history: list[dict[str, float]] = []

        epoch_iter = self._progress(range(1, epochs + 1), training)
        for epoch in epoch_iter:
            start = time.perf_counter()
            train_loss = self._run_epoch(
                train_loader, criterion, optimizer, scaler, training, use_amp
            )
            val_loss = (
                self._evaluate_loss(val_loader, criterion, use_amp)
                if val_loader is not None else None
            )
            monitored = val_loss if val_loss is not None else train_loss
            self._step_scheduler(scheduler, training, monitored)

            lr = optimizer.param_groups[0]["lr"]
            gpu_mb = (
                torch.cuda.max_memory_allocated() / 1024**2
                if self.device == "cuda" else 0.0
            )
            self.history.append({
                "epoch": epoch, "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6) if val_loss is not None else None,
                "lr": lr, "seconds": round(time.perf_counter() - start, 3),
                "gpu_mb": round(gpu_mb, 1),
            })
            logger.info(
                "[%s] epoch %d/%d | loss=%.5f | val_loss=%s | lr=%.2e | "
                "%.2fs | GPU %.0f MB",
                self.name, epoch, epochs, train_loss,
                f"{val_loss:.5f}" if val_loss is not None else "n/a",
                lr, self.history[-1]["seconds"], gpu_mb,
            )

            if monitored < best_loss - min_delta:
                best_loss = monitored
                bad_epochs = 0
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                self._checkpoint(best_state, epoch, best_loss, training)
            else:
                bad_epochs += 1
                if stop_enabled and bad_epochs >= patience:
                    logger.info(
                        "[%s] early stopping at epoch %d (best %.5f, "
                        "patience %d).", self.name, epoch, best_loss, patience,
                    )
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            logger.info("[%s] restored best model (monitored loss %.5f).",
                        self.name, best_loss)
        self.model.eval()
        self.fitted_params["epochs_trained"] = len(self.history)
        self.fitted_params["best_loss"] = best_loss

    def _run_epoch(
        self, loader: "Any", criterion: "Any", optimizer: "Any",
        scaler: "Any", training: Mapping[str, Any], use_amp: bool,
    ) -> float:
        """One training epoch; returns the mean per-sample loss."""
        import torch

        accum = max(1, int(training["gradient_accumulation_steps"]))
        clip = float(training["gradient_clipping"] or 0.0)
        self.model.train()
        optimizer.zero_grad(set_to_none=True)
        total, count = 0.0, 0

        for step, (xb, yb) in enumerate(loader):
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)
            with torch.amp.autocast(self.device, enabled=use_amp):
                loss = criterion(self.model(xb), yb)
            scaler.scale(loss / accum).backward()

            if (step + 1) % accum == 0 or (step + 1) == len(loader):
                if clip > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total += float(loss.detach()) * len(xb)
            count += len(xb)
        return total / max(count, 1)

    def _evaluate_loss(self, loader: "Any", criterion: "Any", use_amp: bool) -> float:
        """Mean per-sample loss over a validation loader."""
        import torch

        self.model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                with torch.amp.autocast(self.device, enabled=use_amp):
                    loss = criterion(self.model(xb), yb)
                total += float(loss) * len(xb)
                count += len(xb)
        return total / max(count, 1)

    # ------------------------------------------------------- fit components

    def _build_optimizer(self, training: Mapping[str, Any]) -> "Any":
        """Construct the configured optimizer."""
        import torch

        name = str(training["optimizer"]).lower()
        lr = float(training["learning_rate"])
        wd = float(training["weight_decay"])
        factories = {
            "adam": lambda p: torch.optim.Adam(p, lr=lr, weight_decay=wd),
            "adamw": lambda p: torch.optim.AdamW(p, lr=lr, weight_decay=wd),
            "sgd": lambda p: torch.optim.SGD(p, lr=lr, weight_decay=wd, momentum=0.9),
        }
        if name not in factories:
            raise ValueError(
                f"Unknown optimizer '{name}'. Supported: {sorted(factories)}"
            )
        return factories[name](self.model.parameters())

    def _build_scheduler(self, optimizer: "Any", training: Mapping[str, Any]) -> "Any":
        """Construct the configured LR scheduler (or ``None``)."""
        import torch

        name = str(training["scheduler"] or "none").lower()
        if name == "none":
            return None
        if name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=2
            )
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=int(training["epochs"])
            )
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=max(1, int(training["epochs"]) // 3), gamma=0.1
            )
        raise ValueError(
            f"Unknown scheduler '{name}'. Supported: none, plateau, cosine, step"
        )

    def _step_scheduler(
        self, scheduler: "Any", training: Mapping[str, Any], monitored: float
    ) -> None:
        """Advance the scheduler (plateau schedulers consume the loss)."""
        if scheduler is None:
            return
        if str(training["scheduler"]).lower() == "plateau":
            scheduler.step(monitored)
        else:
            scheduler.step()

    def _checkpoint(
        self, state: dict[str, Any], epoch: int, loss: float,
        training: Mapping[str, Any],
    ) -> None:
        """Persist the best state dict when checkpointing is enabled."""
        import torch

        cfg = dict(training["checkpointing"] or {})
        if not cfg.get("enabled") or not cfg.get("dir"):
            return
        target = Path(cfg["dir"]) / f"{self.name}_best.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": epoch, "loss": loss, "state_dict": state}, target)
        logger.debug("[%s] checkpoint written: %s", self.name, target)

    def _progress(self, iterable: "Any", training: Mapping[str, Any]) -> "Any":
        """Wrap the epoch iterator in tqdm when configured and available."""
        if not training.get("progress_bar"):
            return iterable
        try:
            from tqdm import tqdm

            return tqdm(iterable, desc=f"{self.name} epochs", unit="epoch")
        except ImportError:  # pragma: no cover - tqdm is an optional extra
            logger.warning("tqdm not installed; continuing without progress bar.")
            return iterable

    # -------------------------------------------------------------- predict

    def _forward_proba(self, x: "Any") -> "Any":
        """Softmax probabilities for a feature matrix (batched, no grad)."""
        import torch

        matrix = torch.as_tensor(_float32_matrix(x))
        batch_size = int(_merged_training_config(self.params)["batch_size"])
        device = next(self.model.parameters()).device
        outputs = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(matrix), batch_size):
                xb = matrix[start:start + batch_size].to(device, non_blocking=True)
                outputs.append(torch.softmax(self.model(xb), dim=1).cpu())
        return torch.cat(outputs).numpy()

    def predict(self, x: "Any") -> "Any":
        """Return predicted labels (mapped back to the fitted class values)."""
        proba = self._forward_proba(x)
        return self._classes[proba.argmax(axis=1)]

    def predict_proba(self, x: "Any") -> "Any | None":
        """Return class probabilities (columns follow ``classes_`` order)."""
        return self._forward_proba(x)

    # ----------------------------------------------------------------- save

    def save(self, path: "str | Path") -> Path:
        """Persist the wrapper with the network on CPU (portable artefact)."""
        device_backup = None
        if self.model is not None:
            device_backup = next(self.model.parameters()).device
            self.model.to("cpu")
        try:
            return super().save(path)
        finally:
            if device_backup is not None:
                self.model.to(device_backup)

    def describe(self) -> dict[str, Any]:
        """Extend the base description with the training history summary."""
        description = super().describe()
        history = getattr(self, "history", None)
        if history:
            description["epochs_trained"] = len(history)
            description["final_train_loss"] = history[-1]["train_loss"]
        return description
