"""Six-class ResNet-18 baseline using frozen real train and validation rows."""

import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18

NUM_CLASSES = 6


def build_model(weights: str, local_weights: Path | None = None) -> nn.Module:
    """Build a model without network access, using local weights only when supplied."""
    if weights == "none" and local_weights is None:
        return resnet18(weights=None, num_classes=NUM_CLASSES)
    if weights != "IMAGENET1K_V1" or local_weights is None:
        raise ValueError("IMAGENET1K_V1 requires an explicitly approved local weights file")
    model = resnet18(weights=None)
    state = torch.load(local_weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> nn.Module:
    """Load a saved baseline model for inference."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["schema_version"] != 1 or checkpoint["architecture"] != "resnet18":
        raise ValueError("Unsupported baseline checkpoint")
    model = build_model("none")
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval()


def seed_training(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch before constructing loaders or model."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str) -> torch.device:
    """Resolve an explicit device and reject unavailable CUDA selections."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and (
        not torch.cuda.is_available()
        or (device.index is not None and device.index >= torch.cuda.device_count())
    ):
        raise ValueError(f"Requested device is unavailable: {name}")
    return device


def _scores(confusion: torch.Tensor) -> dict:
    true = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    correct = confusion.diag()
    precision = correct / predicted.clamp_min(1)
    recall = correct / true.clamp_min(1)
    f1 = 2 * correct / (true + predicted).clamp_min(1)
    return {
        "accuracy": correct.sum().item() / confusion.sum().item(),
        "macro_precision": precision.mean().item(),
        "macro_recall": recall.mean().item(),
        "macro_f1": f1.mean().item(),
        "per_class": [
            {
                "class_id": i,
                "precision": precision[i].item(),
                "recall": recall[i].item(),
                "f1": f1[i].item(),
                "support": true[i].item(),
            }
            for i in range(NUM_CLASSES)
        ],
        "confusion_matrix": confusion.tolist(),
    }


@torch.inference_mode()
def validate_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Score one full real validation pass in stable order."""
    if getattr(loader.dataset, "split", None) != "val":
        raise ValueError("Validation requires the real validation partition")
    model.eval()
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    loss_total = 0.0
    for batch in loader:
        labels = batch["label"].to(device)
        scores = model(batch["image"].to(device))
        loss = nn.functional.cross_entropy(scores, labels, reduction="sum").item()
        if not math.isfinite(loss):
            raise ValueError("Non-finite validation loss")
        loss_total += loss
        indices = (labels * NUM_CLASSES + scores.argmax(1)).cpu()
        confusion += torch.bincount(indices, minlength=NUM_CLASSES**2).reshape_as(confusion)
    return {"loss": loss_total / confusion.sum().item(), **_scores(confusion)}


def fit_baseline(
    model: nn.Module,
    train: DataLoader,
    validation: DataLoader,
    settings: dict,
    device: torch.device,
    checkpoint: Path,
    *,
    on_validation: Callable[[dict], None] | None = None,
) -> dict:
    """Train using real train/val only and save the earliest best validation checkpoint."""
    if (
        getattr(train.dataset, "split", None) != "train"
        or getattr(validation.dataset, "split", None) != "val"
    ):
        raise ValueError("Baseline requires train and val partitions, in that order")
    if settings["optimizer"] != "AdamW" or settings["scheduler"] != "cosine":
        raise ValueError("Baseline supports the protocol AdamW optimizer and cosine scheduler")
    if not len(train) or not len(validation):
        raise ValueError("Training and validation loaders must be nonempty")
    steps = settings["max_steps"] or settings["epochs"] * len(train)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    interval = settings["validation_interval"]
    history = []
    best = None
    update = 0
    train_loss = 0.0
    count = 0
    exposures = 0
    unique_images = set()
    source_counts = Counter()
    class_counts = defaultdict(Counter)
    source_images = defaultdict(set)
    checkpoint.touch(exist_ok=False)
    while update < steps:
        for batch in train:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            labels = batch["label"].to(device)
            scores = model(batch["image"].to(device))
            loss = nn.functional.cross_entropy(scores, labels)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            scheduler.step()
            update += 1
            count += len(labels)
            exposures += len(labels)
            unique_images.update(batch["image_id"])
            for source, label, image_id in zip(
                batch.get("source", ["real"] * len(labels)),
                batch["label"].tolist(),
                batch["image_id"],
                strict=True,
            ):
                source_counts[source] += 1
                class_counts[source][label] += 1
                source_images[source].add(image_id)
            train_loss += loss.item() * len(labels)
            if update % interval == 0 or update == steps:
                metrics = validate_epoch(model, validation, device)
                entry = {
                    "update": update,
                    "train_loss": train_loss / count,
                    "validation": metrics,
                    "next_learning_rate": scheduler.get_last_lr()[0],
                }
                history.append(entry)
                if on_validation is not None:
                    on_validation(entry)
                train_loss, count = 0.0, 0
                if best is None or metrics["macro_f1"] > best["validation"]["macro_f1"]:
                    best = entry
                    temporary = checkpoint.with_suffix(".tmp")
                    torch.save(
                        {
                            "schema_version": 1,
                            "architecture": "resnet18",
                            "update": update,
                            "model_state": model.state_dict(),
                        },
                        temporary,
                    )
                    temporary.replace(checkpoint)
            if update == steps:
                break
    return {
        "updates": update,
        "best": best,
        "history": history,
        "training_exposures": exposures,
        "unique_training_images": len(unique_images),
        "repeated_training_exposures": exposures - len(unique_images),
        "training_by_source": {
            source: {
                "exposures": total,
                "unique_images": len(source_images[source]),
                "repeated_exposures": total - len(source_images[source]),
                "class_exposures": {str(i): class_counts[source][i] for i in range(NUM_CLASSES)},
            }
            for source, total in sorted(source_counts.items())
        },
    }
