"""Load and validate version-1 experiment settings without ML dependencies."""

import math
import re
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

ARMS = ("real_only", "synthetic_only", "real_plus_synthetic")
STAGES = ("split", "preprocessing", "diffusion", "generation", "classification", "evaluation")
CLASS_MAPPING = {
    "crazing": 0,
    "inclusion": 1,
    "patches": 2,
    "pitted_surface": 3,
    "rolled_in_scale": 4,
    "scratches": 5,
}


class ConfigError(ValueError):
    """Configuration is malformed or violates the experiment contract."""


class _UniqueKeysLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        result = super().construct_mapping(node, deep=deep)
        if len(result) != len(node.value):
            raise ConfigError("YAML contains duplicate keys")
        return result


def relative_path(value: Any, field: str) -> str:
    """Check a portable relative path before joining it with a local root."""
    if not isinstance(value, str) or any(c in value for c in "\\:\0"):
        raise ConfigError(f"{field}: expected a relative POSIX path")
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ConfigError(f"{field}: expected a relative POSIX path without traversal")
    return value


def _mapping(value: Any, field: str, keys: set[str], optional: tuple[str, ...] = ()) -> dict:
    if not isinstance(value, dict) or not keys <= value.keys() <= keys.union(optional):
        raise ConfigError(
            f"{field}: required keys {sorted(keys)}; optional keys {sorted(optional)}"
        )
    return value


def _integer(value: Any, field: str, minimum: int = 1, maximum: int | None = None) -> None:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ConfigError(
            f"{field}: expected integer in range [{minimum}, {maximum or 'unbounded'}]"
        )


def _number(value: Any, field: str, minimum: float = 0, maximum: float | None = None) -> None:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise ConfigError(f"{field}: expected finite number in range [{minimum}, {maximum}]")


def _text(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field}: expected non-empty text")


def _device(section: dict, name: str) -> None:
    value = section["device"]
    if not isinstance(value, str) or not re.fullmatch(r"cpu|cuda(?::[0-9]+)?|auto", value):
        raise ConfigError(f"{name}.device: expected cpu, auto, cuda or cuda:<index>")


def _training(section: dict, name: str) -> None:
    _device(section, name)
    _integer(section["batch_size"], f"{name}.batch_size")
    for budget in ("max_steps", "epochs"):
        if section[budget] is not None:
            _integer(section[budget], f"{name}.{budget}")
    if (section["max_steps"] is None) == (section["epochs"] is None):
        raise ConfigError(f"{name}: specify exactly one of max_steps and epochs")
    _number(section["learning_rate"], f"{name}.learning_rate", minimum=0)
    if section["learning_rate"] == 0:
        raise ConfigError(f"{name}.learning_rate: must be positive")


def validate_config(config: Any) -> dict:
    """Validate every stage and return the settings without changing them."""
    _mapping(config, "config", {"schema_version", "status", "classes", "paths", *STAGES})
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ConfigError("schema_version: expected 1")
    if config["status"] not in ("proposed", "smoke", "approved"):
        raise ConfigError("status: expected proposed, smoke or approved")
    classes = _mapping(config["classes"], "classes", set(CLASS_MAPPING))
    if classes != CLASS_MAPPING or any(type(value) is not int for value in classes.values()):
        raise ConfigError("classes: expected canonical NEU-CLS labels and IDs")
    paths = _mapping(config["paths"], "paths", {"data_root", "artifact_root", "manifest"})
    for field, value in paths.items():
        relative_path(value, f"paths.{field}")

    split = _mapping(config["split"], "split", {"seed", "ratios", "group_duplicates"})
    _integer(split["seed"], "split.seed", 0, 2**32 - 1)
    ratios = split["ratios"]
    if not isinstance(ratios, list) or len(ratios) != 3:
        raise ConfigError("split.ratios: expected train, validation and test ratios")
    for ratio in ratios:
        _number(ratio, "split.ratios", 0, 1)
    if not all(ratios) or not math.isclose(sum(ratios), 1, abs_tol=1e-9):
        raise ConfigError("split.ratios: positive values must sum to one")
    if split["group_duplicates"] is not True:
        raise ConfigError("split.group_duplicates must be true")

    prep = _mapping(
        config["preprocessing"],
        "preprocessing",
        {
            "diffusion_resolution",
            "classifier_resolution",
            "train_flip_probability",
            "seed",
        },
    )
    for field in ("diffusion_resolution", "classifier_resolution"):
        _integer(prep[field], f"preprocessing.{field}")
    if prep["diffusion_resolution"] % 8:
        raise ConfigError("preprocessing.diffusion_resolution: expected a multiple of 8")
    _number(prep["train_flip_probability"], "preprocessing.train_flip_probability", 0, 1)

    diffusion = _mapping(
        config["diffusion"],
        "diffusion",
        {
            "model_id",
            "revision",
            "seed",
            "device",
            "learning_rate",
            "batch_size",
            "max_steps",
            "epochs",
            "lora_rank",
            "lora_alpha",
            "gradient_accumulation",
            "precision",
        },
        optional=("optimizer", "checkpoint_interval", "gradient_checkpointing"),
    )
    _training(diffusion, "diffusion")
    if diffusion.get("optimizer", "AdamW") not in ("AdamW", "Adam"):
        raise ConfigError("diffusion.optimizer: expected AdamW or Adam")
    _integer(diffusion.get("checkpoint_interval", 250), "diffusion.checkpoint_interval")
    if type(diffusion.get("gradient_checkpointing", True)) is not bool:
        raise ConfigError("diffusion.gradient_checkpointing: expected a boolean")
    _text(diffusion["model_id"], "diffusion.model_id")
    if not isinstance(diffusion["revision"], str) or not re.fullmatch(
        r"[0-9a-f]{40}",
        diffusion["revision"],
    ):
        raise ConfigError("diffusion.revision: expected an immutable 40-character commit SHA")
    for field in ("lora_rank", "lora_alpha", "gradient_accumulation"):
        _integer(diffusion[field], f"diffusion.{field}")
    if diffusion["precision"] not in ("fp32", "fp16", "bf16"):
        raise ConfigError("diffusion.precision: expected fp32, fp16 or bf16")
    if diffusion["device"] == "cpu" and diffusion["precision"] != "fp32":
        raise ConfigError("diffusion.precision: CPU example requires fp32")

    generation = _mapping(
        config["generation"],
        "generation",
        {
            "seed",
            "device",
            "batch_size",
            "target_accepted",
            "minimum_accepted",
            "maximum_candidates",
            "inference_steps",
            "guidance_scale",
        },
    )
    _device(generation, "generation")
    for field in (
        "batch_size",
        "target_accepted",
        "minimum_accepted",
        "maximum_candidates",
        "inference_steps",
    ):
        _integer(generation[field], f"generation.{field}")
    if (
        not generation["minimum_accepted"]
        <= generation["target_accepted"]
        <= generation["maximum_candidates"]
    ):
        raise ConfigError("generation: expected minimum <= target <= maximum")
    _number(generation["guidance_scale"], "generation.guidance_scale")

    classifier = _mapping(
        config["classification"],
        "classification",
        {
            "architecture",
            "weights",
            "seeds",
            "arms",
            "device",
            "batch_size",
            "max_steps",
            "epochs",
            "optimizer",
            "learning_rate",
            "weight_decay",
            "scheduler",
            "validation_interval",
            "mixed_synthetic_fraction",
            "selection_metric",
        },
    )
    _training(classifier, "classification")
    if classifier["architecture"] != "resnet18":
        raise ConfigError("classification.architecture: expected resnet18")
    _text(classifier["weights"], "classification.weights")
    if config["status"] == "approved" and classifier["weights"] == "none":
        raise ConfigError("classification.weights: approved runs require a weight identifier")
    if classifier["arms"] != list(ARMS):
        raise ConfigError("classification.arms: expected all three canonical arms in order")
    seeds = classifier["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise ConfigError("classification.seeds: expected a nonempty list")
    for seed in seeds:
        _integer(seed, "classification.seeds", 0, 2**32 - 1)
    if len(set(seeds)) != len(seeds):
        raise ConfigError("classification.seeds: repeated seed")
    for field in ("optimizer", "scheduler"):
        _text(classifier[field], f"classification.{field}")
    if classifier["selection_metric"] != "macro_f1":
        raise ConfigError("classification.selection_metric: expected macro_f1")
    _integer(classifier["validation_interval"], "classification.validation_interval")
    if classifier["max_steps"] and classifier["validation_interval"] > classifier["max_steps"]:
        raise ConfigError("classification.validation_interval exceeds max_steps")
    fraction = classifier["mixed_synthetic_fraction"]
    _number(fraction, "classification.mixed_synthetic_fraction", 0, 1)
    synthetic = fraction * classifier["batch_size"]
    if not 0 < synthetic < classifier["batch_size"] or not math.isclose(
        synthetic, round(synthetic)
    ):
        raise ConfigError("classification: mixed ratio must produce whole counts for both sources")
    if config["status"] == "approved" and classifier["max_steps"] is None:
        raise ConfigError("classification.max_steps: approved arms require equal update budgets")

    evaluation = _mapping(config["evaluation"], "evaluation", {"seed", "device", "batch_size"})
    _device(evaluation, "evaluation")
    _integer(evaluation["batch_size"], "evaluation.batch_size")
    for name in ("preprocessing", "diffusion", "generation", "evaluation"):
        _integer(config[name]["seed"], f"{name}.seed", 0, 2**32 - 1)
    return config


def load_config(path: Path | str) -> dict:
    """Read YAML or JSON and reject malformed or ambiguous settings."""
    try:
        config = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueKeysLoader)
    except (yaml.YAMLError, RecursionError) as exc:
        raise ConfigError(f"Invalid YAML or JSON in {path}: {exc}") from exc
    return validate_config(config)
