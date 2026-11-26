"""Safe, serialized adapters around the project's existing inference code."""

from __future__ import annotations

import csv
import json
import threading
import uuid
from contextlib import contextmanager
from hashlib import file_digest
from pathlib import Path

import torch
from PIL import Image

from industrial_defect_diffusion.classification.baseline import load_checkpoint, resolve_device
from industrial_defect_diffusion.data.preprocessing import (
    preprocess_image,
    read_frozen_manifest,
)
from industrial_defect_diffusion.diffusion.training import PROMPTS
from industrial_defect_diffusion.generation.sampling import make_sampler, verify_checkpoint
from industrial_defect_diffusion.utils.experiment_config import (
    ARMS,
    CLASS_MAPPING,
    load_config,
)

ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ROOT = ROOT / "artifacts" / "evaluation" / "phase1-final-20261126"
PACKAGED_RESEARCH_ROOT = ROOT / "presentation" / "web" / "public" / "research"
PACKAGED_GALLERY_ROOT = ROOT / "presentation" / "web" / "public" / "demo-gallery"
SPLIT_MANIFEST = ROOT / "artifacts" / "neu-split-v1" / "manifest.csv"
CONFIG_PATH = ROOT / "artifacts" / "neu-lora-resume-1000.yaml"
GENERATION_ROOT = ROOT / "artifacts" / "synthetic" / "neu-sd15-step-001000"
REVIEW_SELECTION = (
    ROOT / "artifacts" / "synthetic" / "review-contact-sheet-seed42" / "review-selection.json"
)
DEMO_ROOT = ROOT / "artifacts" / "demo" / "presentation"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP"}
FIGURES = {
    f"{stem}.{suffix}"
    for stem in (
        "01_experimental_design",
        "02_accuracy_by_seed",
        "03_macro_f1_by_seed",
        "04_precision_recall",
        "05_mean_sd_all_metrics",
        "06_confusion_real_only",
        "06_confusion_synthetic_only",
        "06_confusion_real_plus_synthetic",
        "09_per_class_f1",
        "10_pairwise_macro_f1_differences",
    )
    for suffix in ("png", "pdf")
}


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


class ProjectRuntime:
    """Lazily load one GPU model at a time and call shared project inference code."""

    def __init__(self) -> None:
        self._config: dict | None = None
        self.gpu_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.active_task: str | None = None
        self.waiting_jobs = 0
        self.active_kind: str | None = None
        self.sampler = None
        self.classifier = None
        self.classifier_key: tuple[str, int] | None = None
        self._gallery_records: list[dict] | None = None

    @property
    def config(self) -> dict:
        """Load configuration only when an endpoint needs local run artifacts."""
        if self._config is None:
            self._config = load_config(CONFIG_PATH)
        return self._config

    @property
    def gallery_records(self) -> list[dict]:
        """Load the fixed contact-sheet sample index lazily."""
        if self._gallery_records is None:
            self._gallery_records = self._read_gallery_records()
        return self._gallery_records

    @contextmanager
    def _task(self, name: str):
        with self.state_lock:
            self.waiting_jobs += 1
        self.gpu_lock.acquire()
        with self.state_lock:
            self.waiting_jobs -= 1
        with self.state_lock:
            self.active_task = name
        try:
            yield
        except torch.cuda.OutOfMemoryError as exc:
            self._release_models()
            raise RuntimeError(
                "GPU memory ran out. The loaded model was released; wait briefly and retry."
            ) from exc
        finally:
            with self.state_lock:
                self.active_task = None
            self.gpu_lock.release()

    def _release_models(self) -> None:
        self.sampler = None
        self.classifier = None
        self.classifier_key = None
        self.active_kind = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _switch_model_kind(self, kind: str) -> None:
        if self.active_kind == kind:
            return
        self._release_models()
        self.active_kind = kind

    def health(self) -> dict:
        with self.state_lock:
            active = self.active_task
            waiting = self.waiting_jobs
        cuda = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda else None
        return {
            "status": "ready",
            "busy": active is not None or waiting > 0,
            "active_task": active,
            "queued_jobs": waiting,
            "device": "cuda:0" if cuda else "cpu",
            "gpu_name": gpu_name,
            "cuda_available": cuda,
        }

    def _checkpoint(self, arm: str, seed: int) -> tuple[Path, dict]:
        if arm not in ARMS or seed not in self.config["classification"]["seeds"]:
            raise ValueError("Choose one of the configured classifier arms and seeds")
        matched = []
        for run_path in (ROOT / "artifacts" / "classification").glob("*/run.json"):
            record = json.loads(run_path.read_text(encoding="utf-8"))
            if record.get("arm") == arm and record.get("seed") == seed:
                matched.append((run_path, record))
        if len(matched) != 1:
            raise FileNotFoundError(f"Expected one trained checkpoint for {arm}, seed {seed}")
        run_path, record = matched[0]
        if record.get("status") != "completed":
            raise ValueError("The selected classifier run is not complete")
        artifact = next(
            (item for item in record.get("artifacts", []) if Path(item["path"]).name == "best.pt"),
            None,
        )
        if artifact is None:
            raise FileNotFoundError("The selected run has no recorded best.pt checkpoint")
        checkpoint = (ROOT / artifact["path"]).resolve()
        if not checkpoint.is_relative_to(ROOT) or not checkpoint.is_file():
            raise FileNotFoundError("The selected checkpoint is missing or outside this project")
        if _digest(checkpoint) != artifact["sha256"]:
            raise ValueError("The selected checkpoint SHA-256 does not match its run record")
        return checkpoint, record

    def model_availability(self) -> dict:
        try:
            config = self.config
        except (OSError, ValueError) as exc:
            return {
                "classifiers": [
                    {
                        "arm": arm,
                        "seed": seed,
                        "available": False,
                        "reason": "Local experiment configuration and checkpoints are unavailable.",
                    }
                    for arm in ARMS
                    for seed in (42, 43, 44)
                ],
                "generator": {
                    "available": False,
                    "reason": f"Local experiment configuration is unavailable: {exc}",
                },
            }
        models = []
        for arm in ARMS:
            for seed in config["classification"]["seeds"]:
                try:
                    checkpoint, record = self._checkpoint(arm, seed)
                    models.append(
                        {
                            "arm": arm,
                            "seed": seed,
                            "available": True,
                            "run_id": record["run_id"],
                            "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
                            "checkpoint_sha256": next(
                                item["sha256"]
                                for item in record["artifacts"]
                                if Path(item["path"]).name == "best.pt"
                            ),
                        }
                    )
                except (OSError, ValueError, KeyError, StopIteration) as exc:
                    models.append(
                        {"arm": arm, "seed": seed, "available": False, "reason": str(exc)}
                    )
        model_dir = ROOT / "models" / "sd-v1-5"
        adapter_dir = self._resolve_adapter_dir()
        return {
            "classifiers": models,
            "generator": {
                "available": model_dir.is_dir() and adapter_dir is not None,
                "base_model": config["diffusion"]["model_id"],
                "base_revision": config["diffusion"]["revision"],
                "local_snapshot": model_dir.relative_to(ROOT).as_posix(),
                "adapter": (
                    adapter_dir.relative_to(ROOT).as_posix() if adapter_dir is not None else None
                ),
                "adapter_checkpoint": "step-001000",
            },
        }

    def _resolve_adapter_dir(self) -> Path | None:
        plan_path = GENERATION_ROOT / "plan.json"
        if not plan_path.is_file():
            return None
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        expected_hash = plan.get("source", {}).get("adapter_checkpoint_sha256")
        matches = []
        for checkpoint_file in (ROOT / "artifacts" / "diffusion").glob(
            "*/step-001000/checkpoint.json"
        ):
            if _digest(checkpoint_file) == expected_hash:
                matches.append(checkpoint_file.parent)
        return matches[0] if len(matches) == 1 else None

    def _create_sampler(self):
        model_dir = ROOT / "models" / "sd-v1-5"
        adapter_dir = self._resolve_adapter_dir()
        if adapter_dir is None:
            raise FileNotFoundError("Verified step-001000 LoRA adapter is unavailable")
        _, metadata = read_frozen_manifest(
            SPLIT_MANIFEST, SPLIT_MANIFEST.with_name("split-metadata.json")
        )
        plan = json.loads((GENERATION_ROOT / "plan.json").read_text(encoding="utf-8"))
        source = verify_checkpoint(
            model_dir,
            adapter_dir,
            self.config,
            metadata,
            plan["source"]["adapter_checkpoint_sha256"],
        )
        if source["real_manifest_sha256"] != plan["source"]["real_manifest_sha256"]:
            raise ValueError("LoRA checkpoint does not match the generation run's frozen split")
        device = resolve_device(self.config["generation"]["device"])
        return make_sampler(
            model_dir,
            adapter_dir,
            source,
            device,
            self.config["generation"].get("precision", "fp32"),
        ), source

    def generate(self, class_name: str, seed: int) -> dict:
        if class_name not in CLASS_MAPPING or type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Choose a supported defect class and an unsigned 32-bit seed")
        with self._task("Generating synthetic image"):
            self._switch_model_kind("diffusion")
            if self.sampler is None:
                self.sampler, self.generator_provenance = self._create_sampler()
            settings = self.config["generation"]
            resolution = self.config["preprocessing"]["diffusion_resolution"]
            class_id = CLASS_MAPPING[class_name]
            prompt = PROMPTS[class_id]
            row = {
                "prompt": prompt,
                "negative_prompt": settings.get("negative_prompt", ""),
                "seed": seed,
                "height": resolution,
                "width": resolution,
                "inference_steps": settings["inference_steps"],
                "guidance_scale": settings["guidance_scale"],
            }
            image = self.sampler([row])[0].convert("RGB")
            image_id = uuid.uuid4().hex
            output = DEMO_ROOT / "generated"
            output.mkdir(parents=True, exist_ok=True)
            image_path = output / f"{image_id}.png"
            temporary = output / f".{image_id}.tmp.png"
            image.save(temporary, format="PNG")
            temporary.replace(image_path)
            digest = _digest(image_path)
            metadata = {
                "image_id": image_id,
                "class_name": class_name,
                "class_id": class_id,
                "seed": seed,
                "prompt": prompt,
                "negative_prompt": row["negative_prompt"],
                "base_model_id": self.generator_provenance["base_model_id"],
                "base_revision": self.generator_provenance["base_revision"],
                "adapter_sha256": self.generator_provenance["adapter_sha256"],
                "adapter_checkpoint_sha256": self.generator_provenance["adapter_checkpoint_sha256"],
                "real_manifest_sha256": self.generator_provenance["real_manifest_sha256"],
                "inference_steps": row["inference_steps"],
                "guidance_scale": row["guidance_scale"],
                "sha256": digest,
                "source": "live_demo_only_not_added_to_candidate_dataset",
            }
            metadata_path = output / f"{image_id}.json"
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            return {**metadata, "image_url": f"/api/demo/images/{image_id}"}

    def classify(self, image: Image.Image, arm: str, seed: int) -> dict:
        image = image.convert("RGB")
        if image.width < 16 or image.height < 16 or image.width * image.height > MAX_IMAGE_PIXELS:
            raise ValueError("Image dimensions must be at least 16 px and below 40 megapixels")
        if arm not in ARMS:
            raise ValueError("Unknown classifier arm")
        with self._task("Classifying image"):
            self._switch_model_kind("classifier")
            if self.classifier is None or self.classifier_key != (arm, seed):
                checkpoint, _record = self._checkpoint(arm, seed)
                device = resolve_device(self.config["evaluation"]["device"])
                self.classifier = load_checkpoint(checkpoint, device)
                self.classifier_key = (arm, seed)
            device = next(self.classifier.parameters()).device
            tensor = (
                preprocess_image(
                    image,
                    "classifier",
                    resolution=self.config["preprocessing"]["classifier_resolution"],
                )
                .unsqueeze(0)
                .to(device)
            )
            probabilities = self.classifier(tensor).softmax(dim=1)[0].cpu().tolist()
            predicted_id = max(range(len(probabilities)), key=probabilities.__getitem__)
            labels = sorted(CLASS_MAPPING, key=CLASS_MAPPING.__getitem__)
            prediction = labels[predicted_id]
            return {
                "predicted_class": prediction,
                "predicted_class_id": predicted_id,
                "confidence": probabilities[predicted_id],
                "probabilities": [
                    {"class_name": label, "class_id": CLASS_MAPPING[label], "score": score}
                    for label, score in zip(labels, probabilities, strict=True)
                ],
                "arm": arm,
                "seed": seed,
                "model_name": f"ResNet-18 {arm.replace('_', ' ')} · seed {seed}",
            }

    def _read_gallery_records(self) -> list[dict]:
        if not REVIEW_SELECTION.is_file() or not (GENERATION_ROOT / "candidates.jsonl").is_file():
            packaged_index = PACKAGED_GALLERY_ROOT / "selection.json"
            if not packaged_index.is_file():
                return []
            selection = json.loads(packaged_index.read_text(encoding="utf-8"))
            return [{**row, "_bundled": True} for row in selection.get("candidates", [])]
        generation_rows = {
            row["image_id"]: row
            for row in (
                json.loads(line)
                for line in (GENERATION_ROOT / "candidates.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            )
        }
        selection = json.loads(REVIEW_SELECTION.read_text(encoding="utf-8"))
        records = []
        for row in selection.get("candidates", []):
            source = generation_rows.get(row.get("image_id"))
            if not source or any(
                row.get(key) != source.get(key) for key in ("class_name", "relative_path", "sha256")
            ):
                continue
            records.append(row)
        return records

    def gallery(self, limit_per_class: int = 3) -> list[dict]:
        if not 1 <= limit_per_class <= 12:
            raise ValueError("Gallery limit must be between 1 and 12 per class")
        counts = {name: 0 for name in CLASS_MAPPING}
        output = []
        for row in self.gallery_records:
            label = row["class_name"]
            if counts[label] >= limit_per_class:
                continue
            image_path = self._gallery_path(row)
            if image_path is None:
                continue
            if _digest(image_path) != row["sha256"]:
                continue
            counts[label] += 1
            output.append(
                {
                    "image_id": row["image_id"],
                    "class_name": label,
                    "class_id": CLASS_MAPPING[label],
                    "image_url": f"/api/gallery/images/{row['image_id']}",
                    "human_review_status": (
                        "sampled_for_contact_sheet_only_pending_candidate_review"
                    ),
                }
            )
        return output

    def gallery_image(self, image_id: str) -> Path:
        row = next((item for item in self.gallery_records if item["image_id"] == image_id), None)
        if row is None:
            raise FileNotFoundError("Unknown demonstration image")
        image_path = self._gallery_path(row)
        if image_path is None or not image_path.is_file() or _digest(image_path) != row["sha256"]:
            raise ValueError("Demonstration image failed its recorded integrity check")
        return image_path

    @staticmethod
    def _gallery_path(row: dict) -> Path | None:
        root = PACKAGED_GALLERY_ROOT if row.get("_bundled") else GENERATION_ROOT
        path = (root / row["relative_path"]).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            return None
        return path

    def research_results(self) -> dict:
        result_root = EVALUATION_ROOT if EVALUATION_ROOT.is_dir() else PACKAGED_RESEARCH_ROOT
        report = json.loads((result_root / "evaluation.json").read_text(encoding="utf-8"))
        class_rows = list(
            csv.DictReader((result_root / "per-class-aggregates.csv").open(encoding="utf-8"))
        )
        pairwise = list(
            csv.DictReader((result_root / "pairwise-seed-differences.csv").open(encoding="utf-8"))
        )
        comparisons = list(csv.DictReader((result_root / "comparison.csv").open(encoding="utf-8")))
        return {
            "status": report["status"],
            "manifest_sha256": report["manifest_sha256"],
            "test_membership_sha256": report["test_membership_sha256"],
            "test_count": report["test_count"],
            "split_counts": report["split_counts"],
            "class_counts_by_split": report["class_counts_by_split"],
            "device": report["device"],
            "torch": report["torch"],
            "aggregates": report["aggregates"],
            "runs": [
                {
                    "arm": run["arm"],
                    "seed": run["seed"],
                    "run_id": run["run_id"],
                    "checkpoint_sha256": run["checkpoint_sha256"],
                    "metrics": run["metrics"],
                }
                for run in report["runs"]
            ],
            "class_aggregates": class_rows,
            "pairwise_seed_differences": pairwise,
            "comparison_rows": comparisons,
            "synthetic_selection": report["synthetic_selection"],
            "figures": sorted(FIGURES),
        }


runtime = ProjectRuntime()
