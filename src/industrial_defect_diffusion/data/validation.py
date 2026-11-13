"""Read-only integrity checks for an extracted NEU-CLS collection."""

from collections import defaultdict
from collections.abc import Iterable
from hashlib import sha256
from io import BytesIO
from itertools import combinations
from pathlib import Path, PurePosixPath
from stat import FILE_ATTRIBUTE_REPARSE_POINT
from typing import Any

from PIL import Image

CLASSES = {
    "crazing": 0,
    "inclusion": 1,
    "patches": 2,
    "pitted_surface": 3,
    "rolled_in_scale": 4,
    "scratches": 5,
}
ALIASES = dict(zip(("cr", "in", "pa", "ps", "rs", "sc"), CLASSES, strict=True))
IMAGE_FORMATS = {"BMP", "PNG", "JPEG", "TIFF"}
IMAGE_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _class_name(value: str) -> str | None:
    name = value.lower().replace("-", "_").replace(" ", "_")
    return name if name in CLASSES else ALIASES.get(name)


def _filename_class(stem: str) -> str | None:
    prefix, separator, index = stem.rpartition("_")
    return _class_name(prefix) if separator and index.isdecimal() else None


def _difference_hash(image: Image.Image) -> int:
    """Encode horizontal brightness changes as 64 bits."""
    pixels = list(image.resize((9, 8), Image.Resampling.LANCZOS).get_flattened_data())
    value = 0
    for row in range(8):
        for col in range(8):
            offset = row * 9 + col
            value = (value << 1) | (pixels[offset] > pixels[offset + 1])
    return value


def _inspect_file(path: Path, relative: str) -> dict[str, Any]:
    parts = PurePosixPath(relative).parts
    folder_label = _class_name(parts[0]) if len(parts) == 2 else None
    file_label = _filename_class(path.stem)
    label = folder_label if len(parts) == 2 else file_label if len(parts) == 1 else None
    problems: list[str] = []
    record = {
        "relative_path": relative,
        "class_name": label,
        "class_id": CLASSES.get(label) if label else None,
        "sha256": None,
        "dhash": None,
        "mode": None,
        "width": None,
        "height": None,
        "problems": problems,
    }
    if len(parts) > 2 or (len(parts) == 2 and folder_label is None):
        problems.append("unexpected_folder_structure")
    if label is None:
        problems.append("unknown_label")
    if folder_label and file_label and folder_label != file_label:
        problems.append("conflicting_label")
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        problems.append("unexpected_file")
        return record

    try:
        with path.open("rb") as stream:
            payload = stream.read(8 * 1024 * 1024 + 1)
        if len(payload) > 8 * 1024 * 1024:
            problems.append("image_file_too_large")
            return record
        record["sha256"] = sha256(payload).hexdigest()
        with Image.open(BytesIO(payload), formats=sorted(IMAGE_FORMATS)) as image:
            record.update(mode=image.mode, width=image.width, height=image.height)
            if image.mode != "L":
                problems.append("expected_grayscale_L")
            if image.size != (200, 200):
                problems.append("expected_200x200")
            if getattr(image, "n_frames", 1) != 1:
                problems.append("expected_single_image")
            if image.width * image.height > 4_000_000:
                problems.append("image_pixel_limit")
            else:
                image.load()
                if image.mode == "L" and not problems:
                    record["dhash"] = f"{_difference_hash(image):016x}"
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        problems.append("unreadable_image")
    return record


def _expected_paths(paths: Iterable[str]) -> set[str]:
    expected = set()
    for path in paths:
        parsed = PurePosixPath(path)
        if (
            parsed.is_absolute()
            or not parsed.parts
            or ".." in parsed.parts
            or parsed.as_posix() != path
            or "\\" in path
            or ":" in path
        ):
            raise ValueError(f"Invalid inventory path: {path!r}")
        if path in expected:
            raise ValueError(f"Duplicate inventory path: {path!r}")
        expected.add(path)
    return expected


def _duplicate_ledger(files: list[dict[str, Any]], threshold: int) -> tuple[list, list]:
    by_sha: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in files:
        if record["sha256"] is not None:
            by_sha[record["sha256"]].append(record)
    exact = []
    for digest, members in by_sha.items():
        if len(members) < 2:
            continue
        labels = {member["class_name"] for member in members} - {None}
        conflict = len(labels) > 1
        exact.append(
            {
                "sha256": digest,
                "paths": [member["relative_path"] for member in members],
                "conflicting_labels": conflict,
                "review_status": "pending",
            }
        )
        for member in members:
            member["problems"].append("exact_duplicate")
            if conflict and "conflicting_label" not in member["problems"]:
                member["problems"].append("conflicting_label")

    hashes = [(row, int(row["dhash"], 16)) for row in files if row["dhash"] is not None]
    near = []
    for (left, first), (right, second) in combinations(hashes, 2):
        distance = (first ^ second).bit_count()
        if left["sha256"] != right["sha256"] and distance <= threshold:
            near.append(
                {
                    "paths": [left["relative_path"], right["relative_path"]],
                    "distance": distance,
                    "different_labels": bool(
                        left["class_name"]
                        and right["class_name"]
                        and left["class_name"] != right["class_name"]
                    ),
                    "review_status": "pending",
                }
            )
            for row in (left, right):
                if "near_duplicate_candidate" not in row["problems"]:
                    row["problems"].append("near_duplicate_candidate")
    return exact, near


def validate_dataset(
    root: Path | str,
    *,
    expected_paths: Iterable[str] | None = None,
    near_threshold: int = 5,
) -> dict[str, Any]:
    """Audit local source files and return a JSON-ready report without editing them."""
    if not 0 <= near_threshold <= 64:
        raise ValueError("near_threshold must be between 0 and 64")
    source = Path(root)
    if not source.is_dir():
        raise NotADirectoryError(f"Dataset root does not exist or is not a directory: {source}")
    expected = _expected_paths(expected_paths) if expected_paths is not None else None
    files = []
    structure = []

    def scan(folder: Path) -> None:
        for path in sorted(folder.iterdir(), key=lambda item: item.name):
            relative = path.relative_to(source).as_posix()
            attributes = getattr(path.lstat(), "st_file_attributes", 0)
            if path.is_symlink() or attributes & FILE_ATTRIBUTE_REPARSE_POINT:
                structure.append({"path": relative, "problem": "linked_entry_not_followed"})
            elif path.is_dir():
                if folder != source or _class_name(path.name) is None:
                    structure.append({"path": relative, "problem": "unexpected_directory"})
                scan(path)
            elif path.is_file():
                files.append(_inspect_file(path, relative))
            else:
                structure.append({"path": relative, "problem": "unsupported_entry"})

    scan(source)
    files.sort(key=lambda row: row["relative_path"])
    exact, near = _duplicate_ledger(files, near_threshold)
    observed = {row["relative_path"] for row in files}
    missing = sorted(expected - observed) if expected is not None else []
    if expected is not None:
        for row in files:
            if row["relative_path"] not in expected:
                row["problems"].append("unexpected_inventory_file")

    counts = dict.fromkeys(CLASSES, 0)
    image_total = 0
    for row in files:
        if "unexpected_file" not in row["problems"]:
            image_total += 1
            if row["class_name"] in counts:
                counts[row["class_name"]] += 1
        problems = set(row["problems"])
        review_only = {"exact_duplicate", "near_duplicate_candidate"}
        row["disposition"] = (
            "proposed_exclusion"
            if problems - review_only
            else "needs_review"
            if problems
            else "valid"
        )
    return {
        "schema_version": 1,
        "class_mapping": CLASSES.copy(),
        "expected_images": 1800,
        "expected_per_class": 300,
        "image_file_count": image_total,
        "class_counts": counts,
        "count_matches": image_total == 1800 and all(count == 300 for count in counts.values()),
        "missing_count_by_class": {label: max(0, 300 - count) for label, count in counts.items()},
        "inventory_checked": expected is not None,
        "missing_paths": missing,
        "structure_findings": structure,
        "perceptual_hash": {
            "algorithm": "horizontal_dhash_64",
            "max_hamming_distance": near_threshold,
        },
        "files": files,
        "exact_duplicate_groups": exact,
        "near_duplicate_pairs": near,
        "status": "review_required"
        if (
            image_total != 1800
            or any(count != 300 for count in counts.values())
            or missing
            or structure
            or any(row["problems"] for row in files)
        )
        else "passed",
    }


def format_summary(report: dict[str, Any]) -> str:
    """Summarize counts and outstanding decisions for a human reviewer."""
    exclusions = sum(row["disposition"] == "proposed_exclusion" for row in report["files"])
    lines = [
        f"Audit status: {report['status']}",
        f"Images found: {report['image_file_count']} / {report['expected_images']}",
        *(f"  {label}: {count} / 300" for label, count in report["class_counts"].items()),
        f"Proposed exclusions: {exclusions}",
        f"Exact duplicate groups: {len(report['exact_duplicate_groups'])}",
        f"Near-duplicate candidates: {len(report['near_duplicate_pairs'])}",
        f"Folder/entry findings: {len(report['structure_findings'])}",
    ]
    if report["inventory_checked"]:
        lines.append(f"Missing inventory paths: {len(report['missing_paths'])}")
    else:
        lines.append(
            "No reference inventory supplied. Counts reveal shortages, not exact missing paths."
        )
    return "\n".join(lines)
