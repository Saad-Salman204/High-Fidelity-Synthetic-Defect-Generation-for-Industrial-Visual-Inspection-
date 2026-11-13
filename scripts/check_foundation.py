"""Validate the planning repository without data, models, network or GPUs."""

import ast
import json
import re
import sys
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlparse

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".git", ".venv", ".cache", "__pycache__", ".ruff_cache", "build", "dist"}


def main() -> int:
    """Fail on broken local file links, invalid configs or cyclic backlog dependencies."""
    errors: list[str] = []
    files = [
        p
        for p in ROOT.rglob("*")
        if p.is_file()
        and not any(part in EXCLUDED or part.startswith(".tmp") for part in p.parts)
        and not any(part.endswith(".egg-info") for part in p.parts)
    ]
    counts = {"markdown": 0, "yaml": 0, "json": 0, "python": 0, "toml": 0}
    for path in files:
        suffix = path.suffix
        if suffix not in {".md", ".yml", ".yaml", ".json", ".py", ".toml"}:
            continue
        try:
            content = path.read_text(encoding="utf-8")
            if suffix == ".md":
                counts["markdown"] += 1
                # Check file targets, not remote URLs or Markdown heading anchors.
                for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", content):
                    target = target.split(' "', 1)[0].strip("<>")
                    if urlparse(target).scheme or target.startswith("#"):
                        continue
                    local = unquote(target.split("#", 1)[0])
                    if local and not (path.parent / local).exists():
                        errors.append(f"{path.relative_to(ROOT)}: missing link {local}")
            elif suffix in {".yaml", ".yml"}:
                list(yaml.safe_load_all(content))
                counts["yaml"] += 1
            elif suffix == ".json":
                json.loads(content)
                counts["json"] += 1
            elif suffix == ".toml":
                tomllib.loads(content)
                counts["toml"] += 1
            elif suffix == ".py":
                ast.parse(content)
                counts["python"] += 1
        except (ValueError, SyntaxError, yaml.YAMLError) as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")

    items = json.loads((ROOT / "docs/planning/issues.json").read_text(encoding="utf-8"))
    graph = {item["id"]: item["dependencies"] for item in items}
    if len(graph) != len(items):
        errors.append("Duplicate issue IDs")
    visited: set[str] = set()
    active: set[str] = set()

    def visit(node: str) -> None:
        if node in active:
            errors.append(f"Dependency cycle: {node}")
            return
        if node in visited:
            return
        if node not in graph:
            errors.append(f"Unknown dependency: {node}")
            return
        active.add(node)
        for dependency in graph[node]:
            visit(dependency)
        active.remove(node)
        visited.add(node)

    headings = [
        "Background",
        "Objective",
        "Scope",
        "Out of scope",
        "Implementation guidance",
        "Dependencies",
        "Deliverables",
        "Acceptance criteria",
        "Testing requirements",
        "Documentation requirements",
        "Planning",
        "Definition of Done",
    ]
    for item in items:
        visit(item["id"])
        body = (ROOT / item["body_path"]).read_text(encoding="utf-8")
        for heading in headings:
            if f"## {heading}" not in body:
                errors.append(f"{item['id']}: missing {heading}")
        if item["week"] not in (1, 2, 3) or item["priority"] not in ("P0", "P1", "P2"):
            errors.append(f"{item['id']}: invalid planning metadata")
        for dependency in item["dependencies"]:
            other = next((x for x in items if x["id"] == dependency), None)
            if other and other["week"] > item["week"]:
                errors.append(f"{item['id']}: depends on a later milestone")

    config = yaml.safe_load((ROOT / "configs/baseline.yaml").read_text(encoding="utf-8"))
    if abs(sum(config["split"]["ratios"]) - 1) > 1e-9:
        errors.append("Split ratios do not sum to 1")
    if sorted(config["classes"].values()) != list(range(6)):
        errors.append("Expected six unique consecutive class IDs")
    if "AGENTS.md" not in (ROOT / "CLAUDE.md").read_text(encoding="utf-8"):
        errors.append("CLAUDE.md must reference canonical policy")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Foundation checks passed: {counts}; {len(items)} issue drafts, acyclic dependencies.")
    print("No ML tests or training executed. Remote links and heading anchors are not checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
