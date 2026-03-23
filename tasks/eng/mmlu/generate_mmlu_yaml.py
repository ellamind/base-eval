#!/usr/bin/env python3
"""Generate per-subject MMLU task YAMLs from a simple template include."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import NamedTuple


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_NAME_RE = re.compile(r"^dataset_name:\s*([A-Za-z0-9_]+)\s*$")
TASK_LIST_ITEM_RE = re.compile(r"^\s*-\s*([A-Za-z0-9_]+)\s*$")


class TaskSpec(NamedTuple):
    task_base: str
    dataset_name: str


def get_dataset_names_from_hf() -> list[str]:
    """Try to load MMLU config names from Hugging Face."""
    try:
        from datasets import get_dataset_config_names  # type: ignore
    except Exception:
        return []

    try:
        names = get_dataset_config_names("cais/mmlu")
    except Exception:
        return []

    # Some cached/offline setups may return only "default".
    filtered = sorted(n for n in names if n not in {"default", "all"})
    return filtered


def get_dataset_names_from_local_yaml() -> list[str]:
    """Fallback: extract dataset_name values from local MMLU YAML files."""
    names: set[str] = set()
    for path in sorted(SCRIPT_DIR.glob("mmlu_*_*.yaml")):
        for line in path.read_text(encoding="utf-8").splitlines():
            match = DATASET_NAME_RE.match(line.strip())
            if match:
                names.add(match.group(1))
                break
    return sorted(names)


def _extract_dataset_name_from_task_yaml(task_id: str) -> str | None:
    task_path = SCRIPT_DIR / f"{task_id}.yaml"
    if not task_path.exists():
        return None
    for line in task_path.read_text(encoding="utf-8").splitlines():
        match = DATASET_NAME_RE.match(line.strip())
        if match:
            return match.group(1)
    return None


def _resolve_group_file_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    candidate = SCRIPT_DIR / raw_path
    if candidate.exists():
        return candidate.resolve()
    return path.resolve()


def _strip_suffix(task_id: str, suffix: str) -> str:
    if suffix and task_id.endswith(suffix):
        return task_id[: -len(suffix)]
    return task_id


def get_task_specs_from_group_files(group_files: list[str], suffix: str) -> list[TaskSpec]:
    specs: dict[str, str] = {}
    missing_tasks: set[str] = set()

    for raw_path in group_files:
        group_path = _resolve_group_file_path(raw_path)
        if not group_path.exists():
            raise SystemExit(f"Group file not found: {raw_path}")

        in_task_section = False
        for raw_line in group_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()

            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("task:"):
                in_task_section = True
                continue
            if not in_task_section:
                continue

            match = TASK_LIST_ITEM_RE.match(line)
            if not match:
                # End of task list block.
                in_task_section = False
                continue

            task_id = _strip_suffix(match.group(1), suffix)
            dataset_name = _extract_dataset_name_from_task_yaml(task_id)
            if dataset_name:
                specs[task_id] = dataset_name
            else:
                missing_tasks.add(task_id)

    if missing_tasks:
        missing = ", ".join(sorted(missing_tasks))
        raise SystemExit(f"Could not map tasks to dataset_name: {missing}")

    return [
        TaskSpec(task_base=task_base, dataset_name=dataset_name)
        for task_base, dataset_name in sorted(specs.items())
    ]


def render_yaml(
    task_base: str,
    dataset_name: str,
    suffix: str,
    template: str,
    add_description: bool,
) -> str:
    subject = dataset_name.replace("_", " ")
    content = (
        f"include: {template}\n"
        f"task: {task_base}{suffix}\n"
        f"dataset_name: {dataset_name}\n"
    )
    if add_description:
        content += (
            f'description: "The following are multiple choice questions (with answers) '
            f'about {subject}.\\n\\n"\n'
        )
    return content


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MMLU YAML files (e.g. mmlu_*_mc.yaml)."
    )
    parser.add_argument(
        "--suffix",
        required=True,
        help='Task/file suffix, e.g. "_mc", "_rc", "_bpb".',
    )
    parser.add_argument(
        "--template",
        required=True,
        help='Template filename for the include line, e.g. "_mmlu_mc_template.yaml".',
    )
    parser.add_argument(
        "--output-dir",
        "--out-dir",
        "--out_dir",
        default=str(SCRIPT_DIR),
        dest="output_dir",
        help="Output directory (defaults to this script's directory).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files.",
    )
    parser.add_argument(
        "--dataset-names",
        default="",
        help='Optional comma-separated dataset names; bypasses auto-discovery.',
    )
    parser.add_argument(
        "--group-files",
        nargs="+",
        default=[],
        help=(
            "Optional group YAML files with task lists (e.g. _mmlu_other_mc.yaml). "
            "Resolved from current working directory first, then this script's directory."
        ),
    )
    parser.add_argument(
        "--no-hf",
        action="store_true",
        help="Skip Hugging Face lookup and use local YAML discovery.",
    )
    parser.add_argument(
        "--add-description",
        action="store_true",
        help=(
            "Add per-task description: "
            "'The following are multiple choice questions (with answers) about {subject}.'"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """
    Run as:
    python3 tasks/eng/mmlu/generate_mmlu_yaml.py \
    --suffix _mc \             
    --template _mmlu_mc_template.yaml \                                                  
    --group-files _mmlu_group_other_mc.yaml _mmlu_group_social_sciences_mc.yaml _mmlu_group_humanities_mc.yaml _mmlu_group_stem_mc.yaml \                                                                                                      
    --add-description --overwrite
    """
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_names.strip():
        dataset_names = sorted(
            {name.strip() for name in args.dataset_names.split(",") if name.strip()}
        )
        task_specs = [
            TaskSpec(task_base=f"mmlu_{dataset_name}", dataset_name=dataset_name)
            for dataset_name in dataset_names
        ]
        source = "cli"
    elif args.group_files:
        task_specs = get_task_specs_from_group_files(args.group_files, args.suffix)
        source = "group files"
    else:
        dataset_names = [] if args.no_hf else get_dataset_names_from_hf()
        source = "cais/mmlu" if dataset_names else "local yaml fallback"
        if not dataset_names:
            dataset_names = get_dataset_names_from_local_yaml()
        task_specs = [
            TaskSpec(task_base=f"mmlu_{dataset_name}", dataset_name=dataset_name)
            for dataset_name in dataset_names
        ]

    if not task_specs:
        raise SystemExit("No dataset names found.")

    created = 0
    skipped = 0
    for spec in task_specs:
        out_path = output_dir / f"{spec.task_base}{args.suffix}.yaml"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        out_path.write_text(
            render_yaml(
                task_base=spec.task_base,
                dataset_name=spec.dataset_name,
                suffix=args.suffix,
                template=args.template,
                add_description=args.add_description,
            ),
            encoding="utf-8",
        )
        created += 1

    print(f"Dataset source: {source}")
    print(f"Datasets: {len(task_specs)}")
    print(f"Created: {created}")
    print(f"Skipped: {skipped}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
