"""Asset discovery, path helpers, and Artiverse articulation metadata."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = Path(r"D:\Datasets\Artiverse\dataset_chunks\data")
USD_ASSET_SUFFIXES = frozenset({".usd", ".usda", ".usdc", ".usdz"})
SUPPORTED_ASSET_SUFFIXES = frozenset({".urdf", ".glb", *USD_ASSET_SUFFIXES})


@dataclass(frozen=True)
class ArticulationRecord:
    pid: int
    type: str
    base: tuple[int, ...]
    range_min: float | None
    range_max: float | None
    motion_states: dict[str, Any]
    raw: dict[str, Any]


@dataclass(frozen=True)
class ArticulationMetadata:
    path: Path | None
    model_id: str | None
    records_by_pid: dict[int, ArticulationRecord]
    explicit_dependencies: list[tuple[str, Any]]


def find_articulation_json(urdf_path: Path) -> Path | None:
    model_dir = urdf_path.parent.parent if urdf_path.parent.name == "urdf_w_collider" else urdf_path.parent
    candidates = [
        model_dir / f"{urdf_path.stem}.articulations.json",
        *sorted(model_dir.glob("*.articulations.json")),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def collect_explicit_dependencies(data: Any) -> list[tuple[str, Any]]:
    dependency_tokens = ("depend", "functional", "trigger", "interaction", "action", "relation")
    ignored_keys = {"articulations", "connectivity", "graphParentByPid", "motionStates"}
    found: list[tuple[str, Any]] = []

    def walk(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                key_text = str(key)
                key_lower = key_text.lower()
                child_path = (*path, key_text)

                if key_text not in ignored_keys and any(token in key_lower for token in dependency_tokens):
                    found.append((".".join(child_path), child))

                walk(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, (*path, str(index)))

    walk(data, ())
    return found


def load_articulation_metadata(urdf_path: Path) -> ArticulationMetadata:
    json_path = find_articulation_json(urdf_path)
    if json_path is None:
        return ArticulationMetadata(
            path=None,
            model_id=None,
            records_by_pid={},
            explicit_dependencies=[],
        )

    data = json.loads(json_path.read_text(encoding="utf-8"))
    records: dict[int, ArticulationRecord] = {}

    for row in data.get("articulations", []):
        if not isinstance(row, dict) or "pid" not in row:
            continue

        pid = int(row["pid"])
        base = row.get("base") or ()
        records[pid] = ArticulationRecord(
            pid=pid,
            type=str(row.get("type", "")),
            base=tuple(int(item) for item in base),
            range_min=float(row["rangeMin"]) if "rangeMin" in row else None,
            range_max=float(row["rangeMax"]) if "rangeMax" in row else None,
            motion_states=row.get("motionStates") if isinstance(row.get("motionStates"), dict) else {},
            raw=row,
        )

    return ArticulationMetadata(
        path=json_path,
        model_id=data.get("modelId"),
        records_by_pid=records,
        explicit_dependencies=collect_explicit_dependencies(data),
    )


def unique_sorted_paths(paths: list[Path]) -> list[Path]:
    unique = {path.resolve(): path for path in paths}
    return sorted(unique.values(), key=lambda path: str(path).lower())


def model_dir_from_urdf(urdf_path: Path) -> Path:
    if urdf_path.parent.name == "urdf_w_collider":
        return urdf_path.parent.parent
    return urdf_path.parent


def describe_urdf_path(urdf_path: Path) -> str:
    model_dir = model_dir_from_urdf(urdf_path)
    provider = model_dir.parent.name if model_dir.parent != model_dir else ""
    category = model_dir.parent.parent.name if model_dir.parent.parent != model_dir.parent else ""

    if category and provider:
        return f"{category}/{provider}/{model_dir.name}"
    if provider:
        return f"{provider}/{model_dir.name}"
    return model_dir.name


def copyable_model_id_from_urdf(urdf_path: Path) -> str:
    return describe_urdf_path(urdf_path).replace("/", " ").replace("\\", " ")


def describe_asset_path(asset_path: Path) -> str:
    """Return a compact browser label for a supported asset."""
    if asset_path.suffix.lower() == ".urdf":
        return describe_urdf_path(asset_path)

    parent = asset_path.parent
    grandparent = parent.parent
    if grandparent != parent:
        return f"{grandparent.name}/{parent.name}/{asset_path.name}"
    if parent.name:
        return f"{parent.name}/{asset_path.name}"
    return asset_path.name


def copyable_model_id_from_asset(asset_path: Path) -> str:
    return describe_asset_path(asset_path).replace("/", " ").replace("\\", " ")


def copy_text_to_clipboard(text: str, imgui: Any | None = None) -> bool:
    set_clipboard_text = getattr(imgui, "set_clipboard_text", None) if imgui is not None else None
    if callable(set_clipboard_text):
        try:
            set_clipboard_text(text)
            return True
        except Exception:
            pass

    commands = (
        ("clip", ()),
        ("pbcopy", ()),
        ("wl-copy", ()),
        ("xclip", ("-selection", "clipboard")),
        ("xsel", ("--clipboard", "--input")),
    )
    run_kwargs: dict[str, Any] = {
        "input": text,
        "text": True,
        "check": True,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "timeout": 2.0,
    }
    if hasattr(subprocess, "CREATE_NO_WINDOW"):
        run_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    for executable, args in commands:
        executable_path = shutil.which(executable)
        if executable_path is None:
            continue

        try:
            subprocess.run([executable_path, *args], **run_kwargs)
            return True
        except (OSError, subprocess.SubprocessError):
            continue

    return False


def matches_categories(urdf_path: Path, categories: tuple[str, ...] | None) -> bool:
    if not categories:
        return True

    wanted = {category.lower() for category in categories}
    return any(part.lower() in wanted for part in urdf_path.parts)


def validate_urdf_xml(urdf_path: Path) -> None:
    """Raise a descriptive error when a URDF is not well-formed XML."""
    try:
        ET.parse(urdf_path)
    except (OSError, ET.ParseError) as exc:
        raise ValueError(f"Malformed URDF XML: {urdf_path}: {exc}") from exc


def _valid_discovered_urdfs(urdf_paths: list[Path]) -> list[Path]:
    valid: list[Path] = []
    for urdf_path in urdf_paths:
        try:
            validate_urdf_xml(urdf_path)
        except ValueError as exc:
            print(f"Skipping {exc}", file=sys.stderr)
        else:
            valid.append(urdf_path)
    return valid


def find_urdfs(source: Path, categories: tuple[str, ...] | None = None) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() != ".urdf":
            raise ValueError(f"Expected a .urdf file, got: {source}")
        if not matches_categories(source, categories):
            raise FileNotFoundError(f"No URDF files matched categories {categories}: {source}")
        validate_urdf_xml(source)
        return [source]

    if not source.exists():
        raise FileNotFoundError(source)

    collider_urdfs = unique_sorted_paths(list(source.rglob("urdf_w_collider/*.urdf")))
    if collider_urdfs:
        filtered = [path for path in collider_urdfs if matches_categories(path, categories)]
        if filtered:
            valid = _valid_discovered_urdfs(filtered)
            if valid:
                return valid
            raise FileNotFoundError(f"No valid collider URDF files found under: {source}")
        raise FileNotFoundError(f"No collider URDF files matched categories {categories} under: {source}")

    recursive = unique_sorted_paths(list(source.rglob("*.urdf")))
    if recursive:
        filtered = [path for path in recursive if matches_categories(path, categories)]
        if filtered:
            valid = _valid_discovered_urdfs(filtered)
            if valid:
                return valid
            raise FileNotFoundError(f"No valid URDF files found under: {source}")
        raise FileNotFoundError(f"No URDF files matched categories {categories} under: {source}")

    raise FileNotFoundError(f"No URDF files found under: {source}")


def find_assets(source: Path, categories: tuple[str, ...] | None = None) -> list[Path]:
    """Discover every supported URDF, USD, and GLB asset below a path."""
    if source.is_file():
        if source.suffix.lower() not in SUPPORTED_ASSET_SUFFIXES:
            suffixes = ", ".join(sorted(SUPPORTED_ASSET_SUFFIXES))
            raise ValueError(f"Expected a supported asset file ({suffixes}), got: {source}")
        if not matches_categories(source, categories):
            raise FileNotFoundError(f"No assets matched categories {categories}: {source}")
        if source.suffix.lower() == ".urdf":
            validate_urdf_xml(source)
        return [source]

    if not source.exists():
        raise FileNotFoundError(source)

    collider_urdfs = unique_sorted_paths(list(source.rglob("urdf_w_collider/*.urdf")))
    urdfs = collider_urdfs or unique_sorted_paths(list(source.rglob("*.urdf")))
    other_assets = [
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_ASSET_SUFFIXES - {".urdf"}
    ]
    candidates = unique_sorted_paths([*urdfs, *other_assets])
    if not candidates:
        suffixes = ", ".join(sorted(SUPPORTED_ASSET_SUFFIXES))
        raise FileNotFoundError(f"No supported asset files ({suffixes}) found under: {source}")

    filtered = [path for path in candidates if matches_categories(path, categories)]
    if not filtered:
        raise FileNotFoundError(f"No assets matched categories {categories} under: {source}")

    valid: list[Path] = []
    for path in filtered:
        if path.suffix.lower() != ".urdf":
            valid.append(path)
            continue
        try:
            validate_urdf_xml(path)
        except ValueError as exc:
            print(f"Skipping {exc}", file=sys.stderr)
        else:
            valid.append(path)

    if valid:
        return valid
    raise FileNotFoundError(f"No valid supported asset files found under: {source}")


def print_urdfs(urdfs: list[Path]) -> None:
    for index, urdf in enumerate(urdfs):
        print(f"{index:03d}: {describe_urdf_path(urdf)} -> {urdf}")


def print_assets(assets: list[Path]) -> None:
    for index, asset in enumerate(assets):
        print(f"{index:03d}: {describe_asset_path(asset)} -> {asset}")


def describe_metadata(metadata: ArticulationMetadata) -> str:
    if metadata.path is None:
        return "no .articulations.json found"

    motion_states = sorted(
        {
            state_name
            for record in metadata.records_by_pid.values()
            for state_name in record.motion_states.keys()
        }
    )
    states_text = ", ".join(motion_states) if motion_states else "none"
    return (
        f"{metadata.path.name}: {len(metadata.records_by_pid)} articulations, "
        f"motion states: {states_text}, explicit dependency fields: {len(metadata.explicit_dependencies)}"
    )
