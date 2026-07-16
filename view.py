from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import newton
import newton.viewer
import numpy as np
import warp as wp


DEFAULT_SOURCE = Path(r"D:\Datasets\Artiverse\dataset_chunks\data")
UNBOUNDED_LIMIT = 1.0e6
DEFAULT_CONTINUOUS_SPEED_DEGREES = 90.0


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


@dataclass(frozen=True)
class JointControl:
    joint_index: int
    coord_index: int
    pid: int | None
    label: str
    body_label: str
    slider_label: str
    joint_type: int
    lower: float
    upper: float
    angular: bool
    metadata: ArticulationRecord | None


@dataclass(frozen=True)
class TractionForceSample:
    body_index: int
    body_label: str
    force: np.ndarray
    torque: np.ndarray
    point_world: np.ndarray
    target_world: np.ndarray
    force_magnitude: float
    torque_magnitude: float
    clamped: bool


class JointControlPanel:
    def __init__(
        self,
        model: newton.Model,
        state: newton.State,
        articulation_metadata: ArticulationMetadata,
        unbounded_range_degrees: float,
        continuous_speed_degrees: float,
    ) -> None:
        self.model = model
        self.state = state
        self.articulation_metadata = articulation_metadata
        self.unbounded_range = math.radians(unbounded_range_degrees)
        self.continuous_speed = math.radians(continuous_speed_degrees)
        self.values = np.array(state.joint_q.numpy(), dtype=np.float32)
        self.initial_values = self.values.copy()
        self.zero_velocities = np.zeros(state.joint_qd.shape[0], dtype=np.float32)
        self.controls = self._collect_controls()
        self.continuous_running = {
            control.coord_index: False for control in self.controls if self.is_continuous_motion_control(control)
        }
        self.continuous_directions = {
            control.coord_index: 1 for control in self.controls if self.is_continuous_motion_control(control)
        }
        self.clamp_to_limits = True
        self.show_body_positions = True

    def set_state(self, state: newton.State) -> None:
        self.state = state

    def _collect_controls(self) -> list[JointControl]:
        q_starts = self.model.joint_q_start.numpy()
        joint_types = self.model.joint_type.numpy()
        lower_limits = self.model.joint_limit_lower.numpy()
        upper_limits = self.model.joint_limit_upper.numpy()
        joint_child = getattr(self.model, "joint_child", None)
        if joint_child is None:
            joint_children = None
        elif hasattr(joint_child, "numpy"):
            joint_children = joint_child.numpy()
        else:
            joint_children = joint_child

        controls: list[JointControl] = []
        for joint_index, label in enumerate(self.model.joint_label):
            q_start = int(q_starts[joint_index])
            q_end = int(q_starts[joint_index + 1])
            if q_end <= q_start:
                continue

            joint_type = int(joint_types[joint_index])
            angular = joint_type == int(newton.JointType.REVOLUTE)
            short_label = label.split("/")[-1]
            pid = parse_joint_pid(label)
            metadata = self.articulation_metadata.records_by_pid.get(pid) if pid is not None else None
            body_label = ""
            if joint_children is not None:
                child_index = int(joint_children[joint_index])
                if 0 <= child_index < len(self.model.body_label):
                    body_label = self.model.body_label[child_index]

            for local_coord, coord_index in enumerate(range(q_start, q_end)):
                lower = self._limit_or_default(lower_limits, coord_index, -self.unbounded_range)
                upper = self._limit_or_default(upper_limits, coord_index, self.unbounded_range)
                if lower > upper:
                    lower, upper = upper, lower

                if q_end - q_start > 1:
                    display_label = f"{short_label}[{local_coord}]"
                else:
                    display_label = short_label

                controls.append(
                    JointControl(
                        joint_index=joint_index,
                        coord_index=coord_index,
                        pid=pid,
                        label=label,
                        body_label=body_label,
                        slider_label=f"{display_label}##joint_{joint_index}_{coord_index}",
                        joint_type=joint_type,
                        lower=lower,
                        upper=upper,
                        angular=angular,
                        metadata=metadata,
                    )
                )

        return controls

    def is_binary_button_control(self, control: JointControl) -> bool:
        if control.metadata is None:
            return False
        if control.metadata.type.lower() != "prismatic":
            return False

        label_text = f"{control.label} {control.body_label}".lower()
        return "button" in label_text

    def is_continuous_motion_control(self, control: JointControl) -> bool:
        return control.metadata is not None and control.metadata.type.lower() == "continuous"

    def display_control_label(self, control: JointControl) -> str:
        if control.body_label:
            return control.body_label.split("/")[-1]
        return control.slider_label.split("##", 1)[0]

    def is_on(self, control: JointControl) -> bool:
        value = float(self.values[control.coord_index])
        midpoint = control.lower + 0.5 * (control.upper - control.lower)
        return value > midpoint

    def render_binary_button_control(self, imgui, control: JointControl) -> bool:
        display_label = self.display_control_label(control)
        state_text = "On" if self.is_on(control) else "Off"
        imgui.text(display_label)
        imgui.same_line()
        if not imgui.button(f"{state_text}##button_toggle_{control.joint_index}_{control.coord_index}"):
            return False

        self.values[control.coord_index] = control.lower if self.is_on(control) else control.upper
        return True

    def render_continuous_motion_control(self, imgui, control: JointControl) -> None:
        coord_index = control.coord_index
        display_label = self.display_control_label(control)
        angle_degrees = math.degrees(float(self.values[coord_index]))
        running = self.continuous_running.get(coord_index, False)
        direction = self.continuous_directions.get(coord_index, 1)
        reverse = direction < 0

        imgui.text(f"{display_label}: {angle_degrees:.1f} deg")
        imgui.same_line()
        changed, running = imgui.checkbox(f"Run##continuous_run_{control.joint_index}_{coord_index}", running)
        if changed:
            self.continuous_running[coord_index] = running
        imgui.same_line()
        changed, reverse = imgui.checkbox(f"Reverse##continuous_direction_{control.joint_index}_{coord_index}", reverse)
        if changed:
            self.continuous_directions[coord_index] = -1 if reverse else 1

    def update_continuous_motion(self, dt: float) -> bool:
        if dt <= 0.0:
            return False

        changed = False
        for control in self.controls:
            if not self.is_continuous_motion_control(control):
                continue
            if not self.continuous_running.get(control.coord_index, False):
                continue

            direction = self.continuous_directions.get(control.coord_index, 1)
            value = float(self.values[control.coord_index]) + direction * self.continuous_speed * dt
            self.values[control.coord_index] = wrap_angle_radians(value)
            changed = True

        if changed:
            self.apply()

        return changed

    def _limit_or_default(self, limits: np.ndarray, coord_index: int, default: float) -> float:
        if coord_index >= len(limits):
            return default

        value = float(limits[coord_index])
        if not math.isfinite(value) or abs(value) > UNBOUNDED_LIMIT:
            return default

        return value

    def apply(self) -> None:
        if self.clamp_to_limits:
            for control in self.controls:
                if self.is_continuous_motion_control(control):
                    continue
                self.values[control.coord_index] = np.clip(
                    self.values[control.coord_index],
                    control.lower,
                    control.upper,
                )

        self.state.joint_q.assign(self.values)
        self.state.joint_qd.assign(self.zero_velocities)
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)

    def reset(self) -> None:
        self.values = self.initial_values.copy()
        for coord_index in self.continuous_running:
            self.continuous_running[coord_index] = False
        self.apply()

    def zero_all(self) -> None:
        self.values.fill(0.0)
        for coord_index in self.continuous_running:
            self.continuous_running[coord_index] = False
        self.apply()

    def set_matching(self, tokens: tuple[str, ...], value: float) -> None:
        for control in self.controls:
            if any(token in control.label.lower() for token in tokens):
                self.values[control.coord_index] = value
        self.apply()

    def available_motion_states(self) -> list[str]:
        states: set[str] = set()
        for control in self.controls:
            if control.metadata is None:
                continue
            states.update(control.metadata.motion_states.keys())
        return sorted(states)

    def motion_state_value(self, control: JointControl, state_name: str) -> float | None:
        if control.metadata is None:
            return None

        state = control.metadata.motion_states.get(state_name)
        if not isinstance(state, dict):
            return None

        raw_value = state.get("value")
        if raw_value is None and "percent" in state:
            try:
                percent = float(state["percent"]) / 100.0
            except (TypeError, ValueError):
                return None
            return control.lower + percent * (control.upper - control.lower)

        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return None

        # Artiverse mostly stores radians, but a few files store revolute
        # motion-state values such as -30 or -180 while ranges are radians.
        if control.angular and abs(value) > 2.0 * math.pi:
            value = math.radians(value)

        return value

    def apply_motion_state(self, state_name: str) -> int:
        count = 0
        for control in self.controls:
            value = self.motion_state_value(control, state_name)
            if value is None:
                continue
            self.values[control.coord_index] = value
            count += 1

        if count:
            self.apply()

        return count

    def render_metadata_ui(self, imgui) -> None:
        metadata = self.articulation_metadata
        imgui.text("Articulation Metadata")
        if metadata.path is None:
            imgui.text("No .articulations.json found.")
            return

        imgui.text(f"JSON: {metadata.path.name}")
        if metadata.model_id:
            imgui.text(f"Model: {metadata.model_id}")
        imgui.text(f"Articulations: {len(metadata.records_by_pid)}")
        if metadata.explicit_dependencies:
            imgui.text(f"Dependency fields: {len(metadata.explicit_dependencies)}")
            for key_path, value in metadata.explicit_dependencies[:8]:
                imgui.text(f"{key_path}: {short_repr(value)}")
        else:
            imgui.text("Explicit dependencies: none")

        states = self.available_motion_states()
        if states:
            imgui.text("Motion states:")
            for state_name in states:
                if imgui.button(f"Apply {state_name}##apply_state_{state_name}"):
                    self.apply_motion_state(state_name)
        else:
            imgui.text("Motion states: none")

        if imgui.tree_node("Raw articulation rows"):
            for record in sorted(metadata.records_by_pid.values(), key=lambda item: item.pid):
                state_names = ",".join(record.motion_states.keys()) if record.motion_states else "-"
                imgui.text(
                    f"pid {record.pid}: {record.type}, base={list(record.base)}, "
                    f"range=({format_optional(record.range_min)}, {format_optional(record.range_max)}), "
                    f"states={state_names}"
                )
            imgui.tree_pop()

    def render_ui(self, imgui) -> None:
        imgui.text("Joint Controls")

        if not self.controls:
            imgui.text("No movable joints found.")
        else:
            _changed, self.clamp_to_limits = imgui.checkbox("Clamp to limits", self.clamp_to_limits)

            if imgui.button("Reset all"):
                self.reset()
            imgui.same_line()
            if imgui.button("Zero all"):
                self.zero_all()

            if imgui.button("Open door 90"):
                self.set_matching(("door", "door_panel"), math.radians(90.0))
            imgui.same_line()
            if imgui.button("Close door"):
                self.set_matching(("door", "door_panel"), 0.0)

            changed_any = False
            for control in self.controls:
                is_binary_button = self.is_binary_button_control(control)
                is_continuous_motion = self.is_continuous_motion_control(control)

                if is_continuous_motion:
                    self.render_continuous_motion_control(imgui, control)
                elif is_binary_button:
                    changed_any = self.render_binary_button_control(imgui, control) or changed_any
                elif control.angular:
                    value = math.degrees(float(self.values[control.coord_index]))
                    lower = math.degrees(control.lower)
                    upper = math.degrees(control.upper)
                    changed, value = imgui.slider_float(control.slider_label, value, lower, upper, format="%.1f deg")
                    if changed:
                        self.values[control.coord_index] = math.radians(value)
                        changed_any = True
                else:
                    value = float(self.values[control.coord_index])
                    changed, value = imgui.slider_float(
                        control.slider_label,
                        value,
                        control.lower,
                        control.upper,
                        format="%.3f",
                    )
                    if changed:
                        self.values[control.coord_index] = value
                        changed_any = True

                if not is_binary_button and not is_continuous_motion:
                    imgui.same_line()
                    if imgui.button(f"0##zero_{control.coord_index}"):
                        self.values[control.coord_index] = 0.0
                        changed_any = True

                if control.metadata and control.metadata.motion_states:
                    imgui.same_line()
                    state_names = ",".join(control.metadata.motion_states.keys())
                    imgui.text(f"pid {control.metadata.pid}: {state_names}")

            if changed_any:
                self.apply()

        imgui.separator()
        self.render_position_ui(imgui)
        imgui.separator()
        self.render_metadata_ui(imgui)

    def render_position_ui(self, imgui) -> None:
        imgui.text("Position Display")
        _changed, self.show_body_positions = imgui.checkbox("Show body positions", self.show_body_positions)

        if imgui.button("Print positions"):
            self.print_body_positions()

        if not self.show_body_positions:
            return

        body_q = self.state.body_q.numpy()
        if imgui.tree_node("Body world positions"):
            for body_index, label in enumerate(self.model.body_label):
                pos = body_q[body_index, 0:3]
                imgui.text(f"{body_index:02d} {label}: x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}")
            imgui.tree_pop()

    def print_body_positions(self) -> None:
        body_q = self.state.body_q.numpy()
        print("Current body positions:")
        for body_index, label in enumerate(self.model.body_label):
            pos = body_q[body_index, 0:3]
            print(f"{body_index:02d} {label}: x={pos[0]:.6f}, y={pos[1]:.6f}, z={pos[2]:.6f}")


def parse_joint_pid(label: str) -> int | None:
    match = re.search(r"(?:^|/)joint_(\d+)_to_", label)
    if match:
        return int(match.group(1))

    match = re.search(r"(?:^|/)link_(\d+)(?:_|$)", label)
    if match:
        return int(match.group(1))

    return None


def format_optional(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6g}"


def wrap_angle_radians(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def short_repr(value: Any, max_len: int = 80) -> str:
    text = repr(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


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
    return describe_urdf_path(urdf_path).replace("/", "\\")


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


def find_urdfs(source: Path, categories: tuple[str, ...] | None = None) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() != ".urdf":
            raise ValueError(f"Expected a .urdf file, got: {source}")
        if not matches_categories(source, categories):
            raise FileNotFoundError(f"No URDF files matched categories {categories}: {source}")
        return [source]

    if not source.exists():
        raise FileNotFoundError(source)

    collider_urdfs = unique_sorted_paths(list(source.rglob("urdf_w_collider/*.urdf")))
    if collider_urdfs:
        filtered = [path for path in collider_urdfs if matches_categories(path, categories)]
        if filtered:
            return filtered
        raise FileNotFoundError(f"No collider URDF files matched categories {categories} under: {source}")

    recursive = unique_sorted_paths(list(source.rglob("*.urdf")))
    if recursive:
        filtered = [path for path in recursive if matches_categories(path, categories)]
        if filtered:
            return filtered
        raise FileNotFoundError(f"No URDF files matched categories {categories} under: {source}")

    raise FileNotFoundError(f"No URDF files found under: {source}")


def set_matching_joint_q(builder: newton.ModelBuilder, tokens: tuple[str, ...], angle_rad: float) -> int:
    count = 0

    for joint_index, joint_label in enumerate(builder.joint_label):
        q_start = builder.joint_q_start[joint_index]
        if joint_index + 1 < len(builder.joint_q_start):
            q_end = builder.joint_q_start[joint_index + 1]
        else:
            q_end = len(builder.joint_q)
        if q_start == q_end:
            continue

        child_index = builder.joint_child[joint_index]
        child_label = builder.body_label[child_index] if child_index >= 0 else ""
        text = f"{joint_label} {child_label}".lower()

        if any(token in text for token in tokens):
            builder.joint_q[q_start] = angle_rad
            builder.joint_target_q[q_start] = angle_rad
            count += 1

    return count


def build_model(args: argparse.Namespace, urdf_path: Path) -> tuple[newton.Model, newton.State, JointControlPanel]:
    builder = newton.ModelBuilder(
        up_axis=newton.Axis.Z,
        gravity=-9.81 if args.simulate else 0.0,
    )

    builder.add_urdf(
        str(urdf_path),
        xform=wp.transform((0.0, 0.0, args.z), wp.quat_identity()),
        floating=False,
        scale=args.scale,
        up_axis=newton.Axis.Z,
        enable_self_collisions=False,
        collapse_fixed_joints=False,
        force_show_colliders=args.show_colliders,
    )

    if args.turntable:
        set_matching_joint_q(builder, ("turntable",), math.radians(args.turntable))

    if args.knobs:
        set_matching_joint_q(builder, ("knob",), math.radians(args.knobs))

    if args.open_door:
        set_matching_joint_q(builder, ("door", "door_panel"), math.radians(args.open_door))

    if args.ground:
        builder.add_ground_plane()

    model = builder.finalize()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    articulation_metadata = load_articulation_metadata(urdf_path)
    joint_panel = JointControlPanel(
        model,
        state,
        articulation_metadata,
        args.joint_range_degrees,
        args.continuous_speed_degrees,
    )
    if args.initial_motion_state:
        joint_panel.apply_motion_state(args.initial_motion_state)
    return model, state, joint_panel


def print_urdfs(urdfs: list[Path]) -> None:
    for index, urdf in enumerate(urdfs):
        print(f"{index:03d}: {describe_urdf_path(urdf)} -> {urdf}")


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


def vec3_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(3)


def rotate_vector_by_quat(quat_xyzw: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quat_vector = quat_xyzw[0:3]
    quat_w = float(quat_xyzw[3])
    cross_term = 2.0 * np.cross(quat_vector, vector)
    return vector + quat_w * cross_term + np.cross(quat_vector, cross_term)


def transform_point_np(transform_xyzw: np.ndarray, point_local: np.ndarray) -> np.ndarray:
    return transform_xyzw[0:3] + rotate_vector_by_quat(transform_xyzw[3:7], point_local)


def compute_picking_traction_force(
    viewer: newton.viewer.ViewerGL,
    model: newton.Model,
    state: newton.State,
) -> TractionForceSample | None:
    """Recompute the right-click picking force using Newton's viewer formula."""
    picking = getattr(viewer, "picking", None)
    if picking is None or getattr(picking, "pick_body", None) is None:
        return None

    pick_body = int(picking.pick_body.numpy()[0])
    if pick_body < 0 or pick_body >= model.body_count:
        return None

    body_flags = model.body_flags.numpy()
    if int(body_flags[pick_body]) & int(newton.BodyFlags.KINEMATIC):
        return None

    pick_state = picking.pick_state.numpy()[0]
    body_q = state.body_q.numpy()
    body_qd = state.body_qd.numpy()
    body_com = model.body_com.numpy()
    body_mass = model.body_mass.numpy()
    effective_mass = getattr(picking, "_pick_effective_mass", None)
    effective_mass_np = effective_mass.numpy() if effective_mass is not None else body_mass

    transform_xyzw = np.asarray(body_q[pick_body], dtype=np.float64)
    picked_point_local = vec3_array(pick_state["picked_point_local"])
    pick_target_world = vec3_array(pick_state["picking_target_world"])
    pick_pos_world = transform_point_np(transform_xyzw, picked_point_local)
    body_com_world = transform_point_np(transform_xyzw, vec3_array(body_com[pick_body]))

    body_velocity = np.asarray(body_qd[pick_body], dtype=np.float64)
    vel_com = body_velocity[0:3]
    angular_vel = body_velocity[3:6]
    offset = pick_pos_world - body_com_world
    vel_at_offset = vel_com + np.cross(angular_vel, offset)

    force_multiplier = 10.0 + float(body_mass[pick_body])
    force = force_multiplier * (
        float(pick_state["pick_stiffness"]) * (pick_target_world - pick_pos_world)
        - float(pick_state["pick_damping"]) * vel_at_offset
    )

    max_force = float(pick_state["pick_max_acceleration"]) * 9.81 * float(effective_mass_np[pick_body])
    force_magnitude = float(np.linalg.norm(force))
    clamped = force_magnitude > max_force and force_magnitude > 0.0
    if clamped:
        force = force * (max_force / force_magnitude)
        force_magnitude = float(np.linalg.norm(force))

    torque = np.cross(offset, force)
    torque_magnitude = float(np.linalg.norm(torque))
    body_label = model.body_label[pick_body] if pick_body < len(model.body_label) else f"body_{pick_body}"

    return TractionForceSample(
        body_index=pick_body,
        body_label=body_label,
        force=force,
        torque=torque,
        point_world=pick_pos_world,
        target_world=pick_target_world,
        force_magnitude=force_magnitude,
        torque_magnitude=torque_magnitude,
        clamped=clamped,
    )


class TractionForceMonitor:
    def __init__(self, model: newton.Model, print_hz: float, print_to_console: bool) -> None:
        self.model = model
        self.print_to_console = print_to_console
        self.print_interval = 0.0 if print_hz <= 0.0 else 1.0 / print_hz
        self.next_print_time = 0.0
        self.active_body: int | None = None
        self.last_sample: TractionForceSample | None = None
        self.last_sample_time = 0.0

    def update(self, viewer: newton.viewer.ViewerGL, state: newton.State, sim_time: float) -> None:
        sample = compute_picking_traction_force(viewer, self.model, state)
        self.last_sample = sample
        self.last_sample_time = sim_time
        if sample is None:
            if self.print_to_console and self.active_body is not None:
                label = self._body_label(self.active_body)
                print(f"[traction t={sim_time:8.4f}s] released body {self.active_body:02d} {label}")
            self.active_body = None
            self.next_print_time = sim_time
            return

        if self.active_body != sample.body_index:
            self.active_body = sample.body_index
            self.next_print_time = sim_time

        if not self.print_to_console:
            return

        if self.print_interval > 0.0 and sim_time + 1.0e-9 < self.next_print_time:
            return

        self.next_print_time = sim_time + self.print_interval
        fx, fy, fz = sample.force
        tx, ty, tz = sample.torque
        px, py, pz = sample.point_world
        clamped_text = " clamped" if sample.clamped else ""
        print(
            f"[traction t={sim_time:8.4f}s] body {sample.body_index:02d} {sample.body_label}: "
            f"F=({fx:+.4f}, {fy:+.4f}, {fz:+.4f}) N |F|={sample.force_magnitude:.4f} N; "
            f"tau=({tx:+.4f}, {ty:+.4f}, {tz:+.4f}) N*m |tau|={sample.torque_magnitude:.4f} N*m; "
            f"point=({px:+.4f}, {py:+.4f}, {pz:+.4f}){clamped_text}"
        )

    def _body_label(self, body_index: int) -> str:
        if 0 <= body_index < len(self.model.body_label):
            return self.model.body_label[body_index]
        return f"body_{body_index}"

    def render_ui(self, imgui) -> None:
        imgui.separator()
        imgui.text("Traction Force")

        sample = self.last_sample
        if sample is None:
            status = "inactive"
            body_text = "none"
            fx, fy, fz = 0.0, 0.0, 0.0
            tx, ty, tz = 0.0, 0.0, 0.0
            px, py, pz = 0.0, 0.0, 0.0
            gx, gy, gz = 0.0, 0.0, 0.0
            force_magnitude = 0.0
            torque_magnitude = 0.0
            clamp_state = "off"
        else:
            status = "active"
            body_text = f"{sample.body_index:02d} {sample.body_label}"
            fx, fy, fz = sample.force
            tx, ty, tz = sample.torque
            px, py, pz = sample.point_world
            gx, gy, gz = sample.target_world
            force_magnitude = sample.force_magnitude
            torque_magnitude = sample.torque_magnitude
            clamp_state = "active" if sample.clamped else "off"

        imgui.text(f"Time: {self.last_sample_time:.4f}s")
        imgui.text(f"Status: {status}")
        imgui.text(f"Body: {body_text}")
        imgui.text(f"Force xyz: {fx:+.4f}, {fy:+.4f}, {fz:+.4f} N")
        imgui.text(f"Force magnitude: {force_magnitude:.4f} N")
        imgui.text(f"Torque xyz: {tx:+.4f}, {ty:+.4f}, {tz:+.4f} N*m")
        imgui.text(f"Torque magnitude: {torque_magnitude:.4f} N*m")
        imgui.text(f"Pick point world: {px:+.4f}, {py:+.4f}, {pz:+.4f}")
        imgui.text(f"Target world: {gx:+.4f}, {gy:+.4f}, {gz:+.4f}")
        imgui.text(f"Clamp: {clamp_state}")


@dataclass
class LoadedAsset:
    index: int
    urdf_path: Path
    model: newton.Model
    state: newton.State
    joint_panel: JointControlPanel
    solver: Any | None
    state_next: newton.State | None
    control: Any | None
    contacts: Any | None
    traction_monitor: TractionForceMonitor | None


class AssetBrowser:
    def __init__(self, urdfs: list[Path], current_index: int) -> None:
        self.urdfs = urdfs
        self.current_index = current_index
        self.requested_index: int | None = None
        self.last_copied_model_id: str | None = None
        self.copy_error_model_id: str | None = None
        self.tree = self._build_tree(urdfs)

    def _build_tree(self, urdfs: list[Path]) -> dict[str, dict[str, list[tuple[int, str]]]]:
        tree: dict[str, dict[str, list[tuple[int, str]]]] = {}
        for index, urdf_path in enumerate(urdfs):
            parts = describe_urdf_path(urdf_path).split("/")
            category = parts[0] if parts else "assets"
            provider = parts[1] if len(parts) > 2 else "models"
            model_name = parts[-1] if parts else urdf_path.stem
            tree.setdefault(category, {}).setdefault(provider, []).append((index, model_name))
        return {
            category: {
                provider: sorted(entries, key=lambda item: (item[1].lower(), item[0]))
                for provider, entries in sorted(providers.items())
            }
            for category, providers in sorted(tree.items())
        }

    def render_ui(self, imgui) -> None:
        imgui.set_next_item_open(True, imgui.Cond_.appearing)
        if not imgui.collapsing_header("Assets"):
            return

        imgui.text(f"Current: {self.current_index:03d} / {len(self.urdfs) - 1:03d}")
        imgui.text(describe_urdf_path(self.urdfs[self.current_index]))
        model_id = copyable_model_id_from_urdf(self.urdfs[self.current_index])
        imgui.text(f"Model ID: {model_id}")

        if imgui.button("Copy Model ID##asset_copy_model_id"):
            if copy_text_to_clipboard(model_id, imgui):
                self.last_copied_model_id = model_id
                self.copy_error_model_id = None
            else:
                self.last_copied_model_id = None
                self.copy_error_model_id = model_id

        if self.last_copied_model_id == model_id:
            imgui.same_line()
            imgui.text("Copied")
        elif self.copy_error_model_id == model_id:
            imgui.same_line()
            imgui.text("Copy failed")

        if imgui.button("Reload##asset_reload"):
            self.requested_index = self.current_index
        imgui.same_line()
        if imgui.button("Previous##asset_previous"):
            self.requested_index = (self.current_index - 1) % len(self.urdfs)
        imgui.same_line()
        if imgui.button("Next##asset_next"):
            self.requested_index = (self.current_index + 1) % len(self.urdfs)

        if imgui.tree_node("Asset List"):
            for category, providers in self.tree.items():
                if imgui.tree_node(category):
                    for provider, entries in providers.items():
                        if imgui.tree_node(provider):
                            for index, model_name in entries:
                                selected = index == self.current_index
                                clicked, _selected = imgui.selectable(
                                    f"{index:03d} {model_name}##asset_{index}",
                                    selected,
                                )
                                if clicked:
                                    self.requested_index = index
                            imgui.tree_pop()
                    imgui.tree_pop()
            imgui.tree_pop()

    def consume_request(self) -> int | None:
        requested_index = self.requested_index
        self.requested_index = None
        return requested_index


class AssetViewerRuntime:
    def __init__(
        self,
        args: argparse.Namespace,
        urdfs: list[Path],
        viewer: newton.viewer.ViewerGL,
        double_sided_state: dict[str, bool],
        asset_browser: AssetBrowser,
    ) -> None:
        self.args = args
        self.urdfs = urdfs
        self.viewer = viewer
        self.double_sided_state = double_sided_state
        self.asset_browser = asset_browser
        self.loaded: LoadedAsset | None = None
        self.sim_time = 0.0

    def load(self, index: int, show_splash: bool = False) -> None:
        if index < 0 or index >= len(self.urdfs):
            raise IndexError(f"Asset index must be between 0 and {len(self.urdfs) - 1}; got {index}")

        urdf_path = self.urdfs[index]
        label = describe_urdf_path(urdf_path)
        if show_splash:
            self._show_splash(f"Loading {label}...")

        try:
            model, state, joint_panel = build_model(self.args, urdf_path)
            solver = None
            state_next = None
            control = None
            contacts = None
            if self.args.simulate:
                solver = newton.solvers.SolverXPBD(model, iterations=self.args.iterations)
                state_next = model.state()
                control = model.control()
                contacts = model.contacts()

            if self.loaded is not None:
                self.viewer.clear_model()

            self.viewer.set_model(model)
            self.viewer.set_camera(
                wp.vec3(self.args.camera_x, self.args.camera_y, self.args.camera_z),
                self.args.pitch,
                self.args.yaw,
            )
            set_double_sided_rendering(self.viewer, self.double_sided_state["enabled"])
            self._set_reset_callback(joint_panel.reset)

            traction_monitor = (
                TractionForceMonitor(model, self.args.traction_print_hz, self.args.print_traction_force)
                if self.args.simulate
                else None
            )
            self._register_asset_ui(joint_panel, traction_monitor)

            self.loaded = LoadedAsset(
                index=index,
                urdf_path=urdf_path,
                model=model,
                state=state,
                joint_panel=joint_panel,
                solver=solver,
                state_next=state_next,
                control=control,
                contacts=contacts,
                traction_monitor=traction_monitor,
            )
            self.asset_browser.current_index = index
            self.sim_time = 0.0
            self._print_loaded_summary(label, urdf_path, model, joint_panel)
        finally:
            if show_splash:
                self._hide_splash()

    def _register_asset_ui(
        self,
        joint_panel: JointControlPanel,
        traction_monitor: TractionForceMonitor | None,
    ) -> None:
        if self.args.joint_ui and hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(joint_panel.render_ui, position="side")
        if traction_monitor is not None and hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(traction_monitor.render_ui, position="side")

    def _set_reset_callback(self, callback) -> None:
        if hasattr(self.viewer, "set_reset_callback"):
            self.viewer.set_reset_callback(callback)
        else:
            self.viewer._reset_callback = callback

    def _show_splash(self, text: str) -> None:
        if not hasattr(self.viewer, "show_loading_splash"):
            return
        self.viewer.show_loading_splash(text)
        for _ in range(2):
            self.viewer.begin_frame(self.sim_time)
            self.viewer.end_frame()

    def _hide_splash(self) -> None:
        if hasattr(self.viewer, "hide_loading_splash"):
            self.viewer.hide_loading_splash()

    def _print_loaded_summary(
        self,
        label: str,
        urdf_path: Path,
        model: newton.Model,
        joint_panel: JointControlPanel,
    ) -> None:
        print(f"Opening {label}: {urdf_path}")
        print(
            f"Loaded {model.body_count} bodies, {model.joint_count} joints, "
            f"{model.shape_count} shapes, {model.joint_coord_count} joint coords."
        )
        print(f"Articulation metadata: {describe_metadata(joint_panel.articulation_metadata)}")
        print(f"Interactive controls: {len(joint_panel.controls)} joint controls in the left panel.")


def set_double_sided_rendering(viewer: newton.viewer.ViewerGL, enabled: bool) -> None:
    """Toggle back-face culling for meshes already registered in ViewerGL."""
    objects = getattr(viewer, "objects", {})
    for obj in objects.values():
        if hasattr(obj, "backface_culling"):
            obj.backface_culling = not enabled


def run_viewer(
    args: argparse.Namespace,
    urdfs: list[Path],
    initial_index: int,
) -> None:
    viewer = newton.viewer.ViewerGL(
        width=args.width,
        height=args.height,
        headless=args.headless,
        paused=not args.simulate,
    )

    double_sided_state = {"enabled": args.double_sided}
    asset_browser = AssetBrowser(urdfs, initial_index)
    runtime = AssetViewerRuntime(args, urdfs, viewer, double_sided_state, asset_browser)

    if hasattr(viewer, "register_ui_callback"):
        viewer.register_ui_callback(asset_browser.render_ui, position="panel")

    def render_double_sided_option(imgui) -> None:
        changed, enabled = imgui.checkbox("Double-sided Meshes", double_sided_state["enabled"])
        if changed:
            double_sided_state["enabled"] = enabled
        set_double_sided_rendering(viewer, double_sided_state["enabled"])

    if hasattr(viewer, "register_ui_callback"):
        viewer.register_ui_callback(render_double_sided_option, position="rendering")

    runtime.load(initial_index)

    frame_dt = 1.0 / args.fps
    sim_dt = frame_dt / args.substeps
    frame = 0

    if not args.simulate and args.print_traction_force:
        print("Traction force printing is idle because --simulate is off.")

    while viewer.is_running():
        requested_index = asset_browser.consume_request()
        if requested_index is not None:
            runtime.load(requested_index, show_splash=True)

        loaded = runtime.loaded
        if loaded is None:
            break

        if args.simulate and viewer.should_step():
            assert loaded.solver is not None
            assert loaded.state_next is not None
            assert loaded.control is not None
            assert loaded.contacts is not None

            for substep in range(args.substeps):
                substep_time = runtime.sim_time + substep * sim_dt
                loaded.state.clear_forces()
                viewer.apply_forces(loaded.state)
                if loaded.traction_monitor is not None:
                    loaded.traction_monitor.update(viewer, loaded.state, substep_time)
                loaded.model.collide(loaded.state, loaded.contacts)
                loaded.solver.step(loaded.state, loaded.state_next, loaded.control, loaded.contacts, sim_dt)
                loaded.state, loaded.state_next = loaded.state_next, loaded.state

            runtime.sim_time += frame_dt

        loaded.joint_panel.set_state(loaded.state)
        loaded.joint_panel.update_continuous_motion(frame_dt)

        viewer.begin_frame(runtime.sim_time)
        viewer.log_state(loaded.state)
        if loaded.contacts is not None:
            viewer.log_contacts(loaded.contacts, loaded.state)
        viewer.end_frame()

        frame += 1
        if args.frames is not None and frame >= args.frames:
            break

    viewer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View Artiverse URDF assets with Newton.")
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Artiverse data root, one category directory, one model directory, or one .urdf file.",
    )
    parser.add_argument("--list", action="store_true", help="List matching URDF files and exit.")
    parser.add_argument(
        "--category",
        action="append",
        default=None,
        help="Limit discovery to a category folder, e.g. microwave or scissors. Repeat for multiple categories.",
    )
    parser.add_argument("--index", type=int, default=0, help="URDF index to open when source contains many models.")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--z", type=float, default=0.0, help="Vertical offset for the imported model.")
    parser.add_argument("--ground", action="store_true", help="Add a ground plane.")
    parser.add_argument("--show-colliders", action="store_true", help="Show collision meshes even when visual meshes exist.")
    parser.add_argument(
        "--double-sided",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render meshes from both sides by disabling back-face culling.",
    )
    parser.add_argument("--open-door", type=float, default=45.0, help="Initial door angle in degrees. Use 0 to close.")
    parser.add_argument("--knobs", type=float, default=0.0, help="Initial knob angle in degrees.")
    parser.add_argument("--turntable", type=float, default=0.0, help="Initial turntable angle in degrees.")
    parser.add_argument("--joint-ui", action=argparse.BooleanOptionalAction, default=True, help="Show joint sliders.")
    parser.add_argument(
        "--joint-range-degrees",
        type=float,
        default=180.0,
        help="Slider half-range for continuous/unbounded revolute joints.",
    )
    parser.add_argument(
        "--continuous-speed-degrees",
        type=float,
        default=DEFAULT_CONTINUOUS_SPEED_DEGREES,
        help="Angular speed in degrees per second for continuous joint start/stop controls.",
    )
    parser.add_argument(
        "--initial-motion-state",
        type=str,
        default=None,
        help="Apply a state from *.articulations.json on startup, e.g. closed.",
    )
    parser.add_argument("--simulate", action="store_true", help="Run Newton dynamics instead of static viewing.")
    parser.add_argument(
        "--print-traction-force",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print the right-click picking traction force to the terminal while dragging in simulation mode.",
    )
    parser.add_argument(
        "--traction-print-hz",
        type=float,
        default=10.0,
        help="Traction-force print rate in Hz. Use 0 to print every physics substep.",
    )
    parser.add_argument("--frames", type=int, default=None, help="Close after N rendered frames.")
    parser.add_argument("--headless", action="store_true", help="Run ViewerGL without opening a visible window.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--camera-x", type=float, default=0.65)
    parser.add_argument("--camera-y", type=float, default=-0.75)
    parser.add_argument("--camera-z", type=float, default=0.45)
    parser.add_argument("--pitch", type=float, default=-25.0)
    parser.add_argument("--yaw", type=float, default=40.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    categories = tuple(args.category) if args.category else None
    urdfs = find_urdfs(args.source, categories)

    if args.list:
        print_urdfs(urdfs)
        return

    if args.index < 0 or args.index >= len(urdfs):
        raise IndexError(f"--index must be between 0 and {len(urdfs) - 1}; got {args.index}")

    run_viewer(args, urdfs, args.index)


if __name__ == "__main__":
    main()
