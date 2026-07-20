"""Joint-control state and the ImGui interface used to inspect articulations."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import newton
import numpy as np

from .assets import ArticulationMetadata, ArticulationRecord


UNBOUNDED_LIMIT = 1.0e6
DEFAULT_CONTINUOUS_SPEED_DEGREES = 90.0
SIDEBAR_SCROLLBAR_SIZE = 22.0
SIDEBAR_WHEEL_SCROLL_PIXELS = 64.0
BINARY_PRISMATIC_MAX_TRAVEL = 0.02
BINARY_PRISMATIC_NAME_TOKENS = ("button", "switch", "trigger", "gas lever")


def widen_imgui_scrollbar(imgui, min_size: float = SIDEBAR_SCROLLBAR_SIZE) -> None:
    get_style = getattr(imgui, "get_style", None)
    if not callable(get_style):
        return

    try:
        style = get_style()
    except Exception:
        return

    for attr in ("scrollbar_size", "ScrollbarSize", "scrollbarSize"):
        try:
            current = getattr(style, attr)
        except Exception:
            continue

        try:
            if float(current) < min_size:
                setattr(style, attr, min_size)
        except (TypeError, ValueError, AttributeError):
            continue
        return


def imgui_float_call(imgui, names: tuple[str, ...]) -> float | None:
    for name in names:
        func = getattr(imgui, name, None)
        if not callable(func):
            continue
        try:
            return float(func())
        except Exception:
            continue
    return None


def imgui_mouse_wheel(imgui) -> float:
    get_io = getattr(imgui, "get_io", None)
    if not callable(get_io):
        return 0.0

    try:
        io = get_io()
    except Exception:
        return 0.0

    for attr in ("mouse_wheel", "MouseWheel"):
        try:
            value = getattr(io, attr)
        except Exception:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    return 0.0


def imgui_current_window_hovered(imgui) -> bool:
    is_window_hovered = getattr(imgui, "is_window_hovered", None)
    if not callable(is_window_hovered):
        return False

    try:
        return bool(is_window_hovered())
    except TypeError:
        return False


def imgui_primary_mouse_down(imgui) -> bool:
    is_mouse_down = getattr(imgui, "is_mouse_down", None)
    if callable(is_mouse_down):
        try:
            return bool(is_mouse_down(0))
        except Exception:
            pass

    get_io = getattr(imgui, "get_io", None)
    if not callable(get_io):
        return False

    try:
        mouse_down = getattr(get_io(), "mouse_down")
        return bool(mouse_down[0])
    except Exception:
        return False


def set_imgui_scroll_y(imgui, value: float) -> bool:
    set_scroll_y = getattr(imgui, "set_scroll_y", None)
    if not callable(set_scroll_y):
        return False

    try:
        set_scroll_y(value)
    except Exception:
        return False
    return True


def assist_imgui_window_wheel_scroll(imgui, previous_scroll_y: float | None) -> float | None:
    current_scroll_y = imgui_float_call(imgui, ("get_scroll_y",))
    max_scroll_y = imgui_float_call(imgui, ("get_scroll_max_y",))
    if current_scroll_y is None or max_scroll_y is None:
        return previous_scroll_y

    wheel = imgui_mouse_wheel(imgui)
    if (
        previous_scroll_y is not None
        and abs(wheel) > 1.0e-6
        and max_scroll_y > 0.0
        and abs(current_scroll_y - previous_scroll_y) < 0.5
        and imgui_current_window_hovered(imgui)
        and not imgui_primary_mouse_down(imgui)
    ):
        next_scroll_y = current_scroll_y - wheel * SIDEBAR_WHEEL_SCROLL_PIXELS
        next_scroll_y = min(max(next_scroll_y, 0.0), max_scroll_y)
        if abs(next_scroll_y - current_scroll_y) > 0.5 and set_imgui_scroll_y(imgui, next_scroll_y):
            current_scroll_y = next_scroll_y

    return current_scroll_y


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
        self.last_side_scroll_y: float | None = None
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

        label_text = f"{control.label} {control.body_label}".lower().replace("_", " ").replace("-", " ")
        if any(token in label_text for token in BINARY_PRISMATIC_NAME_TOKENS):
            return True

        travel = abs(control.upper - control.lower)
        return travel <= BINARY_PRISMATIC_MAX_TRAVEL and bool(control.metadata.motion_states)

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

    def slider_no_input_flag(self, imgui) -> Any | None:
        for attr in ("SLIDER_FLAGS_NO_INPUT", "SliderFlags_NoInput"):
            value = getattr(imgui, attr, None)
            if value is not None:
                return value

        for enum_name in ("SliderFlags_", "SliderFlags"):
            enum = getattr(imgui, enum_name, None)
            if enum is None:
                continue
            for attr in ("no_input", "NoInput", "NO_INPUT"):
                value = getattr(enum, attr, None)
                if value is not None:
                    return value

        return None

    def slider_float_no_input(
        self,
        imgui,
        label: str,
        value: float,
        lower: float,
        upper: float,
        value_format: str,
    ) -> tuple[bool, float]:
        flags = self.slider_no_input_flag(imgui)
        if flags is not None:
            try:
                return imgui.slider_float(label, value, lower, upper, format=value_format, flags=flags)
            except TypeError:
                try:
                    return imgui.slider_float(label, value, lower, upper, value_format, flags)
                except TypeError:
                    pass

        return imgui.slider_float(label, value, lower, upper, format=value_format)

    def set_next_item_width(self, imgui, width: float) -> None:
        set_width = getattr(imgui, "set_next_item_width", None)
        if callable(set_width):
            set_width(width)

    def input_float_value(self, imgui, label: str, value: float, value_format: str) -> tuple[bool, float]:
        input_float = getattr(imgui, "input_float", None)
        if not callable(input_float):
            return False, value

        attempts = (
            lambda: input_float(label, value, format=value_format),
            lambda: input_float(label, value, 0.0, 0.0, format=value_format),
            lambda: input_float(label, value, 0.0, 0.0, value_format),
        )
        for attempt in attempts:
            try:
                result = attempt()
            except TypeError:
                continue

            if isinstance(result, tuple) and len(result) >= 2:
                return bool(result[0]), float(result[1])
            return False, value

        return False, value

    def render_slider_with_input(
        self,
        imgui,
        control: JointControl,
        value: float,
        lower: float,
        upper: float,
        slider_format: str,
        input_format: str,
    ) -> tuple[bool, float]:
        changed, value = self.slider_float_no_input(
            imgui,
            control.slider_label,
            value,
            lower,
            upper,
            slider_format,
        )

        imgui.same_line()
        self.set_next_item_width(imgui, 72.0)
        input_changed, input_value = self.input_float_value(
            imgui,
            f"##value_{control.joint_index}_{control.coord_index}",
            value,
            input_format,
        )
        if input_changed:
            return True, input_value

        return changed, value

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
        widen_imgui_scrollbar(imgui)
        self.last_side_scroll_y = assist_imgui_window_wheel_scroll(imgui, self.last_side_scroll_y)
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
                    changed, value = self.render_slider_with_input(
                        imgui,
                        control,
                        value,
                        lower,
                        upper,
                        "%.1f deg",
                        "%.1f",
                    )
                    if changed:
                        self.values[control.coord_index] = math.radians(value)
                        changed_any = True
                else:
                    value = float(self.values[control.coord_index])
                    changed, value = self.render_slider_with_input(
                        imgui,
                        control,
                        value,
                        control.lower,
                        control.upper,
                        "%.3f",
                        "%.4f",
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
