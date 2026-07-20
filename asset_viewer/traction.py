"""Picking-force reconstruction and live traction diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import newton
import newton.viewer
import numpy as np

from .controls import assist_imgui_window_wheel_scroll, widen_imgui_scrollbar


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
        self.last_side_scroll_y: float | None = None

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
        widen_imgui_scrollbar(imgui)
        self.last_side_scroll_y = assist_imgui_window_wheel_scroll(imgui, self.last_side_scroll_y)
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
