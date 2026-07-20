from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import newton
import newton.viewer
import numpy as np
import warp as wp

from asset_viewer import (
    DEFAULT_SOURCE,
    describe_urdf_path,
    find_urdfs,
    set_double_sided_rendering,
    widen_imgui_scrollbar,
)


PROBE_LABEL = "collision_probe"
DEFAULT_PROBE_RADIUS = 0.02
DEFAULT_SCAN_FRAMES = 240


@dataclass(frozen=True)
class ProbeContact:
    body_index: int
    body_label: str
    shape_index: int
    shape_label: str
    point_world: np.ndarray
    normal_probe_to_asset: np.ndarray
    penetration: float


def quaternion_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = quaternion[:3]
    w = float(quaternion[3])
    twice_cross = 2.0 * np.cross(xyz, vector)
    return vector + w * twice_cross + np.cross(xyz, twice_cross)


def body_point_to_world(body_poses: np.ndarray, body_index: int, point_body: np.ndarray) -> np.ndarray:
    if body_index < 0:
        return np.asarray(point_body, dtype=np.float64)

    pose = body_poses[body_index]
    return np.asarray(pose[:3], dtype=np.float64) + quaternion_rotate(
        np.asarray(pose[3:7], dtype=np.float64),
        np.asarray(point_body, dtype=np.float64),
    )


def set_kinematic_body_position(state: newton.State, body_index: int, position: np.ndarray) -> None:
    body_q = state.body_q.numpy()
    body_q[body_index, :3] = position
    body_q[body_index, 3:7] = (0.0, 0.0, 0.0, 1.0)
    state.body_q.assign(body_q)

    body_qd = state.body_qd.numpy()
    body_qd[body_index, :] = 0.0
    state.body_qd.assign(body_qd)


def collision_bounds(
    model: newton.Model,
    asset_shape_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    flags = model.shape_flags.numpy()[:asset_shape_count]
    collide_flag = int(newton.ShapeFlags.COLLIDE_SHAPES)
    collision_indices = np.flatnonzero((flags.astype(np.int64) & collide_flag) != 0)
    if not len(collision_indices):
        raise ValueError("The asset has no collision shapes to probe.")

    lower = model.shape_collision_aabb_lower.numpy()[collision_indices]
    upper = model.shape_collision_aabb_upper.numpy()[collision_indices]
    finite = np.isfinite(lower).all(axis=1) & np.isfinite(upper).all(axis=1)
    if not finite.any():
        raise ValueError("The asset collision shapes have no finite world-space bounds.")

    return lower[finite].min(axis=0).astype(np.float64), upper[finite].max(axis=0).astype(np.float64)


def collect_probe_contacts(
    model: newton.Model,
    state: newton.State,
    contacts: Any,
    probe_shape: int,
) -> list[ProbeContact]:
    count = min(int(contacts.rigid_contact_count.numpy()[0]), int(contacts.rigid_contact_max))
    if count <= 0:
        return []

    shape0 = contacts.rigid_contact_shape0.numpy()[:count]
    shape1 = contacts.rigid_contact_shape1.numpy()[:count]
    point0 = contacts.rigid_contact_point0.numpy()[:count]
    point1 = contacts.rigid_contact_point1.numpy()[:count]
    normal = contacts.rigid_contact_normal.numpy()[:count]
    margin0 = contacts.rigid_contact_margin0.numpy()[:count]
    margin1 = contacts.rigid_contact_margin1.numpy()[:count]
    shape_bodies = model.shape_body.numpy()
    body_poses = state.body_q.numpy()

    samples: list[ProbeContact] = []
    for index in range(count):
        first_shape = int(shape0[index])
        second_shape = int(shape1[index])
        if probe_shape not in (first_shape, second_shape):
            continue

        first_body = int(shape_bodies[first_shape])
        second_body = int(shape_bodies[second_shape])
        first_world = body_point_to_world(body_poses, first_body, point0[index])
        second_world = body_point_to_world(body_poses, second_body, point1[index])
        normal_first_to_second = np.asarray(normal[index], dtype=np.float64)
        signed_distance = float(np.dot(second_world - first_world, normal_first_to_second))
        signed_distance -= float(margin0[index]) + float(margin1[index])
        # Collision pipelines may retain speculative near-contacts when a
        # positive contact gap is configured. A geometry probe reports only
        # touching or penetrating pairs, not merely nearby pairs.
        if signed_distance > 1.0e-6:
            continue

        if first_shape == probe_shape:
            asset_shape = second_shape
            asset_body = second_body
            probe_point = first_world
            probe_to_asset = normal_first_to_second
        else:
            asset_shape = first_shape
            asset_body = first_body
            probe_point = second_world
            probe_to_asset = -normal_first_to_second

        body_label = (
            model.body_label[asset_body]
            if 0 <= asset_body < len(model.body_label)
            else f"body_{asset_body}"
        )
        shape_label = (
            model.shape_label[asset_shape]
            if 0 <= asset_shape < len(model.shape_label)
            else f"shape_{asset_shape}"
        )
        samples.append(
            ProbeContact(
                body_index=asset_body,
                body_label=body_label,
                shape_index=asset_shape,
                shape_label=shape_label,
                point_world=probe_point,
                normal_probe_to_asset=probe_to_asset,
                penetration=max(0.0, -signed_distance),
            )
        )

    return samples


class CollisionProbePanel:
    def __init__(
        self,
        position: np.ndarray,
        bounds_lower: np.ndarray,
        bounds_upper: np.ndarray,
        radius: float,
        step: float,
        print_contacts: bool,
    ) -> None:
        self.position = np.asarray(position, dtype=np.float64)
        self.bounds_lower = np.asarray(bounds_lower, dtype=np.float64)
        self.bounds_upper = np.asarray(bounds_upper, dtype=np.float64)
        self.radius = radius
        self.step = step
        self.print_contacts = print_contacts
        self.contacts: list[ProbeContact] = []
        self._last_contact_key: tuple[int, tuple[int, ...]] | None = None

        extent = self.bounds_upper - self.bounds_lower
        padding = max(float(np.linalg.norm(extent)) * 0.25, 4.0 * radius)
        self.slider_lower = self.bounds_lower - padding
        self.slider_upper = self.bounds_upper + padding

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.bounds_lower + self.bounds_upper)

    def outside_position(self, axis: int, direction: int) -> np.ndarray:
        position = self.center.copy()
        boundary = self.bounds_upper[axis] if direction > 0 else self.bounds_lower[axis]
        position[axis] = boundary + direction * 2.0 * self.radius
        return position

    def update_contacts(self, samples: list[ProbeContact]) -> None:
        self.contacts = samples
        key = (len(samples), tuple(sorted({sample.body_index for sample in samples})))
        if key == self._last_contact_key:
            return
        self._last_contact_key = key

        if not self.print_contacts:
            return
        if not samples:
            print(f"[probe] clear at {self.position.tolist()}")
            return

        first = samples[0]
        point = ", ".join(f"{value:+.5f}" for value in first.point_world)
        normal = ", ".join(f"{value:+.4f}" for value in first.normal_probe_to_asset)
        print(
            f"[probe] contacts={len(samples)} body={first.body_index} {first.body_label} "
            f"point=({point}) normal=({normal}) penetration={first.penetration:.6f} m"
        )

    def render_ui(self, imgui) -> None:
        widen_imgui_scrollbar(imgui)
        imgui.separator()
        imgui.text("Collision Probe")
        imgui.text(f"Radius: {self.radius:.5f} m")
        imgui.text(f"Step: {self.step:.5f} m")

        axis_names = ("X", "Y", "Z")
        for axis, name in enumerate(axis_names):
            changed, value = imgui.slider_float(
                f"{name}##probe_{name.lower()}",
                float(self.position[axis]),
                float(self.slider_lower[axis]),
                float(self.slider_upper[axis]),
                format="%.5f",
            )
            if changed:
                self.position[axis] = value

        for axis, name in enumerate(axis_names):
            if imgui.button(f"-{name}##probe_step_minus_{name.lower()}"):
                self.position[axis] -= self.step
            imgui.same_line()
            if imgui.button(f"+{name}##probe_step_plus_{name.lower()}"):
                self.position[axis] += self.step

        if imgui.button("Center##probe_center"):
            self.position = self.center.copy()

        imgui.text("Place outside collision bounds:")
        for axis, name in enumerate(axis_names):
            if imgui.button(f"-{name} face##probe_face_minus_{name.lower()}"):
                self.position = self.outside_position(axis, -1)
            imgui.same_line()
            if imgui.button(f"+{name} face##probe_face_plus_{name.lower()}"):
                self.position = self.outside_position(axis, 1)

        imgui.separator()
        imgui.text(f"Contacts: {len(self.contacts)}")
        if not self.contacts:
            imgui.text("Hit body: none")
            return

        first = self.contacts[0]
        px, py, pz = first.point_world
        nx, ny, nz = first.normal_probe_to_asset
        imgui.text(f"Hit body: {first.body_index} {first.body_label}")
        imgui.text(f"Hit shape: {first.shape_index} {first.shape_label}")
        imgui.text(f"Point: {px:+.5f}, {py:+.5f}, {pz:+.5f}")
        imgui.text(f"Normal: {nx:+.4f}, {ny:+.4f}, {nz:+.4f}")
        imgui.text(f"Penetration: {first.penetration:.6f} m")


@dataclass(frozen=True)
class ProbeScene:
    model: newton.Model
    state: newton.State
    contacts: Any
    probe_body: int
    probe_shape: int
    asset_shape_count: int
    bounds_lower: np.ndarray
    bounds_upper: np.ndarray


def build_probe_scene(args: argparse.Namespace, urdf_path: Path) -> ProbeScene:
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
    # Disable speculative contacts so the UI reflects the collision surface
    # itself rather than a solver-oriented advance contact distance.
    builder.default_shape_cfg.gap = 0.0
    builder.add_urdf(
        str(urdf_path),
        xform=wp.transform((0.0, 0.0, args.z), wp.quat_identity()),
        floating=False,
        scale=args.scale,
        up_axis=newton.Axis.Z,
        enable_self_collisions=False,
        collapse_fixed_joints=False,
        force_show_colliders=False,
    )
    asset_shape_count = len(builder.shape_type)

    probe_body = builder.add_body(
        xform=wp.transform((0.0, 0.0, 1000.0), wp.quat_identity()),
        mass=1.0,
        is_kinematic=True,
        label=PROBE_LABEL,
    )
    probe_shape = builder.add_shape_sphere(
        body=probe_body,
        radius=args.probe_radius,
        color=(1.0, 0.2, 0.05),
        label=PROBE_LABEL,
    )

    model = builder.finalize()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    contacts = model.contacts()
    model.collide(state, contacts)
    bounds_lower, bounds_upper = collision_bounds(model, asset_shape_count)

    return ProbeScene(
        model=model,
        state=state,
        contacts=contacts,
        probe_body=probe_body,
        probe_shape=probe_shape,
        asset_shape_count=asset_shape_count,
        bounds_lower=bounds_lower,
        bounds_upper=bounds_upper,
    )


def scan_position(
    panel: CollisionProbePanel,
    axis_name: str,
    frame: int,
    frame_count: int,
) -> np.ndarray:
    axis = {"x": 0, "y": 1, "z": 2}[axis_name]
    start = panel.outside_position(axis, -1)
    end = panel.outside_position(axis, 1)
    fraction = 1.0 if frame_count <= 1 else frame / (frame_count - 1)
    return start + fraction * (end - start)


def run_probe(args: argparse.Namespace, urdf_path: Path) -> None:
    scene = build_probe_scene(args, urdf_path)
    extent = scene.bounds_upper - scene.bounds_lower
    diagonal = float(np.linalg.norm(extent))
    step = args.step if args.step is not None else max(0.25 * args.probe_radius, 0.005 * diagonal)

    if args.probe_position is None:
        initial_position = 0.5 * (scene.bounds_lower + scene.bounds_upper)
        initial_position[0] = scene.bounds_upper[0] + 2.0 * args.probe_radius
    else:
        initial_position = np.asarray(args.probe_position, dtype=np.float64)

    panel = CollisionProbePanel(
        initial_position,
        scene.bounds_lower,
        scene.bounds_upper,
        args.probe_radius,
        step,
        args.print_contacts,
    )

    viewer = newton.viewer.ViewerGL(
        width=args.width,
        height=args.height,
        headless=args.headless,
        paused=True,
    )
    viewer.set_model(scene.model)
    viewer.show_visual = True
    viewer.show_collision = True
    viewer.show_contacts = True
    viewer.set_camera(
        wp.vec3(args.camera_x, args.camera_y, args.camera_z),
        args.pitch,
        args.yaw,
    )
    set_double_sided_rendering(viewer, args.double_sided)
    if hasattr(viewer, "register_ui_callback"):
        viewer.register_ui_callback(panel.render_ui, position="side")

    print(f"Opening {describe_urdf_path(urdf_path)}: {urdf_path}")
    print(
        f"Asset collision bounds: lower={scene.bounds_lower.tolist()}, "
        f"upper={scene.bounds_upper.tolist()}, diagonal={diagonal:.6f} m"
    )
    print(f"Probe radius={args.probe_radius:.6f} m, step={step:.6f} m")

    frame = 0
    scan_frames = args.frames if args.frames is not None else args.scan_frames
    while viewer.is_running():
        if args.scan_axis is not None:
            panel.position = scan_position(panel, args.scan_axis, frame, scan_frames)

        set_kinematic_body_position(scene.state, scene.probe_body, panel.position)
        scene.model.collide(scene.state, scene.contacts)
        samples = collect_probe_contacts(
            scene.model,
            scene.state,
            scene.contacts,
            scene.probe_shape,
        )
        panel.update_contacts(samples)

        viewer.begin_frame(0.0)
        viewer.log_state(scene.state)
        viewer.log_contacts(scene.contacts, scene.state)
        viewer.end_frame()

        frame += 1
        if args.frames is not None and frame >= args.frames:
            break
        if args.scan_axis is not None and frame >= args.scan_frames:
            break

    viewer.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe URDF collision geometry with a kinematic sphere.")
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Artiverse root, category/model directory, or one .urdf file.",
    )
    parser.add_argument("--category", action="append", default=None, help="Limit discovery to a category.")
    parser.add_argument("--index", type=int, default=0, help="Asset index within the discovery result.")
    parser.add_argument("--scale", type=float, default=1.0, help="Uniform import scale.")
    parser.add_argument("--z", type=float, default=0.0, help="Asset vertical offset.")
    parser.add_argument("--probe-radius", type=float, default=DEFAULT_PROBE_RADIUS, help="Probe radius in meters.")
    parser.add_argument("--probe-position", type=float, nargs=3, metavar=("X", "Y", "Z"), default=None)
    parser.add_argument("--step", type=float, default=None, help="UI movement step in meters; defaults from asset size.")
    parser.add_argument("--scan-axis", choices=("x", "y", "z"), default=None, help="Automatically scan one axis.")
    parser.add_argument("--scan-frames", type=int, default=DEFAULT_SCAN_FRAMES)
    parser.add_argument(
        "--print-contacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print contact transitions to the terminal.",
    )
    parser.add_argument("--double-sided", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--frames", type=int, default=None, help="Close after N frames.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-x", type=float, default=0.65)
    parser.add_argument("--camera-y", type=float, default=-0.75)
    parser.add_argument("--camera-z", type=float, default=0.45)
    parser.add_argument("--pitch", type=float, default=-25.0)
    parser.add_argument("--yaw", type=float, default=40.0)
    args = parser.parse_args(argv)

    if args.probe_radius <= 0.0 or not math.isfinite(args.probe_radius):
        parser.error("--probe-radius must be a finite positive number")
    if args.step is not None and (args.step <= 0.0 or not math.isfinite(args.step)):
        parser.error("--step must be a finite positive number")
    if args.scan_frames < 2:
        parser.error("--scan-frames must be at least 2")
    if args.frames is not None and args.frames < 1:
        parser.error("--frames must be at least 1")
    if args.probe_position is not None and not all(math.isfinite(value) for value in args.probe_position):
        parser.error("--probe-position values must be finite")
    if args.headless and args.frames is None and args.scan_axis is None:
        parser.error("--headless requires --frames or --scan-axis")
    return args


def main() -> None:
    args = parse_args()
    categories = tuple(args.category) if args.category else None
    urdfs = find_urdfs(args.source, categories)
    if args.index < 0 or args.index >= len(urdfs):
        raise IndexError(f"--index must be between 0 and {len(urdfs) - 1}; got {args.index}")
    run_probe(args, urdfs[args.index])


if __name__ == "__main__":
    main()
