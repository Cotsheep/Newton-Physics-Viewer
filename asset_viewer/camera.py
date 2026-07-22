"""Asset-aware camera framing and fine navigation controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import newton
import numpy as np


@dataclass(frozen=True)
class AssetBounds:
    """World-space axis-aligned bounds for the imported asset (not the ground)."""

    minimum: np.ndarray
    maximum: np.ndarray

    def __post_init__(self) -> None:
        minimum = np.asarray(self.minimum, dtype=np.float64).reshape(3)
        maximum = np.asarray(self.maximum, dtype=np.float64).reshape(3)
        if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
            raise ValueError("Asset bounds must be finite")
        if np.any(maximum < minimum):
            raise ValueError("Asset bounds maximum must not be below minimum")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    @property
    def center(self) -> np.ndarray:
        return (self.minimum + self.maximum) * 0.5

    @property
    def extents(self) -> np.ndarray:
        return self.maximum - self.minimum

    @property
    def max_extent(self) -> float:
        return float(np.max(self.extents))

    @property
    def radius(self) -> float:
        return float(np.linalg.norm(self.extents) * 0.5)


def _quat_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Return a rotation matrix for a Newton/Warp xyzw quaternion."""
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = x * x + y * y + z * z + w * w
    if norm <= 1.0e-20:
        return np.eye(3)
    scale = 2.0 / norm
    return np.array(
        (
            (1.0 - scale * (y * y + z * z), scale * (x * y - z * w), scale * (x * z + y * w)),
            (scale * (x * y + z * w), 1.0 - scale * (x * x + z * z), scale * (y * z - x * w)),
            (scale * (x * z - y * w), scale * (y * z + x * w), 1.0 - scale * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _shape_local_bounds(model: Any, index: int, shape_type: int, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    geo = newton.GeoType
    scale = np.abs(np.asarray(scale, dtype=np.float64))

    if shape_type in (int(geo.MESH), int(geo.CONVEX_MESH)):
        source = model.shape_source[index]
        if source is None or not hasattr(source, "vertices"):
            return None
        vertices = np.asarray(source.vertices, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
            return None
        vertices = vertices * scale
        return vertices.min(axis=0), vertices.max(axis=0)

    if shape_type == int(geo.BOX):
        half_extents = scale
    elif shape_type == int(geo.SPHERE):
        half_extents = np.full(3, scale[0])
    elif shape_type == int(geo.ELLIPSOID):
        half_extents = scale
    elif shape_type == int(geo.CAPSULE):
        radius, half_height = scale[0], scale[1]
        half_extents = np.array((radius, radius, half_height + radius))
    elif shape_type in (int(geo.CYLINDER), int(geo.CONE)):
        half_extents = np.array((scale[0], scale[0], scale[1]))
    elif shape_type == int(geo.GAUSSIAN):
        half_extents = scale
    else:
        # Plane and height-field shapes are environmental geometry and should
        # not make a small asset look hundreds of metres wide.
        return None

    return -half_extents, half_extents


def _aabb_corners(minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    return np.array(
        [
            (x, y, z)
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ],
        dtype=np.float64,
    )


def compute_asset_bounds(model: Any, state: Any) -> AssetBounds:
    """Compute shape-accurate world bounds while excluding the ground plane."""
    shape_types = np.asarray(model.shape_type.numpy()).reshape(-1)
    shape_scales = np.asarray(model.shape_scale.numpy(), dtype=np.float64)
    shape_transforms = np.asarray(model.shape_transform.numpy(), dtype=np.float64)
    shape_bodies = np.asarray(model.shape_body.numpy()).reshape(-1)
    body_transforms = np.asarray(state.body_q.numpy(), dtype=np.float64)

    world_points: list[np.ndarray] = []
    for index, raw_shape_type in enumerate(shape_types):
        local_bounds = _shape_local_bounds(model, index, int(raw_shape_type), shape_scales[index])
        if local_bounds is None:
            continue

        local_minimum, local_maximum = local_bounds
        corners = _aabb_corners(local_minimum, local_maximum)
        shape_transform = shape_transforms[index]
        shape_rotation = _quat_matrix(shape_transform[3:7])
        shape_translation = shape_transform[:3]
        corners = corners @ shape_rotation.T + shape_translation

        body_index = int(shape_bodies[index])
        if body_index >= 0:
            body_transform = body_transforms[body_index]
            body_rotation = _quat_matrix(body_transform[3:7])
            corners = corners @ body_rotation.T + body_transform[:3]
        world_points.append(corners)

    if world_points:
        points = np.concatenate(world_points, axis=0)
        return AssetBounds(points.min(axis=0), points.max(axis=0))

    # Degenerate assets can still be framed around their bodies. A small
    # non-zero extent prevents a singular camera distance.
    if len(body_transforms):
        centers = body_transforms[:, :3]
        minimum = centers.min(axis=0)
        maximum = centers.max(axis=0)
        if np.allclose(minimum, maximum):
            minimum = minimum - 0.05
            maximum = maximum + 0.05
        return AssetBounds(minimum, maximum)
    return AssetBounds(np.full(3, -0.05), np.full(3, 0.05))


def recommended_camera_speed(bounds: AssetBounds) -> float:
    """Choose a keyboard speed in metres/second that follows asset scale."""
    return float(np.clip(bounds.max_extent * 1.5, 0.01, 4.0))


def frame_camera_on_bounds(camera: Any, bounds: AssetBounds, padding: float = 1.35) -> None:
    """Keep the current viewing direction but place the asset in front and in frame."""
    if padding <= 0.0:
        raise ValueError("Camera padding must be greater than zero")

    radius = max(bounds.radius, 1.0e-4)
    half_fov = np.radians(max(float(camera.fov), 1.0) * 0.5)
    distance = radius * padding / max(np.sin(half_fov), 1.0e-4)
    distance = max(distance, float(camera.near) * 4.0, float(camera.MIN_PIVOT_DISTANCE))
    front = np.asarray(camera.get_front(), dtype=np.float64)
    position = bounds.center - front * distance
    camera.pos = camera._as_vec3(position)
    camera.look_at(bounds.center)


class CameraControlPanel:
    """Asset-scale navigation settings layered on top of Newton ViewerGL."""

    DEFAULT_ORBIT_SENSITIVITY = 0.1
    DEFAULT_DOLLY_DRAG_SENSITIVITY = 0.01

    def __init__(
        self,
        viewer: Any,
        *,
        auto_frame: bool = True,
        padding: float = 1.35,
        speed: float | None = None,
        wheel_sensitivity: float = 0.08,
        fine_scale: float = 0.1,
    ) -> None:
        self.viewer = viewer
        self.auto_frame = auto_frame
        self.padding = padding
        self.requested_speed = speed
        self.base_speed = speed or 1.0
        self.wheel_sensitivity = wheel_sensitivity
        self.fine_scale = fine_scale
        self.fine_mode = False
        self.bounds: AssetBounds | None = None
        self.model: Any | None = None
        self.state: Any | None = None

        gui = getattr(viewer, "gui", None)
        if gui is not None:
            # Newton's built-in F framing uses body origins and clamps the scene
            # to one metre. Replace it with the same shape-aware framing as load.
            gui.frame_camera_on_model = self.frame_asset

    def set_asset(self, bounds: AssetBounds, model: Any | None = None, state: Any | None = None) -> None:
        self.bounds = bounds
        self.model = model
        self.state = state
        self.base_speed = self.requested_speed or recommended_camera_speed(bounds)
        self.apply_sensitivity()
        if self.auto_frame:
            self.frame_asset()

    def frame_asset(self) -> None:
        if self.model is not None and self.state is not None:
            self.bounds = compute_asset_bounds(self.model, self.state)
        if self.bounds is None:
            return
        frame_camera_on_bounds(self.viewer.camera, self.bounds, self.padding)
        if hasattr(self.viewer, "_camera_dirty"):
            self.viewer._camera_dirty = True

    def set_state(self, state: Any) -> None:
        """Track the current double-buffered simulation state for F framing."""
        self.state = state

    def _shift_down(self) -> bool:
        try:
            import pyglet

            renderer = self.viewer.renderer
            return bool(
                renderer.is_key_down(pyglet.window.key.LSHIFT)
                or renderer.is_key_down(pyglet.window.key.RSHIFT)
            )
        except Exception:
            return False

    def apply_sensitivity(self) -> None:
        gui = getattr(self.viewer, "gui", None)
        if gui is None:
            return
        multiplier = self.fine_scale if self.fine_mode or self._shift_down() else 1.0
        gui._cam_speed = self.base_speed * multiplier
        gui._camera_dolly_scroll_sensitivity = self.wheel_sensitivity * multiplier
        gui._camera_orbit_sensitivity = self.DEFAULT_ORBIT_SENSITIVITY * multiplier
        gui._camera_dolly_drag_sensitivity = self.DEFAULT_DOLLY_DRAG_SENSITIVITY * multiplier

    def render_ui(self, imgui: Any) -> None:
        if not imgui.collapsing_header("Camera Navigation"):
            return
        if imgui.button("Frame Asset (F)##camera_frame"):
            self.frame_asset()
        changed, self.fine_mode = imgui.checkbox("Fine Camera Mode##camera_fine", self.fine_mode)
        if changed:
            self.apply_sensitivity()
        changed, speed = imgui.slider_float(
            "Move Speed (m/s)##camera_speed", self.base_speed, 0.005, 4.0, format="%.3f"
        )
        if changed:
            self.base_speed = max(float(speed), 0.005)
            self.apply_sensitivity()
        changed, wheel = imgui.slider_float(
            "Wheel Sensitivity##camera_wheel", self.wheel_sensitivity, 0.005, 0.3, format="%.3f"
        )
        if changed:
            self.wheel_sensitivity = max(float(wheel), 0.005)
            self.apply_sensitivity()
        imgui.text("Hold Shift for temporary fine movement")
        imgui.text("WASD/QE move; wheel dollies; middle drag orbits")
