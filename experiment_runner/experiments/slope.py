from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import newton
import numpy as np
import warp as wp

from asset_viewer.app import build_model
from asset_viewer.camera import AssetBounds, compute_asset_bounds

from ..profiles import ExperimentProfile
from .drop import _viewer_arguments, mujoco_usd_schema_resolvers
from .recording import configure_warp_cpu_only, record_simulation_video


SLOPE_ANGLE_DEGREES = 25.0
INITIAL_HOLD_SECONDS = 0.5


@dataclass(frozen=True)
class SlopeGeometry:
    angle_degrees: float
    characteristic_length: float
    effective_length: float
    down_slope: np.ndarray
    cross_slope: np.ndarray
    surface_normal: np.ndarray
    ramp_center: np.ndarray
    ramp_length: float
    ramp_width: float
    ramp_thickness: float
    surface_gap: float
    asset_translation: np.ndarray
    asset_rotation_xyzw: np.ndarray
    initial_asset_bounds: AssetBounds


@dataclass
class SlopeScene:
    model: newton.Model
    state: newton.State
    state_next: newton.State
    control: Any
    contacts: Any
    solver: newton.solvers.SolverMuJoCo
    ramp_shape_index: int
    asset_min_contact_priority: int
    ramp_contact_priority: int
    tracked_body_index: int
    initial_position: np.ndarray
    sim_time: float = 0.0


def _require_slope_angle(angle_degrees: float) -> float:
    angle = float(angle_degrees)
    if not math.isfinite(angle) or not 0.0 < angle < 90.0:
        raise ValueError("angle_degrees must be finite and between 0 and 90")
    return angle


def slope_basis(angle_degrees: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a right-handed downhill/cross-slope/surface-normal basis."""

    angle = math.radians(_require_slope_angle(angle_degrees))
    down_slope = np.array((math.cos(angle), 0.0, -math.sin(angle)), dtype=np.float64)
    cross_slope = np.array((0.0, 1.0, 0.0), dtype=np.float64)
    surface_normal = np.cross(down_slope, cross_slope)
    return down_slope, cross_slope, surface_normal


def _rotation_y_xyzw(angle_degrees: float) -> np.ndarray:
    half_angle = 0.5 * math.radians(_require_slope_angle(angle_degrees))
    return np.array((0.0, math.sin(half_angle), 0.0, math.cos(half_angle)), dtype=np.float64)


def _quaternion_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion_xyzw, dtype=np.float64)
    return np.array(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def asset_bounds_corners(bounds: AssetBounds) -> np.ndarray:
    return np.array(
        [
            (x, y, z)
            for x in (bounds.minimum[0], bounds.maximum[0])
            for y in (bounds.minimum[1], bounds.maximum[1])
            for z in (bounds.minimum[2], bounds.maximum[2])
        ],
        dtype=np.float64,
    )


def transform_asset_points(
    points: np.ndarray,
    translation: np.ndarray,
    rotation_xyzw: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points @ _quaternion_matrix(rotation_xyzw).T + np.asarray(
        translation,
        dtype=np.float64,
    )


def plan_slope_geometry(
    bounds: AssetBounds,
    *,
    angle_degrees: float = SLOPE_ANGLE_DEGREES,
) -> SlopeGeometry:
    """Plan a finite ramp and place the rotated asset above its top surface."""

    angle = _require_slope_angle(angle_degrees)
    characteristic_length = bounds.max_extent
    if not math.isfinite(characteristic_length) or characteristic_length <= 0.0:
        raise ValueError("Asset collision bounds do not define a positive characteristic length")
    effective_length = float(np.clip(characteristic_length, 0.10, 1.00))
    down_slope, cross_slope, surface_normal = slope_basis(angle)
    ramp_length = max(6.0 * effective_length, 3.0 * float(bounds.extents[0]))
    ramp_width = max(3.0 * effective_length, 1.5 * float(bounds.extents[1]))
    ramp_thickness = max(0.05 * effective_length, 0.02)
    surface_gap = max(0.005 * effective_length, 0.001)

    angle_radians = math.radians(angle)
    lower_surface_height = max(0.05, ramp_thickness)
    ramp_center = np.array(
        (
            0.0,
            0.0,
            lower_surface_height
            + 0.5 * ramp_length * math.sin(angle_radians)
            - 0.5 * ramp_thickness * math.cos(angle_radians),
        ),
        dtype=np.float64,
    )
    down_translation = -0.5 * ramp_length + effective_length - float(bounds.minimum[0])
    cross_translation = -float(bounds.center[1])
    normal_translation = (
        0.5 * ramp_thickness + surface_gap - float(bounds.minimum[2])
    )
    asset_translation = (
        ramp_center
        + down_slope * down_translation
        + cross_slope * cross_translation
        + surface_normal * normal_translation
    )
    return SlopeGeometry(
        angle_degrees=angle,
        characteristic_length=characteristic_length,
        effective_length=effective_length,
        down_slope=down_slope,
        cross_slope=cross_slope,
        surface_normal=surface_normal,
        ramp_center=ramp_center,
        ramp_length=ramp_length,
        ramp_width=ramp_width,
        ramp_thickness=ramp_thickness,
        surface_gap=surface_gap,
        asset_translation=asset_translation,
        asset_rotation_xyzw=_rotation_y_xyzw(angle),
        initial_asset_bounds=bounds,
    )


def measure_slope_geometry(
    asset_path: Path,
    *,
    profile: ExperimentProfile,
    angle_degrees: float = SLOPE_ANGLE_DEGREES,
) -> SlopeGeometry:
    configure_warp_cpu_only()
    arguments = _viewer_arguments(
        asset_path,
        z_offset=0.0,
        profile=profile,
        include_ground=False,
    )
    model, state, _joint_panel = build_model(
        arguments,
        asset_path,
        usd_schema_resolvers=mujoco_usd_schema_resolvers(),
    )
    return plan_slope_geometry(
        compute_asset_bounds(model, state),
        angle_degrees=angle_degrees,
    )


def _wp_transform(translation: np.ndarray, rotation_xyzw: np.ndarray) -> wp.transform:
    return wp.transform(
        wp.vec3(*(float(value) for value in translation)),
        wp.quat(*(float(value) for value in rotation_xyzw)),
    )


def reference_surface_priorities(asset_priorities: list[int]) -> tuple[int, int]:
    """Choose a lower MuJoCo priority without overflowing its int32 storage."""

    if not asset_priorities:
        raise ValueError("Slope smoke requires at least one imported collision shape")
    asset_min_contact_priority = min(int(value) for value in asset_priorities)
    if asset_min_contact_priority <= np.iinfo(np.int32).min:
        raise ValueError(
            "Slope smoke cannot assign the reference surface below the asset's "
            "minimum MuJoCo contact priority"
        )
    return asset_min_contact_priority, asset_min_contact_priority - 1


def create_slope_scene(
    asset_path: Path,
    *,
    profile: ExperimentProfile,
    geometry: SlopeGeometry,
) -> SlopeScene:
    """Create one native-contact MuJoCo CPU scene on a fixed finite ramp."""

    configure_warp_cpu_only()
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    try:
        builder.add_usd(
            str(asset_path),
            xform=_wp_transform(
                geometry.asset_translation,
                geometry.asset_rotation_xyzw,
            ),
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=False,
            force_show_colliders=False,
            schema_resolvers=mujoco_usd_schema_resolvers(),
        )
    except ImportError as exc:
        raise RuntimeError("USD import requires OpenUSD Python bindings (the 'pxr' module)") from exc
    if builder.body_count <= 0:
        raise ValueError("Slope smoke requires at least one imported rigid body")

    priority_attribute = builder.custom_attributes.get("mujoco:geom_priority")
    if priority_attribute is None:
        raise RuntimeError("MuJoCo geom priority metadata is unavailable")
    priority_values = priority_attribute.values or {}
    asset_priorities = [
        int(priority_values.get(index, priority_attribute.default))
        for index in range(builder.shape_count)
    ]
    asset_min_contact_priority, ramp_contact_priority = reference_surface_priorities(
        asset_priorities
    )

    ramp_config = newton.ModelBuilder.ShapeConfig(
        density=0.0,
        # MuJoCo resolves equal-priority contact friction with a component-wise
        # maximum.  The minimum admissible values therefore leave the accepted
        # asset's authored slide/torsion/rolling coefficients in control.
        mu=1.1e-5,
        mu_torsional=1.1e-5,
        mu_rolling=1.1e-5,
        restitution=0.0,
    )
    ramp_shape_index = builder.add_shape_box(
        -1,
        xform=_wp_transform(
            geometry.ramp_center,
            geometry.asset_rotation_xyzw,
        ),
        hx=0.5 * geometry.ramp_length,
        hy=0.5 * geometry.ramp_width,
        hz=0.5 * geometry.ramp_thickness,
        cfg=ramp_config,
        color=wp.vec3(0.36, 0.40, 0.46),
        label="cpu-smoke-ramp-25deg",
        custom_attributes={"mujoco:geom_priority": ramp_contact_priority},
    )
    model = builder.finalize()
    model.joint_qd.zero_()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    state.body_qd.zero_()
    initial_body_q = np.asarray(state.body_q.numpy(), dtype=np.float64)
    initial_position = initial_body_q[0, :3].copy()
    if not np.isfinite(initial_body_q).all():
        raise RuntimeError("Slope scene produced a non-finite initial body state")
    solver = newton.solvers.SolverMuJoCo(
        model,
        iterations=profile.iterations,
        use_mujoco_cpu=profile.use_mujoco_cpu,
        use_mujoco_contacts=profile.use_mujoco_contacts,
    )
    return SlopeScene(
        model=model,
        state=state,
        state_next=model.state(),
        control=model.control(),
        contacts=model.contacts(),
        solver=solver,
        ramp_shape_index=ramp_shape_index,
        asset_min_contact_priority=asset_min_contact_priority,
        ramp_contact_priority=ramp_contact_priority,
        tracked_body_index=0,
        initial_position=initial_position,
    )


def step_slope_scene(scene: SlopeScene, profile: ExperimentProfile) -> None:
    scene.state.clear_forces()
    scene.solver.step(
        scene.state,
        scene.state_next,
        scene.control,
        scene.contacts,
        profile.physics_dt,
    )
    scene.state, scene.state_next = scene.state_next, scene.state
    scene.sim_time += profile.physics_dt


def measure_along_slope_displacement(
    initial_position: np.ndarray,
    final_position: np.ndarray,
    down_slope: np.ndarray,
) -> float:
    values = np.concatenate(
        (
            np.asarray(initial_position, dtype=np.float64).reshape(3),
            np.asarray(final_position, dtype=np.float64).reshape(3),
            np.asarray(down_slope, dtype=np.float64).reshape(3),
        )
    )
    if not np.isfinite(values).all():
        raise ValueError("Slope displacement inputs must be finite")
    return float(
        np.dot(
            np.asarray(final_position, dtype=np.float64)
            - np.asarray(initial_position, dtype=np.float64),
            np.asarray(down_slope, dtype=np.float64),
        )
    )


def classify_development_outcome(
    along_slope_displacement: float,
    *,
    effective_length: float,
) -> str:
    displacement = float(along_slope_displacement)
    length = float(effective_length)
    if not math.isfinite(displacement) or not math.isfinite(length):
        raise ValueError("Slope outcome inputs must be finite")
    if length <= 0.0:
        raise ValueError("effective_length must be greater than zero")
    if displacement >= 0.05 * length:
        return "moved"
    if abs(displacement) <= 0.02 * length:
        return "stayed_near_start"
    return "inconclusive"


def _require_finite_state(scene: SlopeScene) -> tuple[np.ndarray, np.ndarray]:
    body_q = np.asarray(scene.state.body_q.numpy(), dtype=np.float64)
    body_qd = np.asarray(scene.state.body_qd.numpy(), dtype=np.float64)
    if not np.isfinite(body_q).all() or not np.isfinite(body_qd).all():
        raise RuntimeError("MuJoCo produced a non-finite body state")
    return body_q, body_qd


def record_slope_case(
    scene: SlopeScene,
    *,
    profile: ExperimentProfile,
    geometry: SlopeGeometry,
    output_directory: Path,
    duration_seconds: float,
    initial_hold_seconds: float = INITIAL_HOLD_SECONDS,
) -> dict[str, Any]:
    initial_scene_bounds = compute_asset_bounds(scene.model, scene.state)
    recording = record_simulation_video(
        scene,
        profile=profile,
        output_directory=output_directory,
        duration_seconds=duration_seconds,
        initial_hold_seconds=initial_hold_seconds,
        camera_bounds=initial_scene_bounds,
        step_scene=step_slope_scene,
        validate_scene=lambda current_scene: _require_finite_state(current_scene),
        camera_pitch=-24.0,
        camera_yaw=38.0,
        camera_padding=1.35,
    )

    body_q, body_qd = _require_finite_state(scene)
    final_position = body_q[scene.tracked_body_index, :3].copy()
    final_linear_velocity = body_qd[scene.tracked_body_index, :3].copy()
    displacement = measure_along_slope_displacement(
        scene.initial_position,
        final_position,
        geometry.down_slope,
    )
    return {
        **recording,
        "slope_angle_degrees": geometry.angle_degrees,
        "initial_position": scene.initial_position.tolist(),
        "final_position": final_position.tolist(),
        "displacement_along_slope": displacement,
        "final_linear_velocity": final_linear_velocity.tolist(),
        "measurement_units": {
            "position": "m",
            "displacement": "m",
            "linear_velocity": "m/s",
        },
        "finite": True,
        "development_outcome": classify_development_outcome(
            displacement,
            effective_length=geometry.effective_length,
        ),
        "contact_policy": {
            "native_mujoco_contacts": True,
            "asset_min_priority": scene.asset_min_contact_priority,
            "reference_surface_priority": scene.ramp_contact_priority,
            "asset_controls_contact_parameters": True,
        },
    }
