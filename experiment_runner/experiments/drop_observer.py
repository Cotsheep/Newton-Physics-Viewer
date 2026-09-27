"""Read-only diagnostics of compiled collision geometry against one Z-up plane.

Uses completed Newton body poses, not MuJoCo's potentially pre-integration
geom_xpos. Compiled mesh coordinates and initial geom poses include the mesh
centering/rotation performed by MuJoCo. No solver state is modified.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import mujoco
import numpy as np

from ..drop_metrics import DropAccumulator, DROP_KEYS


class UnsupportedDropGeometry(ValueError):
    pass


def rotation_xyzw(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.dot(q, q))
    if q.shape != (4,) or not np.isfinite(q).all() or norm < 1e-20:
        raise ValueError("Invalid body rotation")
    x, y, z, w = q / math.sqrt(norm)
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)],
    ])


@dataclass
class CollisionShape:
    geom_id: int
    body_index: int
    kind: int
    size: np.ndarray
    local_position: np.ndarray
    local_rotation: np.ndarray
    vertices: np.ndarray | None = None

    def minimum_z(self, position: np.ndarray, rotation: np.ndarray) -> float:
        center_z = float(position[2] + rotation[2] @ self.local_position)
        vertical = rotation[2] @ self.local_rotation
        geo = mujoco.mjtGeom
        if self.kind == geo.mjGEOM_MESH:
            return center_z + float(np.min(self.vertices @ vertical))
        if self.kind == geo.mjGEOM_SPHERE:
            extent = self.size[0]
        elif self.kind == geo.mjGEOM_BOX:
            extent = np.abs(vertical) @ self.size
        elif self.kind == geo.mjGEOM_ELLIPSOID:
            extent = np.linalg.norm(vertical * self.size)
        elif self.kind == geo.mjGEOM_CAPSULE:
            extent = self.size[0] + self.size[1] * abs(vertical[2])
        elif self.kind == geo.mjGEOM_CYLINDER:
            extent = self.size[0] * np.linalg.norm(vertical[:2]) + self.size[1] * abs(vertical[2])
        else:
            raise UnsupportedDropGeometry("unsupported_collision_shape")
        return center_z - float(extent)


class CollisionClearance:
    def __init__(self, scene: Any):
        model, data = scene.solver.mj_model, scene.solver.mj_data
        if not isinstance(model, mujoco.MjModel) or not isinstance(data, mujoco.MjData):
            raise UnsupportedDropGeometry("compiled_mujoco_model_unavailable")
        mapping = np.asarray(scene.solver.mjc_body_to_newton.numpy())
        if mapping.shape != (1, model.nbody) or model.npair:
            raise UnsupportedDropGeometry("requires_one_world_without_explicit_contact_pairs")
        if model.nflex:
            raise UnsupportedDropGeometry("deformable_geometry_not_supported")
        enabled = [i for i in range(model.ngeom) if model.geom_contype[i] or model.geom_conaffinity[i]]
        ground = [i for i in enabled if model.geom_bodyid[i] == 0]
        if len(ground) != 1 or model.geom_type[ground[0]] != mujoco.mjtGeom.mjGEOM_PLANE:
            raise UnsupportedDropGeometry("requires_one_static_ground_plane")
        ground_id = ground[0]
        if not np.allclose(data.geom_xmat[ground_id].reshape(3, 3)[:, 2], [0, 0, 1], atol=1e-7):
            raise UnsupportedDropGeometry("requires_horizontal_z_up_ground")
        self.ground_z = float(data.geom_xpos[ground_id, 2])
        self.ground_id = ground_id
        self.shapes: list[CollisionShape] = []
        body_q = np.asarray(scene.state.body_q.numpy(), dtype=np.float64)
        allowed = {mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_BOX,
                   mujoco.mjtGeom.mjGEOM_ELLIPSOID, mujoco.mjtGeom.mjGEOM_CAPSULE,
                   mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_MESH}
        for geom_id in enabled:
            if geom_id == ground_id:
                continue
            body = int(mapping[0, model.geom_bodyid[geom_id]])
            if not 0 <= body < len(body_q):
                raise UnsupportedDropGeometry("unmapped_asset_body")
            # An excluded asset collider cannot be silently dropped from the
            # whole-asset measurement; the definition would become misleading.
            if not (model.geom_contype[geom_id] & model.geom_conaffinity[ground_id]
                    or model.geom_contype[ground_id] & model.geom_conaffinity[geom_id]):
                raise UnsupportedDropGeometry("asset_collider_excludes_ground")
            body_id = int(model.geom_bodyid[geom_id])
            if any(int(signature) == body_id for signature in model.exclude_signature):
                raise UnsupportedDropGeometry("asset_body_excludes_ground")
            kind = int(model.geom_type[geom_id])
            if kind not in allowed:
                raise UnsupportedDropGeometry("unsupported_collision_shape")
            rotation = rotation_xyzw(body_q[body, 3:])
            vertices = None
            if kind == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = int(model.geom_dataid[geom_id])
                start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
                vertices = np.asarray(model.mesh_vert[start:start + count], dtype=np.float64).copy()
                if not count or not np.isfinite(vertices).all():
                    raise UnsupportedDropGeometry("invalid_compiled_mesh")
                # Match the convex support map used by the pinned MuJoCo Warp
                # collision implementation, including a reduced compiled hull.
                graph_start = int(model.mesh_graphadr[mesh])
                if graph_start >= 0 and count >= 10:
                    hull_count = int(model.mesh_graph[graph_start])
                    ids = model.mesh_graph[graph_start + 2 + hull_count:graph_start + 2 + 2*hull_count]
                    if hull_count <= 0 or len(ids) != hull_count or np.any(ids < 0) or np.any(ids >= count):
                        raise UnsupportedDropGeometry("invalid_compiled_hull")
                    vertices = vertices[ids]
            self.shapes.append(CollisionShape(
                geom_id, body, kind, model.geom_size[geom_id].copy(),
                rotation.T @ (data.geom_xpos[geom_id] - body_q[body, :3]),
                rotation.T @ data.geom_xmat[geom_id].reshape(3, 3), vertices,
            ))
        if not self.shapes:
            raise UnsupportedDropGeometry("no_asset_collision_geometry")

    def measure(self, body_q: np.ndarray) -> float:
        if not np.isfinite(body_q).all():
            raise ValueError("Non-finite body state")
        transforms = {shape.body_index: None for shape in self.shapes}
        for body in transforms:
            transforms[body] = (body_q[body, :3], rotation_xyzw(body_q[body, 3:]))
        value = min(shape.minimum_z(*transforms[shape.body_index]) for shape in self.shapes) - self.ground_z
        if not math.isfinite(value):
            raise ValueError("Non-finite clearance")
        return value


class DropObserver:
    def __init__(self, scene: Any, physics_dt: float):
        started = time.perf_counter()
        self.reason: str | None = None
        self.sampling_wall_seconds = 0.0
        self.accumulator = DropAccumulator(physics_dt)
        self.geometry = None
        try:
            if scene.sim_time != 0.0:
                raise UnsupportedDropGeometry("requires_fresh_scene")
            self.geometry = CollisionClearance(scene)
        except UnsupportedDropGeometry as exc:
            self.reason = str(exc)
        self.setup_wall_seconds = time.perf_counter() - started
        self.sample(scene)

    def sample(self, scene: Any) -> None:
        if self.reason:
            return
        started = time.perf_counter()
        try:
            self.accumulator.add(scene.sim_time, self.geometry.measure(
                np.asarray(scene.state.body_q.numpy(), dtype=np.float64)))
        except ValueError:
            self.accumulator.invalid = True
        finally:
            self.sampling_wall_seconds += time.perf_counter() - started

    def result(self, duration: float) -> dict[str, Any]:
        audit = {
            "geometry_source": "compiled_mujoco_collision_geometry",
            "pose_source": "completed_newton_body_state",
            "diagnostic_compute": "cpu_numpy_read_only",
            "setup_wall_seconds": self.setup_wall_seconds,
            "sampling_wall_seconds": self.sampling_wall_seconds,
        }
        if self.reason:
            audit["reason_code"] = self.reason
            observations = {key: {"status": "unsupported", "value": None,
                                 "unit": "1" if key.endswith("ratio") else "m",
                                 "reason_code": self.reason} for key in DROP_KEYS}
        else:
            audit.update(ground_geom_id=self.geometry.ground_id, ground_z_m=self.geometry.ground_z,
                         asset_geom_ids=[shape.geom_id for shape in self.geometry.shapes])
            observations = self.accumulator.result(duration)
        return {"observations": observations, "drop_measurement": audit}
