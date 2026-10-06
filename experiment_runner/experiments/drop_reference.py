"""Reference conditions and visual-only scale markers for drop validation."""
from dataclasses import dataclass

import newton
import numpy as np
import warp as wp

from asset_viewer.camera import AssetBounds, compute_asset_bounds


def set_neutral_ground_priority(model) -> None:
    bodies = model.shape_body.numpy()
    flags = model.shape_flags.numpy()
    kinds = model.shape_type.numpy()
    enabled = (flags & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0
    asset_ids = np.flatnonzero(enabled & (bodies >= 0))
    ground_ids = np.flatnonzero(enabled & (bodies < 0))
    if len(ground_ids) != 1 or kinds[ground_ids[0]] != int(newton.GeoType.PLANE) or not len(asset_ids):
        raise ValueError("Drop validation requires asset colliders and exactly one static plane")
    priorities = model.mujoco.geom_priority.numpy()
    minimum = int(np.min(priorities[asset_ids]))
    if minimum <= np.iinfo(np.int32).min:
        raise ValueError("Reference plane priority cannot be lowered safely")
    priorities[ground_ids[0]] = minimum - 1
    model.mujoco.geom_priority.assign(priorities)


@dataclass
class DropScaleReference:
    starts: np.ndarray
    ends: np.ndarray
    camera_bounds: AssetBounds
    _arrays: object = None

    def draw(self, viewer):
        if self._arrays is None:
            self._arrays = (wp.array(self.starts, dtype=wp.vec3, device=viewer.device),
                            wp.array(self.ends, dtype=wp.vec3, device=viewer.device))
        viewer.log_lines("/drop-validation/scale-meters", *self._arrays, colors=(1.0, 0.8, 0.2))


def scale_reference(bounds: AssetBounds, effective_length: float) -> DropScaleReference:
    """A metre-labelled vertical ruler plus a square of side L_eff, outside physics."""
    length = float(effective_length)
    if not np.isfinite(length) or length <= 0:
        raise ValueError("Scale length must be finite and positive")
    # Keep labels legible when a small asset shares the one-metre framing.
    marker_length = max(length, 0.5)
    x, y = float(bounds.minimum[0] - 0.5 * marker_length), float(bounds.center[1])
    segments = []
    def line(a, b):
        segments.append((a, b))
    release_span = max(2.0 * length, 1.0)
    top = release_span + float(bounds.extents[2])
    line((x, y, 0.0), (x, y, top))
    # Seven-segment numeric labels in world coordinates, measured in metres.
    glyphs = {'0':'abcdef', '1':'bc', '2':'abged', '3':'abgcd', '4':'fgbc',
              '5':'afgcd', '6':'afgecd', '7':'abc', '8':'abcdefg', '9':'abfgcd'}
    strokes = {'a':((0,2),(1,2)), 'b':((1,2),(1,1)), 'c':((1,1),(1,0)),
               'd':((1,0),(0,0)), 'e':((0,0),(0,1)), 'f':((0,1),(0,2)), 'g':((0,1),(1,1))}
    unit = 0.035 * marker_length
    ticks = np.arange(int(np.floor(release_span / (0.5 * length))) + 1) * 0.5 * length
    if not np.isclose(ticks, 1.0, rtol=0.0, atol=1e-8).any():
        ticks = np.sort(np.append(ticks, 1.0))
    for z in ticks:
        line((x-0.08*marker_length,y,z), (x+0.08*marker_length,y,z))
        # Keep the 1 m label clear when a relative tick falls almost on it.
        if abs(z - 1.0) < 2.5 * unit and not np.isclose(z, 1.0, rtol=0.0, atol=1e-8):
            continue
        label = (f'{z:.3f}'.rstrip('0').rstrip('.') or '0') + 'm'
        for index, char in enumerate(label):
            origin = x - (len(label)-index) * 1.5 * unit - 0.12 * marker_length
            if char == '.':
                parts = [((0,0),(0.2,0))]
            elif char == 'm':
                parts = [((0,0),(0,1)), ((0,1),(.5,.5)), ((.5,.5),(1,1)), ((1,1),(1,0))]
            else:
                parts = [strokes[s] for s in glyphs[char]]
            for a,b in parts:
                line((origin+a[0]*unit,y,z+(a[1]-1)*unit),
                     (origin+b[0]*unit,y,z+(b[1]-1)*unit))
    # Non-colliding metre scale on the ground, raised slightly to avoid z fighting.
    corners = [(x-length,y-length,0.001*length),(x,y-length,0.001*length),
               (x,y,0.001*length),(x-length,y,0.001*length)]
    for a,b in zip(corners,corners[1:]+corners[:1]):
        line(a,b)
    starts, ends = (np.asarray(v, dtype=np.float32) for v in zip(*segments))
    points = np.concatenate((starts,ends,[bounds.minimum,bounds.maximum]))
    # Identical framing across relative heights and fixed 1 m for the same asset pose.
    minimum, maximum = points.min(axis=0), points.max(axis=0)
    minimum[2] = 0.0
    maximum[2] = top
    return DropScaleReference(starts, ends, AssetBounds(minimum,maximum))


def attach_drop_reference(scene, geometry) -> dict:
    observer = scene.drop_observer
    if observer is None or observer.reason:
        raise ValueError("Drop validation requires a working drop observer")
    model = scene.solver.mj_model
    ground = observer.geometry.ground_id
    ids = [s.geom_id for s in observer.geometry.shapes]
    if not all(model.geom_priority[ground] < model.geom_priority[i] for i in ids):
        raise ValueError("Compiled reference plane does not preserve asset contact priority")
    bounds = compute_asset_bounds(scene.model, scene.state)
    scene.scale_reference = scale_reference(bounds, geometry.effective_length)
    return {
        "reference_policy": "asset_priority_over_ground_v1",
        "reference_ground_priority": int(model.geom_priority[ground]),
        "asset_min_contact_priority": int(min(model.geom_priority[i] for i in ids)),
        "actual_initial_clearance_m": observer.geometry.measure(scene.state.body_q.numpy()),
        "initial_body_poses_xyzw": scene.state.body_q.numpy().tolist(),
        "initial_joint_positions": scene.model.joint_q.numpy().tolist(),
        "scale_reference": {"kind": "visual_only_meter_ruler_and_square_v1", "label_unit": "m",
                            "tick_interval_m": 0.5 * geometry.effective_length,
                            "ruler_top_m": float(scene.scale_reference.camera_bounds.maximum[2]),
                            "fixed_height_tick_m": 1.0,
                            "square_side_m": geometry.effective_length, "affects_physics": False},
    }
