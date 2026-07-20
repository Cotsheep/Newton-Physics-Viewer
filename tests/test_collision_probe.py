from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import collision_probe


BOX_URDF = """\
<?xml version="1.0"?>
<robot name="collision_probe_fixture">
  <link name="box">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
    <visual>
      <geometry><box size="1 1 1"/></geometry>
    </visual>
    <collision>
      <geometry><box size="1 1 1"/></geometry>
    </collision>
  </link>
</robot>
"""


class CollisionProbeTests(unittest.TestCase):
    def test_probe_reports_intersection_but_not_separation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            urdf_path = Path(temp_dir) / "box.urdf"
            urdf_path.write_text(BOX_URDF, encoding="utf-8")
            args = collision_probe.parse_args(
                [str(urdf_path), "--probe-radius", "0.1", "--headless", "--frames", "1"]
            )
            scene = collision_probe.build_probe_scene(args, urdf_path)

            np.testing.assert_allclose(scene.bounds_lower, (-0.5, -0.5, -0.5), atol=1.0e-6)
            np.testing.assert_allclose(scene.bounds_upper, (0.5, 0.5, 0.5), atol=1.0e-6)

            collision_probe.set_kinematic_body_position(
                scene.state,
                scene.probe_body,
                np.array((0.55, 0.0, 0.0)),
            )
            scene.model.collide(scene.state, scene.contacts)
            hits = collision_probe.collect_probe_contacts(
                scene.model,
                scene.state,
                scene.contacts,
                scene.probe_shape,
            )
            self.assertTrue(hits)
            self.assertTrue(all(hit.penetration > 0.0 for hit in hits))

            collision_probe.set_kinematic_body_position(
                scene.state,
                scene.probe_body,
                np.array((1.0, 0.0, 0.0)),
            )
            scene.model.collide(scene.state, scene.contacts)
            hits = collision_probe.collect_probe_contacts(
                scene.model,
                scene.state,
                scene.contacts,
                scene.probe_shape,
            )
            self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
