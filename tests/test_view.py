from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import view


MINIMAL_ARTICULATED_URDF = """\
<?xml version="1.0"?>
<robot name="pose_fidelity_fixture">
  <link name="base">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
  <link name="door_release_button">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
  <link name="door_panel">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
  <joint name="door_release" type="prismatic">
    <parent link="base"/>
    <child link="door_release_button"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="0.01" effort="10" velocity="1"/>
  </joint>
  <joint name="door_hinge" type="revolute">
    <parent link="base"/>
    <child link="door_panel"/>
    <axis xyz="0 0 1"/>
    <limit lower="0" upper="1.57" effort="10" velocity="1"/>
  </joint>
</robot>
"""


class AssetPoseFidelityTests(unittest.TestCase):
    def test_build_model_preserves_urdf_joint_pose_regardless_of_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            urdf_path = Path(temp_dir) / "asset.urdf"
            urdf_path.write_text(MINIMAL_ARTICULATED_URDF, encoding="utf-8")
            args = view.parse_args([str(urdf_path), "--simulate"])

            model, state, _joint_panel = view.build_model(args, urdf_path)

            np.testing.assert_array_equal(model.joint_q.numpy(), np.zeros(model.joint_coord_count))
            np.testing.assert_array_equal(state.joint_q.numpy(), np.zeros(model.joint_coord_count))

    def test_simulation_defaults_to_paused_original_pose(self) -> None:
        args = view.parse_args([])
        self.assertFalse(args.start_running)


if __name__ == "__main__":
    unittest.main()
