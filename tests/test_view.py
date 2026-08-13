from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import numpy as np

import view


TEST_FIXTURES = Path(__file__).parent / "fixtures"


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

MALFORMED_URDF = """\
<?xml version="1.0"?>
<robot name="malformed">
  <link name="base">
</robot>
"""


class CopyableModelIdTests(unittest.TestCase):
    def test_urdf_model_id_uses_spaces_between_path_components(self) -> None:
        urdf_path = (
            Path("Datasets")
            / "Artiverse"
            / "dataset_chunks"
            / "data"
            / "microwave"
            / "3dc200"
            / "7e_000"
            / "urdf_w_collider"
            / "7e_000.urdf"
        )

        self.assertEqual(view.copyable_model_id_from_urdf(urdf_path), "microwave 3dc200 7e_000")
        self.assertEqual(view.copyable_model_id_from_asset(urdf_path), "microwave 3dc200 7e_000")


class AssetPoseFidelityTests(unittest.TestCase):
    def test_build_model_preserves_urdf_joint_pose_regardless_of_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            urdf_path = Path(temp_dir) / "asset.urdf"
            urdf_path.write_text(MINIMAL_ARTICULATED_URDF, encoding="utf-8")
            args = view.parse_args(
                [str(urdf_path), "--simulate", "--no-floating", "--no-collapse-fixed-joints", "--z", "0"]
            )

            model, state, _joint_panel = view.build_model(args, urdf_path)

            np.testing.assert_array_equal(model.joint_q.numpy(), np.zeros(model.joint_coord_count))
            np.testing.assert_array_equal(state.joint_q.numpy(), np.zeros(model.joint_coord_count))

    def test_simulation_defaults_to_paused_original_pose(self) -> None:
        args = view.parse_args([])
        self.assertFalse(args.start_running)

    def test_ground_is_enabled_by_default_and_can_be_disabled(self) -> None:
        self.assertTrue(view.parse_args([]).ground)
        self.assertFalse(view.parse_args(["--no-ground"]).ground)

    def test_glb_builds_as_a_dynamic_free_rigid_body(self) -> None:
        import newton
        import trimesh

        with tempfile.TemporaryDirectory() as temp_dir:
            glb_path = Path(temp_dir) / "box.glb"
            trimesh.creation.box(extents=(0.2, 0.3, 0.4)).export(glb_path)
            # A standalone rigid GLB must not accidentally parse neighboring
            # articulation metadata intended for a different workflow.
            glb_path.with_suffix(".articulations.json").write_text("\ufeff{}", encoding="utf-8")
            args = view.parse_args([str(glb_path), "--glb-mass", "2.5"])

            model, _state, joint_panel = view.build_model(args, glb_path)

            self.assertEqual(model.body_count, 1)
            self.assertEqual(model.joint_count, 1)
            self.assertEqual(int(model.joint_type.numpy()[0]), int(newton.JointType.FREE))
            self.assertEqual(int(model.body_flags.numpy()[0]), int(newton.BodyFlags.DYNAMIC))
            self.assertAlmostEqual(float(model.body_mass.numpy()[0]), 2.5)
            self.assertTrue(int(model.shape_flags.numpy()[0]) & int(newton.ShapeFlags.COLLIDE_SHAPES))
            self.assertTrue(int(model.shape_flags.numpy()[0]) & int(newton.ShapeFlags.VISIBLE))
            self.assertEqual(joint_panel.controls, [])

    def test_convex_glb_mode_keeps_visual_and_collision_shapes(self) -> None:
        import newton
        import trimesh

        with tempfile.TemporaryDirectory() as temp_dir:
            glb_path = Path(temp_dir) / "box.glb"
            trimesh.creation.box().export(glb_path)
            args = view.parse_args([str(glb_path), "--glb-collision", "convex"])

            model, _state, _joint_panel = view.build_model(args, glb_path)

            shape_flags = [int(flags) for flags in model.shape_flags.numpy()]
            self.assertTrue(any(flags & int(newton.ShapeFlags.VISIBLE) for flags in shape_flags))
            self.assertTrue(any(flags & int(newton.ShapeFlags.COLLIDE_SHAPES) for flags in shape_flags))

    def test_usd_import_uses_physics_settings_and_authored_root_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            usd_path = Path(temp_dir) / "asset.usda"
            usd_path.write_text("#usda 1.0\n", encoding="utf-8")
            args = view.parse_args(
                [
                    str(usd_path),
                    "--self-collisions",
                    "--collapse-fixed-joints",
                    "--show-colliders",
                ]
            )

            def add_test_body(builder, _source, **_kwargs):
                builder.add_body(mass=1.0, label="usd_test_body")
                return {}

            with mock.patch(
                "asset_viewer.app.newton.ModelBuilder.add_usd",
                autospec=True,
                side_effect=add_test_body,
            ) as add_usd:
                view.build_model(args, usd_path)

            kwargs = add_usd.call_args.kwargs
            self.assertIsNone(kwargs["floating"])
            self.assertTrue(kwargs["enable_self_collisions"])
            self.assertTrue(kwargs["collapse_fixed_joints"])
            self.assertTrue(kwargs["force_show_colliders"])
            self.assertIsNone(kwargs["schema_resolvers"])

    def test_usd_root_mode_maps_floating_and_fixed_to_newton(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            usd_path = Path(temp_dir) / "asset.usda"
            usd_path.write_text("#usda 1.0\n", encoding="utf-8")

            for root_mode, expected in (("floating", True), ("fixed", False)):
                with self.subTest(root_mode=root_mode):
                    args = view.parse_args(
                        [str(usd_path), "--usd-root-mode", root_mode, "--no-ground"]
                    )
                    def add_test_body(builder, _source, **_kwargs):
                        builder.add_body(mass=1.0, label="usd_test_body")
                        return {}

                    with mock.patch(
                        "asset_viewer.app.newton.ModelBuilder.add_usd",
                        autospec=True,
                        side_effect=add_test_body,
                    ) as add_usd:
                        view.build_model(args, usd_path)

                    self.assertIs(add_usd.call_args.kwargs["floating"], expected)


class AssetDiscoveryTests(unittest.TestCase):
    def test_direct_glb_file_is_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            glb_path = Path(temp_dir) / "asset.glb"
            glb_path.write_bytes(b"placeholder")

            self.assertEqual(view.find_assets(glb_path), [glb_path])

    def test_mixed_directory_lists_every_supported_asset_format(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            urdf_path = root / "asset.urdf"
            glb_path = root / "asset.glb"
            urdf_path.write_text(MINIMAL_ARTICULATED_URDF, encoding="utf-8")
            glb_path.write_bytes(b"placeholder")
            usd_paths = [root / f"asset{suffix}" for suffix in view.USD_ASSET_SUFFIXES]
            for usd_path in usd_paths:
                usd_path.write_bytes(b"placeholder")

            expected = sorted([urdf_path, glb_path, *usd_paths], key=lambda path: str(path).lower())
            self.assertEqual(view.find_assets(root), expected)

    def test_each_usd_suffix_is_accepted_as_a_direct_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            for suffix in view.USD_ASSET_SUFFIXES:
                with self.subTest(suffix=suffix):
                    usd_path = Path(temp_dir) / f"asset{suffix}"
                    usd_path.write_bytes(b"placeholder")

                    self.assertEqual(view.find_assets(usd_path), [usd_path])

    def test_malformed_urdf_is_skipped_during_directory_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            valid_dir = root / "valid" / "urdf_w_collider"
            invalid_dir = root / "invalid" / "urdf_w_collider"
            valid_dir.mkdir(parents=True)
            invalid_dir.mkdir(parents=True)
            valid_path = valid_dir / "valid.urdf"
            invalid_path = invalid_dir / "invalid.urdf"
            valid_path.write_text(MINIMAL_ARTICULATED_URDF, encoding="utf-8")
            invalid_path.write_text(MALFORMED_URDF, encoding="utf-8")

            self.assertEqual(view.find_assets(root), [valid_path])

    def test_direct_malformed_urdf_reports_xml_location(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            urdf_path = Path(temp_dir) / "broken.urdf"
            urdf_path.write_text(MALFORMED_URDF, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r"broken\.urdf.*line 4"):
                view.find_assets(urdf_path)


class AssetTreeTests(unittest.TestCase):
    def test_tree_mirrors_real_directory_containment_and_exact_file_types(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_root = Path(temp_dir) / "data"
            model_root = source_root / "microwave" / "3dc200" / "7e_000"
            collider_root = model_root / "urdf_w_collider"
            assets = [
                model_root / "7e_000.segmented.glb",
                collider_root / "7e_000.urdf",
                collider_root / "glbs" / "1_base.glb",
                collider_root / "asset.usda",
                collider_root / "asset.usdc",
                collider_root / "asset.usdz",
            ]

            roots = view.build_asset_tree(assets, source_root)

            self.assertEqual(len(roots), 1)
            source = roots[0]
            self.assertEqual(source.name, "data")
            self.assertEqual(source.asset_count, len(assets))
            model = (
                source.directories["microwave"]
                .directories["3dc200"]
                .directories["7e_000"]
            )
            self.assertEqual(model.files[0].display_label, "[GLB] 7e_000.segmented.glb")
            collider = model.directories["urdf_w_collider"]
            self.assertEqual(
                sorted(entry.display_label for entry in collider.files),
                [
                    "[URDF] 7e_000.urdf",
                    "[USDA] asset.usda",
                    "[USDC] asset.usdc",
                    "[USDZ] asset.usdz",
                ],
            )
            self.assertEqual(
                collider.directories["glbs"].files[0].display_label,
                "[GLB] 1_base.glb",
            )

    def test_external_assets_share_an_imported_absolute_directory_branch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "data"
            source_asset = source_root / "inside.urdf"
            shared_external = root / "external" / "shared"
            external_assets = [
                shared_external / "first.usd",
                shared_external / "second.glb",
            ]

            roots = view.build_asset_tree(
                [source_asset, *external_assets],
                source_root,
            )

            self.assertEqual([node.name for node in roots], ["data", "Imported Assets"])
            imported = roots[1]
            self.assertEqual(imported.asset_count, 2)

            def descendants(node):
                yield node
                for child in node.directories.values():
                    yield from descendants(child)

            shared = next(node for node in descendants(imported) if node.name == "shared")
            self.assertEqual(shared.asset_count, 2)
            self.assertEqual(
                sorted(entry.display_label for entry in shared.files),
                ["[GLB] second.glb", "[USD] first.usd"],
            )

    def test_asset_tree_renders_directories_before_files_without_numeric_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_root = Path(temp_dir) / "data"
            assets = [
                source_root / "z_file.glb",
                source_root / "a_dir" / "nested.urdf",
                source_root / "a_file.usda",
            ]
            browser = view.AssetBrowser(assets, 1, source_root=source_root)
            calls: list[tuple[str, str]] = []

            class FakeImgui:
                class Cond_:
                    always = "always"
                    appearing = "appearing"

                def set_next_item_open(self, _opened, _condition) -> None:
                    pass

                def tree_node(self, label):
                    calls.append(("directory", label.split("##", 1)[0]))
                    return True

                def is_item_hovered(self):
                    return False

                def set_tooltip(self, _text) -> None:
                    pass

                def selectable(self, label, _selected):
                    calls.append(("file", label.split("##", 1)[0]))
                    return False, False

                def tree_pop(self) -> None:
                    pass

            browser._render_asset_tree(FakeImgui())

            visible = [call for call in calls if call[1] != "Asset Tree"]
            self.assertEqual(
                visible,
                [
                    ("directory", "data (3)"),
                    ("directory", "a_dir (1)"),
                    ("file", "[URDF] nested.urdf"),
                    ("file", "[USDA] a_file.usda"),
                    ("file", "[GLB] z_file.glb"),
                ],
            )
            self.assertFalse(any(label[:1].isdigit() for _kind, label in visible))

    def test_list_output_uses_type_tags_and_relative_paths_without_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_root = Path(temp_dir) / "data"
            assets = [
                source_root / "microwave" / "asset.urdf",
                source_root / "microwave" / "mesh.glb",
            ]
            output = StringIO()

            with redirect_stdout(output):
                view.print_assets(assets, source_root)

            self.assertEqual(
                output.getvalue().splitlines(),
                [
                    f"[URDF] {Path('microwave') / 'asset.urdf'}",
                    f"[GLB] {Path('microwave') / 'mesh.glb'}",
                ],
            )

    def test_viewer_command_line_no_longer_accepts_public_asset_indices(self) -> None:
        self.assertFalse(hasattr(view.parse_args([]), "index"))
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            view.parse_args(["--index", "1"])


class AssetBrowserFileSelectionTests(unittest.TestCase):
    def test_file_dialog_is_made_topmost_before_it_opens(self) -> None:
        calls: list[tuple[object, ...]] = []
        dialog_kwargs: dict[str, object] = {}

        class FakeRoot:
            def withdraw(self) -> None:
                calls.append(("withdraw",))

            def attributes(self, name: str, value: bool) -> None:
                calls.append(("attributes", name, value))

            def update_idletasks(self) -> None:
                calls.append(("update_idletasks",))

            def destroy(self) -> None:
                calls.append(("destroy",))

        def fake_dialog(**kwargs) -> str:
            dialog_kwargs.update(kwargs)
            self.assertIn(("attributes", "-topmost", True), calls)
            self.assertIn(("update_idletasks",), calls)
            return ""

        selected = view.choose_urdf_file(
            Path.cwd(),
            root_factory=FakeRoot,
            askopenfilename=fake_dialog,
        )

        self.assertIsNone(selected)
        self.assertEqual(calls[-1], ("destroy",))
        self.assertIn(("GLB files", "*.glb"), dialog_kwargs["filetypes"])
        self.assertIn(("USD files", "*.usd *.usda *.usdc *.usdz"), dialog_kwargs["filetypes"])

    def test_selected_urdf_is_added_and_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first.urdf"
            selected = Path(temp_dir) / "selected.urdf"
            first.write_text(MINIMAL_ARTICULATED_URDF, encoding="utf-8")
            selected.write_text(MINIMAL_ARTICULATED_URDF, encoding="utf-8")
            browser = view.AssetBrowser([first], 0, file_picker=lambda _initial_dir: selected)

            browser.open_urdf_dialog()

            self.assertEqual(browser.urdfs, [first, selected.resolve()])
            self.assertEqual(browser.consume_request(), 1)
            self.assertIsNone(browser.file_error)

    def test_selected_glb_is_added_and_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first.urdf"
            selected = Path(temp_dir) / "selected.glb"
            first.write_text(MINIMAL_ARTICULATED_URDF, encoding="utf-8")
            selected.write_bytes(b"placeholder")
            browser = view.AssetBrowser([first], 0, file_picker=lambda _initial_dir: selected)

            browser.open_asset_dialog()

            self.assertEqual(browser.urdfs, [first, selected.resolve()])
            self.assertEqual(browser.consume_request(), 1)
            self.assertIsNone(browser.file_error)

    def test_selected_usd_is_added_and_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first.urdf"
            selected = Path(temp_dir) / "selected.usdc"
            first.write_text(MINIMAL_ARTICULATED_URDF, encoding="utf-8")
            selected.write_bytes(b"placeholder")
            browser = view.AssetBrowser([first], 0, file_picker=lambda _initial_dir: selected)

            browser.open_asset_dialog()

            self.assertEqual(browser.urdfs, [first, selected.resolve()])
            self.assertEqual(browser.consume_request(), 1)
            self.assertIsNone(browser.file_error)

    def test_selecting_existing_urdf_does_not_duplicate_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            urdf_path = Path(temp_dir) / "asset.urdf"
            urdf_path.write_text(MINIMAL_ARTICULATED_URDF, encoding="utf-8")
            browser = view.AssetBrowser([urdf_path], 0, file_picker=lambda _initial_dir: urdf_path)

            browser.open_urdf_dialog()

            self.assertEqual(browser.urdfs, [urdf_path])
            self.assertEqual(browser.consume_request(), 0)

    def test_canceling_file_dialog_keeps_current_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            urdf_path = Path(temp_dir) / "asset.urdf"
            urdf_path.write_text(MINIMAL_ARTICULATED_URDF, encoding="utf-8")
            browser = view.AssetBrowser([urdf_path], 0, file_picker=lambda _initial_dir: None)

            browser.open_urdf_dialog()

            self.assertEqual(browser.urdfs, [urdf_path])
            self.assertIsNone(browser.consume_request())

    def test_invalid_selected_file_is_reported_without_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            urdf_path = Path(temp_dir) / "asset.urdf"
            invalid_path = Path(temp_dir) / "notes.txt"
            urdf_path.write_text(MINIMAL_ARTICULATED_URDF, encoding="utf-8")
            invalid_path.write_text("not a URDF", encoding="utf-8")
            browser = view.AssetBrowser([urdf_path], 0, file_picker=lambda _initial_dir: invalid_path)

            browser.open_urdf_dialog()

            self.assertEqual(browser.urdfs, [urdf_path])
            self.assertIsNone(browser.consume_request())
            self.assertIn("Expected a supported asset file", browser.file_error or "")

    def test_malformed_selected_urdf_is_reported_without_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first.urdf"
            malformed = Path(temp_dir) / "malformed.urdf"
            first.write_text(MINIMAL_ARTICULATED_URDF, encoding="utf-8")
            malformed.write_text(MALFORMED_URDF, encoding="utf-8")
            browser = view.AssetBrowser([first], 0, file_picker=lambda _initial_dir: malformed)

            browser.open_urdf_dialog()

            self.assertEqual(browser.urdfs, [first])
            self.assertIsNone(browser.consume_request())
            self.assertIn("Malformed URDF XML", browser.file_error or "")


class AssetRuntimeLoadErrorTests(unittest.TestCase):
    def test_malformed_urdf_does_not_escape_runtime_or_replace_loaded_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            malformed = Path(temp_dir) / "malformed.urdf"
            malformed.write_text(MALFORMED_URDF, encoding="utf-8")
            args = view.parse_args([str(malformed)])
            browser = view.AssetBrowser([malformed], 0)

            class FakeViewer:
                pass

            runtime = view.AssetViewerRuntime(
                args,
                [malformed],
                FakeViewer(),
                {"enabled": True},
                browser,
            )

            self.assertFalse(runtime.load(0))
            self.assertIsNone(runtime.loaded)
            self.assertIn("Malformed URDF XML", browser.file_error or "")


class WarpCpuFallbackTests(unittest.TestCase):
    def test_cpu_only_viewer_uses_pageable_memory_for_pinned_requests(self) -> None:
        class FakeCpuDevice:
            default_allocator = object()
            pinned_allocator = object()

        cpu = FakeCpuDevice()
        with (
            mock.patch("asset_viewer.app.wp.is_cuda_available", return_value=False),
            mock.patch("asset_viewer.app.wp.get_device", return_value=cpu),
        ):
            self.assertTrue(view.configure_warp_cpu_fallback())

        self.assertIs(cpu.pinned_allocator, cpu.default_allocator)


class PhysicsStepRequestTests(unittest.TestCase):
    def test_active_drag_steps_physics_while_viewer_is_paused(self) -> None:
        class Picking:
            def is_picking(self) -> bool:
                return True

        class Viewer:
            picking = Picking()

            def should_step(self) -> bool:
                return False

        self.assertTrue(view.viewer_requests_physics_step(Viewer()))

    def test_paused_viewer_without_drag_does_not_step(self) -> None:
        class Picking:
            def is_picking(self) -> bool:
                return False

        class Viewer:
            picking = Picking()

            def should_step(self) -> bool:
                return False

        self.assertFalse(view.viewer_requests_physics_step(Viewer()))


class SolverSelectionTests(unittest.TestCase):
    def test_mujoco_is_the_default_solver(self) -> None:
        self.assertEqual(view.parse_args([]).solver, "mujoco")

    def test_solver_can_be_selected_from_command_line(self) -> None:
        for solver_name in view.SOLVER_NAMES:
            with self.subTest(solver=solver_name):
                self.assertEqual(view.parse_args(["--solver", solver_name]).solver, solver_name)

    def test_changing_solver_requests_current_asset_reload(self) -> None:
        browser = view.AssetBrowser([Path("asset.urdf")], 0)

        changed = browser.set_solver("featherstone")

        self.assertTrue(changed)
        self.assertEqual(browser.solver_name, "featherstone")
        self.assertEqual(browser.consume_request(), 0)
        self.assertFalse(browser.set_solver("featherstone"))

    def test_all_exposed_solvers_initialize_and_step_an_articulation(self) -> None:
        import newton

        expected_classes = {
            "xpbd": newton.solvers.SolverXPBD,
            "semi-implicit": newton.solvers.SolverSemiImplicit,
            "featherstone": newton.solvers.SolverFeatherstone,
            "vbd": newton.solvers.SolverVBD,
            "mujoco": newton.solvers.SolverMuJoCo,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            urdf_path = Path(temp_dir) / "asset.urdf"
            urdf_path.write_text(MINIMAL_ARTICULATED_URDF, encoding="utf-8")

            for solver_name, solver_class in expected_classes.items():
                with self.subTest(solver=solver_name):
                    args = view.parse_args(
                        [str(urdf_path), "--simulate", "--solver", solver_name]
                    )
                    model, state, _joint_panel = view.build_model(args, urdf_path)
                    solver = view.create_solver(model, solver_name, args.iterations)
                    state_next = model.state()
                    control = model.control()
                    contacts = model.contacts()

                    state.clear_forces()
                    model.collide(state, contacts)
                    solver.step(state, state_next, control, contacts, 1.0 / 240.0)

                    self.assertIsInstance(solver, solver_class)
                    self.assertTrue(np.isfinite(state_next.body_q.numpy()).all())
                    self.assertTrue(np.isfinite(state_next.body_qd.numpy()).all())


class FixedJointCollapseTests(unittest.TestCase):
    def test_fixed_joint_collapsing_defaults_to_disabled(self) -> None:
        self.assertFalse(view.parse_args([]).collapse_fixed_joints)

    def test_fixed_joint_collapsing_can_be_disabled_from_command_line(self) -> None:
        self.assertFalse(view.parse_args(["--no-collapse-fixed-joints"]).collapse_fixed_joints)

    def test_changing_fixed_joint_collapsing_requests_current_asset_reload(self) -> None:
        browser = view.AssetBrowser([Path("asset.urdf")], 0)

        changed = browser.set_collapse_fixed_joints(True)

        self.assertTrue(changed)
        self.assertTrue(browser.collapse_fixed_joints_enabled)
        self.assertEqual(browser.consume_request(), 0)
        self.assertFalse(browser.set_collapse_fixed_joints(True))

    def test_setting_changes_the_imported_model_topology(self) -> None:
        import newton

        with tempfile.TemporaryDirectory() as temp_dir:
            urdf_path = Path(temp_dir) / "asset.urdf"
            urdf_path.write_text(MINIMAL_ARTICULATED_URDF, encoding="utf-8")
            collapsed_args = view.parse_args(
                [str(urdf_path), "--no-floating", "--collapse-fixed-joints"]
            )
            expanded_args = view.parse_args(
                [str(urdf_path), "--no-floating", "--no-collapse-fixed-joints"]
            )

            collapsed_model, _state, _panel = view.build_model(collapsed_args, urdf_path)
            expanded_model, _state, _panel = view.build_model(expanded_args, urdf_path)

            self.assertEqual(expanded_model.body_count, collapsed_model.body_count + 1)
            self.assertEqual(expanded_model.joint_count, collapsed_model.joint_count + 1)
            self.assertIn(int(newton.JointType.FIXED), expanded_model.joint_type.numpy())
            self.assertNotIn(int(newton.JointType.FIXED), collapsed_model.joint_type.numpy())


class FloatingRootTests(unittest.TestCase):
    def test_floating_defaults_to_enabled(self) -> None:
        self.assertTrue(view.parse_args([]).floating)

    def test_floating_can_be_enabled_from_command_line(self) -> None:
        self.assertTrue(view.parse_args(["--floating"]).floating)

    def test_changing_floating_requests_current_asset_reload(self) -> None:
        browser = view.AssetBrowser([Path("asset.urdf")], 0)

        changed = browser.set_floating(False)

        self.assertTrue(changed)
        self.assertFalse(browser.floating_enabled)
        self.assertEqual(browser.consume_request(), 0)
        self.assertFalse(browser.set_floating(False))

    def test_floating_adds_a_free_root_without_joint_sliders(self) -> None:
        import newton

        with tempfile.TemporaryDirectory() as temp_dir:
            urdf_path = Path(temp_dir) / "asset.urdf"
            urdf_path.write_text(MINIMAL_ARTICULATED_URDF, encoding="utf-8")
            fixed_args = view.parse_args(
                [str(urdf_path), "--no-floating", "--no-collapse-fixed-joints"]
            )
            floating_args = view.parse_args(
                [str(urdf_path), "--floating", "--no-collapse-fixed-joints"]
            )

            fixed_model, _state, fixed_panel = view.build_model(fixed_args, urdf_path)
            floating_model, _state, floating_panel = view.build_model(floating_args, urdf_path)

            self.assertEqual(floating_model.body_count, fixed_model.body_count)
            self.assertEqual(floating_model.joint_count, fixed_model.joint_count)
            self.assertEqual(floating_model.joint_coord_count, fixed_model.joint_coord_count + 7)
            self.assertIn(int(newton.JointType.FREE), floating_model.joint_type.numpy())
            self.assertEqual(len(floating_panel.controls), len(fixed_panel.controls))


class UsdRootModeTests(unittest.TestCase):
    def test_usd_root_mode_defaults_to_authored(self) -> None:
        self.assertEqual(view.parse_args([]).usd_root_mode, "authored")

    def test_all_usd_root_modes_are_accepted(self) -> None:
        for mode in view.USD_ROOT_MODES:
            with self.subTest(mode=mode):
                self.assertEqual(
                    view.parse_args(["--usd-root-mode", mode]).usd_root_mode,
                    mode,
                )

    def test_changing_usd_root_mode_requests_current_asset_reload(self) -> None:
        browser = view.AssetBrowser([Path("asset.usda")], 0)

        changed = browser.set_usd_root_mode("fixed")

        self.assertTrue(changed)
        self.assertEqual(browser.usd_root_mode, "fixed")
        self.assertEqual(browser.consume_request(), 0)
        self.assertFalse(browser.set_usd_root_mode("fixed"))


class UsdIntegrationTests(unittest.TestCase):
    def test_minimal_physics_usd_builds_a_dynamic_rigid_body(self) -> None:
        import newton

        usd_path = TEST_FIXTURES / "minimal_rigid.usda"
        args = view.parse_args(
            [str(usd_path), "--no-ground", "--z", "0", "--usd-root-mode", "authored"]
        )

        model, _state, joint_panel = view.build_model(args, usd_path)

        self.assertEqual(model.body_count, 1)
        self.assertEqual(model.shape_count, 1)
        self.assertEqual(model.joint_count, 1)
        self.assertEqual(int(model.joint_type.numpy()[0]), int(newton.JointType.FREE))
        self.assertEqual(joint_panel.controls, [])


class CameraControlTests(unittest.TestCase):
    def test_current_parameter_defaults_are_documented_behavior(self) -> None:
        args = view.parse_args([])

        self.assertEqual(args.z, 5.0)
        self.assertEqual(args.camera_z, 5.0)
        self.assertEqual(args.solver, "mujoco")
        self.assertTrue(args.floating)
        self.assertFalse(args.collapse_fixed_joints)
        self.assertTrue(args.auto_frame)

    def test_asset_bounds_ignore_ground_and_preserve_small_mesh_size(self) -> None:
        import trimesh

        with tempfile.TemporaryDirectory() as temp_dir:
            glb_path = Path(temp_dir) / "tiny.glb"
            trimesh.creation.box(extents=(0.02, 0.04, 0.06)).export(glb_path)
            args = view.parse_args([str(glb_path)])

            model, state, _panel = view.build_model(args, glb_path)
            bounds = view.compute_asset_bounds(model, state)

            np.testing.assert_allclose(bounds.center, (0.0, 0.0, 5.0), atol=1.0e-5)
            np.testing.assert_allclose(bounds.extents, (0.02, 0.04, 0.06), atol=1.0e-5)

    def test_framing_points_camera_at_asset_and_scales_distance(self) -> None:
        from newton._src.viewer.camera import Camera

        camera = Camera(pos=(0.65, -0.75, 5.0), up_axis="Z")
        small = view.AssetBounds(
            minimum=np.array((-0.01, -0.02, 4.97)),
            maximum=np.array((0.01, 0.02, 5.03)),
        )

        view.frame_camera_on_bounds(camera, small, padding=1.35)

        to_asset = small.center - np.asarray(camera.pos)
        to_asset /= np.linalg.norm(to_asset)
        np.testing.assert_allclose(np.asarray(camera.get_front()), to_asset, atol=1.0e-6)
        np.testing.assert_allclose(np.asarray(camera.pivot), small.center, atol=1.0e-6)
        self.assertLess(camera.pivot_distance, 0.25)

    def test_small_assets_get_slower_adaptive_camera_motion(self) -> None:
        tiny = view.AssetBounds(np.zeros(3), np.array((0.02, 0.04, 0.06)))
        large = view.AssetBounds(np.zeros(3), np.array((2.0, 4.0, 6.0)))

        self.assertLess(view.recommended_camera_speed(tiny), view.recommended_camera_speed(large))
        self.assertGreater(view.recommended_camera_speed(tiny), 0.0)


class SelfCollisionControlTests(unittest.TestCase):
    def test_self_collisions_default_to_disabled(self) -> None:
        args = view.parse_args([])
        self.assertFalse(args.self_collisions)

    def test_self_collisions_can_be_enabled_from_command_line(self) -> None:
        args = view.parse_args(["--self-collisions"])
        self.assertTrue(args.self_collisions)

    def test_changing_self_collisions_requests_current_asset_reload(self) -> None:
        browser = view.AssetBrowser([Path("asset.urdf")], 0)

        changed = browser.set_self_collisions(True)

        self.assertTrue(changed)
        self.assertTrue(browser.self_collisions_enabled)
        self.assertEqual(browser.consume_request(), 0)

    def test_unchanged_self_collisions_does_not_request_reload(self) -> None:
        browser = view.AssetBrowser([Path("asset.urdf")], 0)

        changed = browser.set_self_collisions(False)

        self.assertFalse(changed)
        self.assertIsNone(browser.consume_request())


if __name__ == "__main__":
    unittest.main()
