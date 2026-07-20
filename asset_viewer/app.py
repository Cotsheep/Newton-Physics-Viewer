"""Newton model loading, viewer runtime, and command-line interface."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import newton
import newton.viewer
import warp as wp

from .assets import (
    DEFAULT_SOURCE,
    copy_text_to_clipboard,
    copyable_model_id_from_urdf,
    describe_metadata,
    describe_urdf_path,
    find_urdfs,
    load_articulation_metadata,
    print_urdfs,
)
from .controls import (
    DEFAULT_CONTINUOUS_SPEED_DEGREES,
    JointControlPanel,
    assist_imgui_window_wheel_scroll,
    widen_imgui_scrollbar,
)
from .traction import TractionForceMonitor


def build_model(args: argparse.Namespace, urdf_path: Path) -> tuple[newton.Model, newton.State, JointControlPanel]:
    builder = newton.ModelBuilder(
        up_axis=newton.Axis.Z,
        gravity=-9.81 if args.simulate else 0.0,
    )

    builder.add_urdf(
        str(urdf_path),
        xform=wp.transform((0.0, 0.0, args.z), wp.quat_identity()),
        floating=False,
        scale=args.scale,
        up_axis=newton.Axis.Z,
        enable_self_collisions=False,
        collapse_fixed_joints=False,
        force_show_colliders=args.show_colliders,
    )

    if args.ground:
        builder.add_ground_plane()

    model = builder.finalize()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    articulation_metadata = load_articulation_metadata(urdf_path)
    joint_panel = JointControlPanel(
        model,
        state,
        articulation_metadata,
        args.joint_range_degrees,
        args.continuous_speed_degrees,
    )
    if args.initial_motion_state:
        joint_panel.apply_motion_state(args.initial_motion_state)
    return model, state, joint_panel


@dataclass
class LoadedAsset:
    index: int
    urdf_path: Path
    model: newton.Model
    state: newton.State
    joint_panel: JointControlPanel
    solver: Any | None
    state_next: newton.State | None
    control: Any | None
    contacts: Any | None
    traction_monitor: TractionForceMonitor | None


class AssetBrowser:
    def __init__(self, urdfs: list[Path], current_index: int) -> None:
        self.urdfs = urdfs
        self.current_index = current_index
        self.requested_index: int | None = None
        self.last_copied_model_id: str | None = None
        self.copy_error_model_id: str | None = None
        self.last_panel_scroll_y: float | None = None
        self.tree = self._build_tree(urdfs)

    def _build_tree(self, urdfs: list[Path]) -> dict[str, dict[str, list[tuple[int, str]]]]:
        tree: dict[str, dict[str, list[tuple[int, str]]]] = {}
        for index, urdf_path in enumerate(urdfs):
            parts = describe_urdf_path(urdf_path).split("/")
            category = parts[0] if parts else "assets"
            provider = parts[1] if len(parts) > 2 else "models"
            model_name = parts[-1] if parts else urdf_path.stem
            tree.setdefault(category, {}).setdefault(provider, []).append((index, model_name))
        return {
            category: {
                provider: sorted(entries, key=lambda item: (item[1].lower(), item[0]))
                for provider, entries in sorted(providers.items())
            }
            for category, providers in sorted(tree.items())
        }

    def render_ui(self, imgui) -> None:
        widen_imgui_scrollbar(imgui)
        self.last_panel_scroll_y = assist_imgui_window_wheel_scroll(imgui, self.last_panel_scroll_y)
        imgui.set_next_item_open(True, imgui.Cond_.appearing)
        if not imgui.collapsing_header("Assets"):
            return

        imgui.text(f"Current: {self.current_index:03d} / {len(self.urdfs) - 1:03d}")
        imgui.text(describe_urdf_path(self.urdfs[self.current_index]))
        model_id = copyable_model_id_from_urdf(self.urdfs[self.current_index])
        imgui.text(f"Model ID: {model_id}")

        if imgui.button("Copy Model ID##asset_copy_model_id"):
            if copy_text_to_clipboard(model_id, imgui):
                self.last_copied_model_id = model_id
                self.copy_error_model_id = None
            else:
                self.last_copied_model_id = None
                self.copy_error_model_id = model_id

        if self.last_copied_model_id == model_id:
            imgui.same_line()
            imgui.text("Copied")
        elif self.copy_error_model_id == model_id:
            imgui.same_line()
            imgui.text("Copy failed")

        if imgui.button("Reload##asset_reload"):
            self.requested_index = self.current_index
        imgui.same_line()
        if imgui.button("Previous##asset_previous"):
            self.requested_index = (self.current_index - 1) % len(self.urdfs)
        imgui.same_line()
        if imgui.button("Next##asset_next"):
            self.requested_index = (self.current_index + 1) % len(self.urdfs)

        if imgui.tree_node("Asset List"):
            for category, providers in self.tree.items():
                if imgui.tree_node(category):
                    for provider, entries in providers.items():
                        if imgui.tree_node(provider):
                            for index, model_name in entries:
                                selected = index == self.current_index
                                clicked, _selected = imgui.selectable(
                                    f"{index:03d} {model_name}##asset_{index}",
                                    selected,
                                )
                                if clicked:
                                    self.requested_index = index
                            imgui.tree_pop()
                    imgui.tree_pop()
            imgui.tree_pop()

    def consume_request(self) -> int | None:
        requested_index = self.requested_index
        self.requested_index = None
        return requested_index


class AssetViewerRuntime:
    def __init__(
        self,
        args: argparse.Namespace,
        urdfs: list[Path],
        viewer: newton.viewer.ViewerGL,
        double_sided_state: dict[str, bool],
        asset_browser: AssetBrowser,
    ) -> None:
        self.args = args
        self.urdfs = urdfs
        self.viewer = viewer
        self.double_sided_state = double_sided_state
        self.asset_browser = asset_browser
        self.loaded: LoadedAsset | None = None
        self.sim_time = 0.0

    def load(self, index: int, show_splash: bool = False) -> None:
        if index < 0 or index >= len(self.urdfs):
            raise IndexError(f"Asset index must be between 0 and {len(self.urdfs) - 1}; got {index}")

        urdf_path = self.urdfs[index]
        label = describe_urdf_path(urdf_path)
        if show_splash:
            self._show_splash(f"Loading {label}...")

        try:
            model, state, joint_panel = build_model(self.args, urdf_path)
            solver = None
            state_next = None
            control = None
            contacts = None
            if self.args.simulate:
                solver = newton.solvers.SolverXPBD(model, iterations=self.args.iterations)
                state_next = model.state()
                control = model.control()
                contacts = model.contacts()

            if self.loaded is not None:
                self.viewer.clear_model()

            self.viewer.set_model(model)
            self.viewer.set_camera(
                wp.vec3(self.args.camera_x, self.args.camera_y, self.args.camera_z),
                self.args.pitch,
                self.args.yaw,
            )
            set_double_sided_rendering(self.viewer, self.double_sided_state["enabled"])
            self._set_reset_callback(joint_panel.reset)

            traction_monitor = (
                TractionForceMonitor(model, self.args.traction_print_hz, self.args.print_traction_force)
                if self.args.simulate
                else None
            )
            self._register_asset_ui(joint_panel, traction_monitor)

            self.loaded = LoadedAsset(
                index=index,
                urdf_path=urdf_path,
                model=model,
                state=state,
                joint_panel=joint_panel,
                solver=solver,
                state_next=state_next,
                control=control,
                contacts=contacts,
                traction_monitor=traction_monitor,
            )
            self.asset_browser.current_index = index
            self.sim_time = 0.0
            self._print_loaded_summary(label, urdf_path, model, joint_panel)
        finally:
            if show_splash:
                self._hide_splash()

    def _register_asset_ui(
        self,
        joint_panel: JointControlPanel,
        traction_monitor: TractionForceMonitor | None,
    ) -> None:
        if self.args.joint_ui and hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(joint_panel.render_ui, position="side")
        if traction_monitor is not None and hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(traction_monitor.render_ui, position="side")

    def _set_reset_callback(self, callback) -> None:
        if hasattr(self.viewer, "set_reset_callback"):
            self.viewer.set_reset_callback(callback)
        else:
            self.viewer._reset_callback = callback

    def _show_splash(self, text: str) -> None:
        if not hasattr(self.viewer, "show_loading_splash"):
            return
        self.viewer.show_loading_splash(text)
        for _ in range(2):
            self.viewer.begin_frame(self.sim_time)
            self.viewer.end_frame()

    def _hide_splash(self) -> None:
        if hasattr(self.viewer, "hide_loading_splash"):
            self.viewer.hide_loading_splash()

    def _print_loaded_summary(
        self,
        label: str,
        urdf_path: Path,
        model: newton.Model,
        joint_panel: JointControlPanel,
    ) -> None:
        print(f"Opening {label}: {urdf_path}")
        print(
            f"Loaded {model.body_count} bodies, {model.joint_count} joints, "
            f"{model.shape_count} shapes, {model.joint_coord_count} joint coords."
        )
        print(f"Articulation metadata: {describe_metadata(joint_panel.articulation_metadata)}")
        print(f"Interactive controls: {len(joint_panel.controls)} joint controls in the left panel.")


def set_double_sided_rendering(viewer: newton.viewer.ViewerGL, enabled: bool) -> None:
    """Toggle back-face culling for meshes already registered in ViewerGL."""
    objects = getattr(viewer, "objects", {})
    for obj in objects.values():
        if hasattr(obj, "backface_culling"):
            obj.backface_culling = not enabled


def run_viewer(
    args: argparse.Namespace,
    urdfs: list[Path],
    initial_index: int,
) -> None:
    viewer = newton.viewer.ViewerGL(
        width=args.width,
        height=args.height,
        headless=args.headless,
        paused=not (args.simulate and args.start_running),
    )

    double_sided_state = {"enabled": args.double_sided}
    asset_browser = AssetBrowser(urdfs, initial_index)
    runtime = AssetViewerRuntime(args, urdfs, viewer, double_sided_state, asset_browser)

    if hasattr(viewer, "register_ui_callback"):
        viewer.register_ui_callback(asset_browser.render_ui, position="panel")

    def render_double_sided_option(imgui) -> None:
        widen_imgui_scrollbar(imgui)
        changed, enabled = imgui.checkbox("Double-sided Meshes", double_sided_state["enabled"])
        if changed:
            double_sided_state["enabled"] = enabled
        set_double_sided_rendering(viewer, double_sided_state["enabled"])

    if hasattr(viewer, "register_ui_callback"):
        viewer.register_ui_callback(render_double_sided_option, position="rendering")

    runtime.load(initial_index)

    frame_dt = 1.0 / args.fps
    sim_dt = frame_dt / args.substeps
    frame = 0

    if not args.simulate and args.print_traction_force:
        print("Traction force printing is idle because --simulate is off.")

    while viewer.is_running():
        requested_index = asset_browser.consume_request()
        if requested_index is not None:
            runtime.load(requested_index, show_splash=True)

        loaded = runtime.loaded
        if loaded is None:
            break

        if args.simulate and viewer.should_step():
            assert loaded.solver is not None
            assert loaded.state_next is not None
            assert loaded.control is not None
            assert loaded.contacts is not None

            for substep in range(args.substeps):
                substep_time = runtime.sim_time + substep * sim_dt
                loaded.state.clear_forces()
                viewer.apply_forces(loaded.state)
                if loaded.traction_monitor is not None:
                    loaded.traction_monitor.update(viewer, loaded.state, substep_time)
                loaded.model.collide(loaded.state, loaded.contacts)
                loaded.solver.step(loaded.state, loaded.state_next, loaded.control, loaded.contacts, sim_dt)
                loaded.state, loaded.state_next = loaded.state_next, loaded.state

            runtime.sim_time += frame_dt

        loaded.joint_panel.set_state(loaded.state)
        loaded.joint_panel.update_continuous_motion(frame_dt)

        viewer.begin_frame(runtime.sim_time)
        viewer.log_state(loaded.state)
        if loaded.contacts is not None:
            viewer.log_contacts(loaded.contacts, loaded.state)
        viewer.end_frame()

        frame += 1
        if args.frames is not None and frame >= args.frames:
            break

    viewer.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View Artiverse URDF assets with Newton.")
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Artiverse data root, one category directory, one model directory, or one .urdf file.",
    )
    parser.add_argument("--list", action="store_true", help="List matching URDF files and exit.")
    parser.add_argument(
        "--category",
        action="append",
        default=None,
        help="Limit discovery to a category folder, e.g. microwave or scissors. Repeat for multiple categories.",
    )
    parser.add_argument("--index", type=int, default=0, help="URDF index to open when source contains many models.")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--z", type=float, default=0.0, help="Vertical offset for the imported model.")
    parser.add_argument("--ground", action="store_true", help="Add a ground plane.")
    parser.add_argument("--show-colliders", action="store_true", help="Show collision meshes even when visual meshes exist.")
    parser.add_argument(
        "--double-sided",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render meshes from both sides by disabling back-face culling.",
    )
    parser.add_argument("--joint-ui", action=argparse.BooleanOptionalAction, default=True, help="Show joint sliders.")
    parser.add_argument(
        "--joint-range-degrees",
        type=float,
        default=180.0,
        help="Slider half-range for continuous/unbounded revolute joints.",
    )
    parser.add_argument(
        "--continuous-speed-degrees",
        type=float,
        default=DEFAULT_CONTINUOUS_SPEED_DEGREES,
        help="Angular speed in degrees per second for continuous joint start/stop controls.",
    )
    parser.add_argument(
        "--initial-motion-state",
        type=str,
        default=None,
        help="Apply a state from *.articulations.json on startup, e.g. closed.",
    )
    parser.add_argument("--simulate", action="store_true", help="Run Newton dynamics instead of static viewing.")
    parser.add_argument(
        "--start-running",
        action="store_true",
        help="Start stepping immediately in simulation mode. By default simulation loads paused at the asset's original pose.",
    )
    parser.add_argument(
        "--print-traction-force",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print the right-click picking traction force to the terminal while dragging in simulation mode.",
    )
    parser.add_argument(
        "--traction-print-hz",
        type=float,
        default=10.0,
        help="Traction-force print rate in Hz. Use 0 to print every physics substep.",
    )
    parser.add_argument("--frames", type=int, default=None, help="Close after N rendered frames.")
    parser.add_argument("--headless", action="store_true", help="Run ViewerGL without opening a visible window.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--camera-x", type=float, default=0.65)
    parser.add_argument("--camera-y", type=float, default=-0.75)
    parser.add_argument("--camera-z", type=float, default=0.45)
    parser.add_argument("--pitch", type=float, default=-25.0)
    parser.add_argument("--yaw", type=float, default=40.0)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    categories = tuple(args.category) if args.category else None
    urdfs = find_urdfs(args.source, categories)

    if args.list:
        print_urdfs(urdfs)
        return

    if args.index < 0 or args.index >= len(urdfs):
        raise IndexError(f"--index must be between 0 and {len(urdfs) - 1}; got {args.index}")

    run_viewer(args, urdfs, args.index)


if __name__ == "__main__":
    main()
