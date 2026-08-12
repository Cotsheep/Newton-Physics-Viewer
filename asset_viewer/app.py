"""Newton model loading, viewer runtime, and command-line interface."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import newton
import newton.viewer
import numpy as np
import warp as wp
from newton._src.utils.mesh import load_meshes_from_file

from .assets import (
    DEFAULT_SOURCE,
    SUPPORTED_ASSET_SUFFIXES,
    USD_ASSET_SUFFIXES,
    ArticulationMetadata,
    AssetTreeNode,
    asset_listing_path,
    asset_type_label,
    build_asset_tree,
    copy_text_to_clipboard,
    copyable_model_id_from_asset,
    describe_asset_path,
    describe_metadata,
    find_assets,
    load_articulation_metadata,
    print_assets,
    validate_urdf_xml,
)
from .controls import (
    DEFAULT_CONTINUOUS_SPEED_DEGREES,
    JointControlPanel,
    assist_imgui_window_wheel_scroll,
    widen_imgui_scrollbar,
)
from .camera import CameraControlPanel, compute_asset_bounds
from .solvers import SOLVER_LABELS, SOLVER_NAMES, create_solver, prepare_builder_for_solver
from .traction import TractionForceMonitor


USD_ROOT_MODES = ("authored", "floating", "fixed")
USD_ROOT_MODE_LABELS = {
    "authored": "Authored",
    "floating": "Floating",
    "fixed": "Fixed",
}


def is_usd_asset(asset_path: Path) -> bool:
    return asset_path.suffix.lower() in USD_ASSET_SUFFIXES


def usd_root_floating(root_mode: str) -> bool | None:
    if root_mode == "authored":
        return None
    if root_mode == "floating":
        return True
    if root_mode == "fixed":
        return False
    choices = ", ".join(USD_ROOT_MODES)
    raise ValueError(f"Unknown USD root mode {root_mode!r}; expected one of: {choices}")


def configure_warp_cpu_fallback() -> bool:
    """Avoid CUDA-backed pinned allocations when Warp has no usable CUDA device.

    Newton's OpenGL viewer requests pinned host buffers even on its CPU render
    path. Warp implements those buffers through the CUDA driver, so the request
    fails on systems where the CUDA toolkit is present but the driver is not.
    Ordinary pageable host memory is sufficient for the viewer's CPU path.
    """
    if wp.is_cuda_available():
        return False

    cpu_device = wp.get_device("cpu")
    cpu_device.pinned_allocator = cpu_device.default_allocator
    return True


def choose_asset_file(
    initial_dir: Path,
    *,
    root_factory: Callable[[], Any] | None = None,
    askopenfilename: Callable[..., str] | None = None,
) -> Path | None:
    """Open a foreground file dialog and return the selected asset path."""
    if root_factory is None or askopenfilename is None:
        from tkinter import Tk, filedialog

        root_factory = root_factory or Tk
        askopenfilename = askopenfilename or filedialog.askopenfilename

    root = root_factory()
    try:
        root.withdraw()
        # ViewerGL owns a separate native window and can otherwise cover this
        # modal dialog while askopenfilename() continues blocking the UI callback.
        root.attributes("-topmost", True)
        root.update_idletasks()
        selected = askopenfilename(
            parent=root,
            title="Import URDF, USD, or GLB asset",
            initialdir=str(initial_dir),
            filetypes=(
                ("Supported assets", "*.urdf *.usd *.usda *.usdc *.usdz *.glb"),
                ("URDF files", "*.urdf"),
                ("USD files", "*.usd *.usda *.usdc *.usdz"),
                ("GLB files", "*.glb"),
                ("All files", "*.*"),
            ),
        )
    finally:
        root.destroy()

    return Path(selected) if selected else None


def choose_urdf_file(
    initial_dir: Path,
    *,
    root_factory: Callable[[], Any] | None = None,
    askopenfilename: Callable[..., str] | None = None,
) -> Path | None:
    """Backward-compatible alias for the supported-asset dialog."""
    return choose_asset_file(
        initial_dir,
        root_factory=root_factory,
        askopenfilename=askopenfilename,
    )


def _box_inertia(mass: float, extents: np.ndarray) -> wp.mat33:
    dx, dy, dz = np.maximum(np.asarray(extents, dtype=np.float64), 1.0e-6)
    ixx = mass * (dy * dy + dz * dz) / 12.0
    iyy = mass * (dx * dx + dz * dz) / 12.0
    izz = mass * (dx * dx + dy * dy) / 12.0
    return wp.mat33(ixx, 0.0, 0.0, 0.0, iyy, 0.0, 0.0, 0.0, izz)


def _add_glb(builder: newton.ModelBuilder, args: argparse.Namespace, glb_path: Path) -> None:
    # Newton's URDF importer uses the same loader. Keeping it here preserves
    # GLB materials, textures, UVs, and multiple geometry/material primitives.
    meshes = load_meshes_from_file(
        str(glb_path),
        scale=(args.scale, args.scale, args.scale),
        maxhullvert=args.glb_max_hull_vertices,
    )
    if not meshes:
        raise ValueError(f"GLB contains no triangle meshes: {glb_path}")

    vertices = [np.asarray(mesh.vertices, dtype=np.float64) for mesh in meshes if len(mesh.vertices)]
    if not vertices:
        raise ValueError(f"GLB contains no mesh vertices: {glb_path}")

    all_vertices = np.concatenate(vertices, axis=0)
    if all_vertices.ndim != 2 or all_vertices.shape[1] != 3 or not np.isfinite(all_vertices).all():
        raise ValueError(f"GLB contains invalid mesh vertices: {glb_path}")

    bounds_min = all_vertices.min(axis=0)
    bounds_max = all_vertices.max(axis=0)
    center = (bounds_min + bounds_max) * 0.5
    extents = bounds_max - bounds_min
    body = builder.add_body(
        xform=wp.transform((0.0, 0.0, args.z), wp.quat_identity()),
        com=wp.vec3(*center),
        inertia=_box_inertia(args.glb_mass, extents),
        mass=args.glb_mass,
        label=glb_path.stem,
        lock_inertia=True,
    )

    exact_mesh_cfg = newton.ModelBuilder.ShapeConfig(density=0.0)
    visual_only_cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0,
        has_shape_collision=False,
        has_particle_collision=False,
        is_visible=True,
    )
    convex_collision_cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0,
        is_visible=args.show_colliders,
    )
    for mesh_index, mesh in enumerate(meshes):
        label = f"{glb_path.stem}/mesh_{mesh_index}"
        if args.glb_collision == "mesh":
            builder.add_shape_mesh(body, mesh=mesh, cfg=exact_mesh_cfg, label=label)
        else:
            builder.add_shape_mesh(body, mesh=mesh, cfg=visual_only_cfg, label=f"{label}/visual")
            builder.add_shape_convex_hull(
                body,
                mesh=mesh,
                cfg=convex_collision_cfg,
                label=f"{label}/collision",
            )


def build_model(args: argparse.Namespace, asset_path: Path) -> tuple[newton.Model, newton.State, JointControlPanel]:
    is_glb = asset_path.suffix.lower() == ".glb"
    is_usd = is_usd_asset(asset_path)
    builder = newton.ModelBuilder(
        up_axis=newton.Axis.Z,
        gravity=-9.81 if args.simulate or is_glb else 0.0,
    )

    if is_glb:
        _add_glb(builder, args, asset_path)
    elif asset_path.suffix.lower() == ".urdf":
        builder.add_urdf(
            str(asset_path),
            xform=wp.transform((0.0, 0.0, args.z), wp.quat_identity()),
            floating=args.floating,
            scale=args.scale,
            up_axis=newton.Axis.Z,
            enable_self_collisions=args.self_collisions,
            collapse_fixed_joints=args.collapse_fixed_joints,
            force_show_colliders=args.show_colliders,
        )
    elif is_usd:
        if args.scale != 1.0:
            raise ValueError(
                "--scale does not apply to USD assets; author metersPerUnit in the USD stage instead"
            )
        try:
            builder.add_usd(
                str(asset_path),
                xform=wp.transform((0.0, 0.0, args.z), wp.quat_identity()),
                floating=usd_root_floating(args.usd_root_mode),
                enable_self_collisions=args.self_collisions,
                collapse_fixed_joints=args.collapse_fixed_joints,
                force_show_colliders=args.show_colliders,
            )
        except ImportError as exc:
            raise RuntimeError(
                "USD import requires OpenUSD Python bindings (the 'pxr' module)"
            ) from exc
    else:
        suffixes = ", ".join(sorted(SUPPORTED_ASSET_SUFFIXES))
        raise ValueError(f"Expected a supported asset file ({suffixes}), got: {asset_path}")

    if args.ground:
        builder.add_ground_plane()

    if (args.simulate or is_glb) and args.solver == "vbd":
        prepare_builder_for_solver(builder, args.solver)

    model = builder.finalize()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    articulation_metadata = (
        ArticulationMetadata(path=None, model_id=None, records_by_pid={}, explicit_dependencies=[])
        if asset_path.suffix.lower() != ".urdf"
        else load_articulation_metadata(asset_path)
    )
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
    asset_path: Path
    model: newton.Model
    state: newton.State
    joint_panel: JointControlPanel
    solver: Any | None
    state_next: newton.State | None
    control: Any | None
    contacts: Any | None
    traction_monitor: TractionForceMonitor | None

    @property
    def urdf_path(self) -> Path:
        """Compatibility name retained for integrations written before GLB support."""
        return self.asset_path


class AssetBrowser:
    def __init__(
        self,
        urdfs: list[Path],
        current_index: int,
        file_picker: Callable[[Path], Path | None] | None = None,
        self_collisions_enabled: bool = False,
        collapse_fixed_joints_enabled: bool = False,
        floating_enabled: bool = True,
        usd_root_mode: str = "authored",
        solver_name: str = "mujoco",
        source_root: Path | None = None,
    ) -> None:
        if not urdfs:
            raise ValueError("AssetBrowser requires at least one asset")
        if current_index < 0 or current_index >= len(urdfs):
            raise IndexError(
                f"Current asset position must be between 0 and {len(urdfs) - 1}; got {current_index}"
            )
        self.urdfs = urdfs
        self.current_index = current_index
        self.source_root = (
            source_root.resolve(strict=False)
            if source_root is not None
            else urdfs[0].resolve(strict=False).parent
        )
        self.file_picker = file_picker or choose_asset_file
        self.self_collisions_enabled = self_collisions_enabled
        self.collapse_fixed_joints_enabled = collapse_fixed_joints_enabled
        self.floating_enabled = floating_enabled
        if usd_root_mode not in USD_ROOT_MODES:
            choices = ", ".join(USD_ROOT_MODES)
            raise ValueError(f"Unknown USD root mode {usd_root_mode!r}; expected one of: {choices}")
        self.usd_root_mode = usd_root_mode
        self.solver_name = solver_name
        self.requested_index: int | None = None
        self.file_error: str | None = None
        self.load_warning: str | None = None
        self.last_copied_model_id: str | None = None
        self.copy_error_model_id: str | None = None
        self.last_panel_scroll_y: float | None = None
        self.tree_roots = build_asset_tree(urdfs, self.source_root)
        self.reveal_current_asset = True

    def request_asset(self, asset_path: Path) -> int:
        """Add a selected supported asset to the browser, if needed, and request it."""
        candidate = Path(asset_path).expanduser()
        if candidate.suffix.lower() not in SUPPORTED_ASSET_SUFFIXES:
            suffixes = ", ".join(sorted(SUPPORTED_ASSET_SUFFIXES))
            raise ValueError(f"Expected a supported asset file ({suffixes}), got: {candidate}")

        candidate = candidate.resolve(strict=True)
        if not candidate.is_file():
            raise ValueError(f"Expected a file, got: {candidate}")
        if candidate.suffix.lower() == ".urdf":
            validate_urdf_xml(candidate)

        index = next(
            (
                existing_index
                for existing_index, existing_path in enumerate(self.urdfs)
                if existing_path.resolve() == candidate
            ),
            None,
        )
        if index is None:
            self.urdfs.append(candidate)
            index = len(self.urdfs) - 1
            self.tree_roots = build_asset_tree(self.urdfs, self.source_root)

        self.requested_index = index
        self.file_error = None
        return index

    def request_urdf(self, urdf_path: Path) -> int:
        """Backward-compatible name for request_asset()."""
        return self.request_asset(urdf_path)

    def open_asset_dialog(self) -> None:
        """Ask the user for a supported asset path and queue it for loading."""
        self.file_error = None
        try:
            selected = self.file_picker(self.urdfs[self.current_index].parent)
            if selected is not None:
                self.request_asset(selected)
        except Exception as exc:
            self.file_error = str(exc) or type(exc).__name__

    def open_urdf_dialog(self) -> None:
        """Backward-compatible name for open_asset_dialog()."""
        self.open_asset_dialog()

    def set_self_collisions(self, enabled: bool) -> bool:
        """Update self-collision handling and queue the current asset for rebuilding."""
        if enabled == self.self_collisions_enabled:
            return False

        self.self_collisions_enabled = enabled
        self.requested_index = self.current_index
        return True

    def set_solver(self, solver_name: str) -> bool:
        """Select a physics solver and queue the current asset for rebuilding."""
        if solver_name not in SOLVER_NAMES:
            choices = ", ".join(SOLVER_NAMES)
            raise ValueError(f"Unknown solver {solver_name!r}; expected one of: {choices}")
        if solver_name == self.solver_name:
            return False

        self.solver_name = solver_name
        self.requested_index = self.current_index
        return True

    def set_collapse_fixed_joints(self, enabled: bool) -> bool:
        """Update fixed-joint collapsing and queue the current URDF for rebuilding."""
        if enabled == self.collapse_fixed_joints_enabled:
            return False

        self.collapse_fixed_joints_enabled = enabled
        self.requested_index = self.current_index
        return True

    def set_floating(self, enabled: bool) -> bool:
        """Update URDF root mobility and queue the current asset for rebuilding."""
        if enabled == self.floating_enabled:
            return False

        self.floating_enabled = enabled
        self.requested_index = self.current_index
        return True

    def set_usd_root_mode(self, root_mode: str) -> bool:
        """Update USD root mobility and queue the current asset for rebuilding."""
        if root_mode not in USD_ROOT_MODES:
            choices = ", ".join(USD_ROOT_MODES)
            raise ValueError(f"Unknown USD root mode {root_mode!r}; expected one of: {choices}")
        if root_mode == self.usd_root_mode:
            return False

        self.usd_root_mode = root_mode
        self.requested_index = self.current_index
        return True

    def set_current_index(self, index: int) -> None:
        """Record the loaded asset and reveal its directory branch once."""
        if index < 0 or index >= len(self.urdfs):
            raise IndexError(
                f"Current asset position must be between 0 and {len(self.urdfs) - 1}; got {index}"
            )
        changed = index != self.current_index
        self.current_index = index
        if changed:
            self.reveal_current_asset = True

    def _render_asset_directory(self, imgui, node: AssetTreeNode) -> None:
        contains_current = self.current_index in node.asset_indices
        if self.reveal_current_asset and contains_current:
            imgui.set_next_item_open(True, imgui.Cond_.always)

        opened = imgui.tree_node(
            f"{node.name} ({node.asset_count})##asset_directory_{node.key}"
        )
        if node.path is not None and imgui.is_item_hovered():
            imgui.set_tooltip(str(node.path))
        if not opened:
            return

        for child in sorted(node.directories.values(), key=lambda item: item.name.casefold()):
            self._render_asset_directory(imgui, child)

        for entry in sorted(
            node.files,
            key=lambda item: (item.path.name.casefold(), str(item.path).casefold()),
        ):
            selected = entry.asset_index == self.current_index
            file_key = str(entry.path.resolve(strict=False)).casefold()
            clicked, _selected = imgui.selectable(
                f"{entry.display_label}##asset_file_{file_key}",
                selected,
            )
            if clicked:
                self.requested_index = entry.asset_index

        imgui.tree_pop()

    def _render_asset_tree(self, imgui) -> None:
        if self.reveal_current_asset:
            imgui.set_next_item_open(True, imgui.Cond_.always)
        else:
            imgui.set_next_item_open(True, imgui.Cond_.appearing)

        if not imgui.tree_node("Asset Tree"):
            return

        for root in self.tree_roots:
            self._render_asset_directory(imgui, root)
        imgui.tree_pop()
        self.reveal_current_asset = False

    def render_ui(self, imgui) -> None:
        widen_imgui_scrollbar(imgui)
        self.last_panel_scroll_y = assist_imgui_window_wheel_scroll(imgui, self.last_panel_scroll_y)
        imgui.set_next_item_open(True, imgui.Cond_.appearing)
        if not imgui.collapsing_header("Assets"):
            return

        current_asset = self.urdfs[self.current_index]
        imgui.text(f"Current: [{asset_type_label(current_asset)}] {current_asset.name}")
        imgui.text(f"Path: {asset_listing_path(current_asset, self.source_root)}")
        model_id = copyable_model_id_from_asset(current_asset)
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

        if imgui.button("Import Asset...##asset_import"):
            self.open_asset_dialog()
        if self.file_error is not None:
            imgui.text(f"Import failed: {self.file_error}")
        if self.load_warning is not None:
            imgui.text(f"Asset warning: {self.load_warning}")

        if imgui.button("Reload##asset_reload"):
            self.requested_index = self.current_index
        imgui.same_line()
        if imgui.button("Previous##asset_previous"):
            self.requested_index = (self.current_index - 1) % len(self.urdfs)
        imgui.same_line()
        if imgui.button("Next##asset_next"):
            self.requested_index = (self.current_index + 1) % len(self.urdfs)

        self._render_asset_tree(imgui)

        if current_asset.suffix.lower() == ".urdf" or is_usd_asset(current_asset):
            changed, self_collisions_enabled = imgui.checkbox(
                "Enable Self Collisions##asset_self_collisions",
                self.self_collisions_enabled,
            )
            if changed:
                self.set_self_collisions(self_collisions_enabled)

            changed, collapse_fixed_joints_enabled = imgui.checkbox(
                "Collapse Fixed Joints##asset_collapse_fixed_joints",
                self.collapse_fixed_joints_enabled,
            )
            if changed:
                self.set_collapse_fixed_joints(collapse_fixed_joints_enabled)

        if current_asset.suffix.lower() == ".urdf":
            changed, floating_enabled = imgui.checkbox(
                "Floating Base##asset_floating",
                self.floating_enabled,
            )
            if changed:
                self.set_floating(floating_enabled)
        elif is_usd_asset(current_asset):
            root_mode_index = USD_ROOT_MODES.index(self.usd_root_mode)
            changed, root_mode_index = imgui.combo(
                "USD Root Mode##asset_usd_root_mode",
                root_mode_index,
                [USD_ROOT_MODE_LABELS[mode] for mode in USD_ROOT_MODES],
            )
            if changed:
                self.set_usd_root_mode(USD_ROOT_MODES[root_mode_index])
        else:
            imgui.text("GLB rigid body: right-drag to test physics")

        solver_index = SOLVER_NAMES.index(self.solver_name)
        changed, solver_index = imgui.combo(
            "Physics Solver##asset_solver",
            solver_index,
            [SOLVER_LABELS[name] for name in SOLVER_NAMES],
        )
        if changed:
            self.set_solver(SOLVER_NAMES[solver_index])

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
        camera_controls: CameraControlPanel | None = None,
    ) -> None:
        self.args = args
        self.urdfs = urdfs
        self.viewer = viewer
        self.double_sided_state = double_sided_state
        self.asset_browser = asset_browser
        self.camera_controls = camera_controls
        self.loaded: LoadedAsset | None = None
        self.sim_time = 0.0

    def load(self, index: int, show_splash: bool = False) -> bool:
        if index < 0 or index >= len(self.urdfs):
            raise IndexError(f"Asset index must be between 0 and {len(self.urdfs) - 1}; got {index}")

        asset_path = self.urdfs[index]
        label = describe_asset_path(asset_path)
        self.args.self_collisions = self.asset_browser.self_collisions_enabled
        self.args.collapse_fixed_joints = self.asset_browser.collapse_fixed_joints_enabled
        self.args.floating = self.asset_browser.floating_enabled
        self.args.usd_root_mode = self.asset_browser.usd_root_mode
        self.args.solver = self.asset_browser.solver_name
        if show_splash:
            self._show_splash(f"Loading {label}...")

        try:
            if asset_path.suffix.lower() == ".urdf":
                validate_urdf_xml(asset_path)
            model, state, joint_panel = build_model(self.args, asset_path)
            solver = None
            state_next = None
            control = None
            contacts = None
            physics_enabled = asset_path.suffix.lower() == ".glb" or (
                self.args.simulate and model.body_count > 0
            )
            if physics_enabled:
                try:
                    solver = create_solver(model, self.args.solver, self.args.iterations)
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to initialize {SOLVER_LABELS[self.args.solver]} solver for {asset_path}: {exc}"
                    ) from exc
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
            if self.camera_controls is not None:
                self.camera_controls.set_asset(compute_asset_bounds(model, state), model, state)
            set_double_sided_rendering(self.viewer, self.double_sided_state["enabled"])
            self._set_reset_callback(self._make_reset_callback(joint_panel, solver))

            traction_monitor = (
                TractionForceMonitor(model, self.args.traction_print_hz, self.args.print_traction_force)
                if physics_enabled
                else None
            )
            self._register_asset_ui(joint_panel, traction_monitor)

            self.loaded = LoadedAsset(
                index=index,
                asset_path=asset_path,
                model=model,
                state=state,
                joint_panel=joint_panel,
                solver=solver,
                state_next=state_next,
                control=control,
                contacts=contacts,
                traction_monitor=traction_monitor,
            )
            self.asset_browser.set_current_index(index)
            self.asset_browser.file_error = None
            self.asset_browser.load_warning = (
                "USD contains no rigid bodies; imported geometry is static and cannot be manipulated dynamically."
                if is_usd_asset(asset_path) and model.body_count == 0
                else None
            )
            self.sim_time = 0.0
            self._print_loaded_summary(label, asset_path, model, joint_panel)
            return True
        except Exception as exc:
            message = str(exc) or type(exc).__name__
            self.asset_browser.file_error = f"Failed to load {label}: {message}"
            self.asset_browser.load_warning = None
            print(self.asset_browser.file_error, file=sys.stderr)
            return False
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

    def _make_reset_callback(self, joint_panel: JointControlPanel, solver: Any | None):
        def reset() -> None:
            joint_panel.reset()
            if solver is not None:
                solver.reset(joint_panel.state)

        return reset

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
        asset_path: Path,
        model: newton.Model,
        joint_panel: JointControlPanel,
    ) -> None:
        print(f"Opening {label}: {asset_path}")
        print(
            f"Loaded {model.body_count} bodies, {model.joint_count} joints, "
            f"{model.shape_count} shapes, {model.joint_coord_count} joint coords."
        )
        if (
            asset_path.suffix.lower() == ".glb"
            or self.args.simulate
            and model.body_count > 0
        ):
            print(f"Physics solver: {SOLVER_LABELS[self.args.solver]}")
        if asset_path.suffix.lower() == ".glb":
            print(
                f"GLB rigid body: mass={self.args.glb_mass:g} kg, "
                f"collision={self.args.glb_collision}; right-drag to apply picking forces."
            )
        elif asset_path.suffix.lower() == ".urdf":
            print(f"Floating base: {self.args.floating}")
            print(f"Collapse fixed joints: {self.args.collapse_fixed_joints}")
            print(f"Articulation metadata: {describe_metadata(joint_panel.articulation_metadata)}")
            print(f"Interactive controls: {len(joint_panel.controls)} joint controls in the left panel.")
        else:
            print(f"USD root mode: {USD_ROOT_MODE_LABELS[self.args.usd_root_mode]}")
            print(f"Collapse fixed joints: {self.args.collapse_fixed_joints}")
            print(f"Interactive controls: {len(joint_panel.controls)} joint controls in the left panel.")
            if model.body_count == 0:
                print(
                    "WARNING: USD contains no rigid bodies; imported geometry is static "
                    "and cannot be manipulated dynamically.",
                    file=sys.stderr,
                )


def set_double_sided_rendering(viewer: newton.viewer.ViewerGL, enabled: bool) -> None:
    """Toggle back-face culling for meshes already registered in ViewerGL."""
    objects = getattr(viewer, "objects", {})
    for obj in objects.values():
        if hasattr(obj, "backface_culling"):
            obj.backface_culling = not enabled


def viewer_requests_physics_step(viewer: Any) -> bool:
    """Keep a paused simulation responsive while the user right-drags a body."""
    if viewer.should_step():
        return True
    picking = getattr(viewer, "picking", None)
    is_picking = getattr(picking, "is_picking", None)
    return bool(is_picking()) if callable(is_picking) else False


def run_viewer(
    args: argparse.Namespace,
    urdfs: list[Path],
) -> None:
    initial_position = 0
    resolved_source = args.source.expanduser().resolve(strict=False)
    source_root = resolved_source if resolved_source.is_dir() else resolved_source.parent
    configure_warp_cpu_fallback()
    viewer = newton.viewer.ViewerGL(
        width=args.width,
        height=args.height,
        headless=args.headless,
        paused=not (
            args.start_running
            and (args.simulate or urdfs[initial_position].suffix.lower() == ".glb")
        ),
    )

    double_sided_state = {"enabled": args.double_sided}
    asset_browser = AssetBrowser(
        urdfs,
        initial_position,
        self_collisions_enabled=args.self_collisions,
        collapse_fixed_joints_enabled=args.collapse_fixed_joints,
        floating_enabled=args.floating,
        usd_root_mode=args.usd_root_mode,
        solver_name=args.solver,
        source_root=source_root,
    )
    camera_controls = CameraControlPanel(
        viewer,
        auto_frame=args.auto_frame,
        padding=args.camera_padding,
        speed=args.camera_speed,
        wheel_sensitivity=args.camera_wheel_sensitivity,
        fine_scale=args.camera_fine_scale,
    )
    runtime = AssetViewerRuntime(
        args,
        urdfs,
        viewer,
        double_sided_state,
        asset_browser,
        camera_controls,
    )

    if hasattr(viewer, "register_ui_callback"):
        viewer.register_ui_callback(asset_browser.render_ui, position="panel")

    def render_double_sided_option(imgui) -> None:
        widen_imgui_scrollbar(imgui)
        changed, enabled = imgui.checkbox("Double-sided Meshes", double_sided_state["enabled"])
        if changed:
            double_sided_state["enabled"] = enabled
        set_double_sided_rendering(viewer, double_sided_state["enabled"])

    if hasattr(viewer, "register_ui_callback"):
        viewer.register_ui_callback(camera_controls.render_ui, position="rendering")
        viewer.register_ui_callback(render_double_sided_option, position="rendering")

    if not runtime.load(initial_position):
        viewer.close()
        return

    frame_dt = 1.0 / args.fps
    sim_dt = frame_dt / args.substeps
    frame = 0

    has_glb = any(path.suffix.lower() == ".glb" for path in urdfs)
    if not args.simulate and not has_glb and args.print_traction_force:
        print("Traction force printing is idle because --simulate is off.")

    while viewer.is_running():
        requested_index = asset_browser.consume_request()
        if requested_index is not None:
            runtime.load(requested_index, show_splash=True)

        loaded = runtime.loaded
        if loaded is None:
            break

        if loaded.solver is not None and viewer_requests_physics_step(viewer):
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
        camera_controls.set_state(loaded.state)
        camera_controls.apply_sensitivity()

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
    parser = argparse.ArgumentParser(description="Inspect URDF, USD, and GLB assets with Newton.")
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Asset directory or one supported URDF, USD, or GLB file.",
    )
    parser.add_argument("--list", action="store_true", help="List matching asset files and exit.")
    parser.add_argument(
        "--category",
        action="append",
        default=None,
        help="Limit discovery to a category folder, e.g. microwave or scissors. Repeat for multiple categories.",
    )
    parser.add_argument("--scale", type=float, default=1.0, help="Scale for URDF and GLB imports.")
    parser.add_argument(
        "--glb-mass",
        type=float,
        default=1.0,
        help="Mass in kilograms assigned to an imported GLB rigid body.",
    )
    parser.add_argument(
        "--glb-collision",
        choices=("mesh", "convex"),
        default="mesh",
        help="GLB collider: exact triangle mesh for inspection or per-primitive convex hull for robust dynamics.",
    )
    parser.add_argument(
        "--glb-max-hull-vertices",
        type=int,
        default=64,
        help="Maximum convex-hull vertices per GLB primitive.",
    )
    parser.add_argument("--z", type=float, default=5.0, help="Vertical offset for the imported model.")
    parser.add_argument(
        "--ground",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add a ground plane at z=0 (enabled by default).",
    )
    parser.add_argument("--show-colliders", action="store_true", help="Show collision meshes even when visual meshes exist.")
    parser.add_argument(
        "--self-collisions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable collisions between links in the same imported asset.",
    )
    parser.add_argument(
        "--collapse-fixed-joints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Merge bodies connected by fixed joints (disabled by default).",
    )
    parser.add_argument(
        "--floating",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Import a URDF with a free-floating root body (enabled by default).",
    )
    parser.add_argument(
        "--usd-root-mode",
        choices=USD_ROOT_MODES,
        default="authored",
        help="USD root mobility: preserve authored/default behavior, force floating, or force fixed.",
    )
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
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run Newton dynamics for URDF and USD assets. GLB rigid bodies are always physics-enabled.",
    )
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
    parser.add_argument(
        "--solver",
        choices=SOLVER_NAMES,
        default="mujoco",
        help="Physics solver to use (default: mujoco).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Iteration count for XPBD, VBD, and MuJoCo solvers.",
    )
    parser.add_argument("--camera-x", type=float, default=0.65)
    parser.add_argument("--camera-y", type=float, default=-0.75)
    parser.add_argument("--camera-z", type=float, default=5.0)
    parser.add_argument("--pitch", type=float, default=-25.0)
    parser.add_argument("--yaw", type=float, default=40.0)
    parser.add_argument(
        "--auto-frame",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically point the camera at and frame each loaded asset.",
    )
    parser.add_argument(
        "--camera-padding",
        type=float,
        default=1.35,
        help="Padding around an automatically framed asset.",
    )
    parser.add_argument(
        "--camera-speed",
        type=float,
        default=None,
        help="Keyboard camera speed in m/s (default: automatic based on asset size).",
    )
    parser.add_argument(
        "--camera-wheel-sensitivity",
        type=float,
        default=0.08,
        help="Mouse-wheel dolly sensitivity.",
    )
    parser.add_argument(
        "--camera-fine-scale",
        type=float,
        default=0.1,
        help="Sensitivity multiplier while Shift or Fine Camera Mode is active.",
    )
    args = parser.parse_args(argv)
    if args.scale <= 0.0:
        parser.error("--scale must be greater than 0")
    if args.glb_mass <= 0.0:
        parser.error("--glb-mass must be greater than 0")
    if args.glb_max_hull_vertices < 4:
        parser.error("--glb-max-hull-vertices must be at least 4")
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if args.camera_padding <= 0.0:
        parser.error("--camera-padding must be greater than 0")
    if args.camera_speed is not None and args.camera_speed <= 0.0:
        parser.error("--camera-speed must be greater than 0")
    if args.camera_wheel_sensitivity <= 0.0:
        parser.error("--camera-wheel-sensitivity must be greater than 0")
    if not 0.0 < args.camera_fine_scale <= 1.0:
        parser.error("--camera-fine-scale must be in the range (0, 1]")
    return args


def main() -> None:
    args = parse_args()
    categories = tuple(args.category) if args.category else None
    assets = find_assets(args.source, categories)
    resolved_source = args.source.expanduser().resolve(strict=False)
    source_root = resolved_source if resolved_source.is_dir() else resolved_source.parent

    if args.list:
        print_assets(assets, source_root)
        return

    run_viewer(args, assets)


if __name__ == "__main__":
    main()
