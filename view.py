"""Compatibility entry point for the modular Artiverse asset viewer.

New code should import from :mod:`asset_viewer`.  This module keeps the
original `python view.py` command and public names working.
"""

# This compatibility module intentionally re-exports the implementation
# modules' public API for existing scripts and tests.
# ruff: noqa: F401

from asset_viewer.app import (
    AssetBrowser,
    AssetViewerRuntime,
    LoadedAsset,
    USD_ROOT_MODE_LABELS,
    USD_ROOT_MODES,
    build_model,
    choose_asset_file,
    choose_urdf_file,
    configure_warp_cpu_fallback,
    main,
    parse_args,
    run_viewer,
    set_double_sided_rendering,
    usd_root_floating,
    viewer_requests_physics_step,
)
from asset_viewer.assets import (
    DEFAULT_SOURCE,
    SUPPORTED_ASSET_SUFFIXES,
    USD_ASSET_SUFFIXES,
    ArticulationMetadata,
    ArticulationRecord,
    AssetTreeFile,
    AssetTreeNode,
    asset_listing_path,
    asset_path_relative_to_root,
    asset_type_label,
    build_asset_tree,
    collect_explicit_dependencies,
    copy_text_to_clipboard,
    copyable_model_id_from_asset,
    copyable_model_id_from_urdf,
    describe_asset_path,
    describe_metadata,
    describe_urdf_path,
    find_assets,
    find_articulation_json,
    find_urdfs,
    load_articulation_metadata,
    matches_categories,
    model_dir_from_urdf,
    print_assets,
    print_urdfs,
    unique_sorted_paths,
    validate_urdf_xml,
)
from asset_viewer.controls import (
    BINARY_PRISMATIC_MAX_TRAVEL,
    BINARY_PRISMATIC_NAME_TOKENS,
    DEFAULT_CONTINUOUS_SPEED_DEGREES,
    SIDEBAR_SCROLLBAR_SIZE,
    SIDEBAR_WHEEL_SCROLL_PIXELS,
    UNBOUNDED_LIMIT,
    JointControl,
    JointControlPanel,
    assist_imgui_window_wheel_scroll,
    format_optional,
    imgui_current_window_hovered,
    imgui_float_call,
    imgui_mouse_wheel,
    imgui_primary_mouse_down,
    parse_joint_pid,
    set_imgui_scroll_y,
    short_repr,
    widen_imgui_scrollbar,
    wrap_angle_radians,
)
from asset_viewer.camera import (
    AssetBounds,
    CameraControlPanel,
    compute_asset_bounds,
    frame_camera_on_bounds,
    recommended_camera_speed,
)
from asset_viewer.solvers import SOLVER_LABELS, SOLVER_NAMES, create_solver, prepare_builder_for_solver
from asset_viewer.traction import (
    TractionForceMonitor,
    TractionForceSample,
    compute_picking_traction_force,
    rotate_vector_by_quat,
    transform_point_np,
    vec3_array,
)


if __name__ == "__main__":
    main()
