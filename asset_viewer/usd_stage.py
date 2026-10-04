"""Create a metadata-only USD root in memory, keeping every source layer read-only."""
from pathlib import Path
from typing import Any


def open_usd_stage(entrypoint: Path, metadata: dict[str, Any] | None = None) -> Any:
    from pxr import Sdf, Usd

    stage = Usd.Stage.Open(str(entrypoint))
    if stage is None:
        raise ValueError("Cannot open USD entrypoint")
    if not metadata:
        return stage
    if set(metadata) - {"metersPerUnit", "kilogramsPerUnit", "upAxis"}:
        raise ValueError("Only length/mass units and up axis can be adapted")
    # A session layer cannot supply root stage metadata. Use an anonymous root
    # above the original instead; relative dependencies stay anchored to their
    # original layers, and no source layer is edited, exported or saved.
    original = stage.GetRootLayer()
    root = Sdf.Layer.CreateAnonymous("readonly-stage-metadata.usda")
    root.subLayerPaths = [original.identifier]
    for key in original.pseudoRoot.ListInfoKeys():
        if key not in {"subLayers", "subLayerOffsets"}:
            root.pseudoRoot.SetInfo(key, original.pseudoRoot.GetInfo(key))
    for key, value in metadata.items():
        if original.pseudoRoot.HasInfo(key):
            raise ValueError("Stage adaptation must not override authored root metadata")
        root.pseudoRoot.SetInfo(key, value)
    return Usd.Stage.Open(root)
