# Upstream provenance

Retrieved on 2026-07-20.

## Artiverse viewer

- Repository: https://github.com/3dlg-hcvc/artiverse
- Commit: `44f3d41d015018e9b4dff2cbf01fd0892fe6b2c5`
- Included unchanged: `.gitignore`, `.gitmodules`, `README.md`,
  `requirements.txt`, `view_model.py`, and `utils/`
- `view_model.py` SHA-256:
  `2393A89FF20FBEC1CE5A9CF31591431716AF3EA195DAE0F59272B3F5BD009D14`
- `utils/render_part_articulation.py` SHA-256:
  `5BA4B3D1AABAB3D0C700F0EA72A80EF225705B922C01F667B8F9A87A718A5C20`
- `utils/brown_photostudio_01_4k.exr` SHA-256:
  `618C919371207CFAC00E2256E9CB87D8F6886182167ABA82521854178FF1DA03`

The website files under the official repository's `docs/` directory are not
vendored because they are not runtime dependencies of the asset renderer.

## pygltftoolkit dependency

- Repository: https://github.com/3dlg-hcvc/pygltftoolkit
- Commit: `2275d159f8a24b677858489e1257b274e73582ed`
- Included unchanged under `upstream/external/pygltftoolkit/`

The Artiverse repository declares this path in `.gitmodules` but, at the
commit above, does not contain a gitlink for it. It was therefore retrieved
from the declared official repository and pinned separately.

Neither fetched repository contained a LICENSE file at these commits. The
copied material remains subject to its upstream terms and the Artiverse
dataset remains subject to the terms shown on its Hugging Face dataset page.

`README.md`, `PROVENANCE.md`, `requirements.txt`,
`check_blender_dependencies.py`, and `run_view_model.py` in the parent
directory are local files and are not represented as official Artiverse code.
