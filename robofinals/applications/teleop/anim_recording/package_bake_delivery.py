#!/usr/bin/env python3
"""Package a teleop bake run into a self-contained delivery folder.

Given an input directory such as

    .../anim_recordings/2026_04_24_140816/

which contains the outputs of ``teleop_main.py`` (``teleop_baked_with_deformable.usda``
plus the per-layer USD files and OVD artifacts), this module:

1. Resolves every USD / texture dependency via
   ``UsdUtils.ComputeAllDependencies``.
2. Copies every dependency into ``<output_dir>/`` while keeping a clean,
   self-contained layout:

       <output_dir>/
           teleop_baked_with_deformable.usda   (entry point)
           flattened_playback_over.usda
           layer_over.usda
           layer_root.usda
           deformable_animation_over.usda      (kept as a backup)
           deformable_bake_debug.json
           scene/
               old_schema/
                   overview.usd
                   Bag001/Bag001.usd + texture/*.png
                   ...
           robot/
               g1_wuji.usd

3. Rewrites every ``@...@`` asset reference inside the copied USD layers via
   ``UsdUtils.ModifyAssetPaths`` so the output folder is fully relocatable
   (no absolute paths, no ``../../..`` escapes).

Unresolved Omniverse MDL shaders (``OmniPBR.mdl`` / ``OmniSurface.mdl``) are
left untouched: those are provided by the Kit runtime at open time.

The module can be invoked in two ways:

* As a function from within ``teleop_main.py`` (pxr imports are deferred so
  that ``import package_bake_delivery`` is safe before ``AppLauncher`` has
  started Isaac Sim):

      from robofinals.applications.teleop.anim_recording import package_bake_delivery
      package_bake_delivery(input_dir, output_dir)

* As a standalone script using Isaac Sim's bundled Python:

      python.sh package_bake_delivery.py --input <run_dir> [--output <dst>]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys


ENTRY_USD_NAME = "teleop_baked_with_deformable.usda"

# Dependencies outside these prefix sets are routed to ``external/`` so the
# delivery folder is always complete. Order matters: first match wins.
_SCENE_TOKEN = "old_schema"
_ROBOT_ASSET_SUBPATHS = (
    os.path.join("robofinals", "data", "assets"),
    os.path.join("lwlab", "data", "assets"),
)


def _norm(path: str) -> str:
    return os.path.normpath(os.path.realpath(path))


def _split_dependency_path(path: str) -> tuple[str, str]:
    """Return ``(outer_path, inner_package_path)`` for a USD dependency path.

    USD package-relative asset paths use the syntax ``foo.usdz[layer.usd]``.
    Only the outer package exists on the host filesystem, so copying and
    destination classification must operate on ``foo.usdz`` while reference
    rewriting must preserve the ``[layer.usd]`` suffix.
    """
    from pxr import Ar

    if path and Ar.IsPackageRelativePath(path):
        outer_path, inner_path = Ar.SplitPackageRelativePathOuter(path)
        return _norm(outer_path), inner_path
    return _norm(path), ""


def _unique_paths(paths: list[str]) -> list[str]:
    """Preserve order while dropping duplicates."""
    return list(dict.fromkeys(paths))


def _classify_destination(source_abs: str, input_dir: str, output_dir: str) -> str:
    """Decide where a dependency should live inside ``output_dir``."""
    source_abs = _norm(source_abs)
    input_dir = _norm(input_dir)

    if source_abs == input_dir or source_abs.startswith(input_dir + os.sep):
        rel = os.path.relpath(source_abs, input_dir)
        return os.path.join(output_dir, rel)

    tokens = source_abs.split(os.sep)

    if _SCENE_TOKEN in tokens:
        idx = tokens.index(_SCENE_TOKEN)
        rel = os.sep.join(tokens[idx:])
        return os.path.join(output_dir, "scene", rel)

    for robot_asset_subpath in _ROBOT_ASSET_SUBPATHS:
        robot_norm = os.sep + robot_asset_subpath + os.sep
        if robot_norm in source_abs:
            rel = source_abs.split(robot_norm, 1)[1]
            return os.path.join(output_dir, "robot", rel)

    return os.path.join(output_dir, "external", os.path.basename(source_abs))


def _compute_dependencies(entry_path: str):
    from pxr import UsdUtils

    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(entry_path)
    layer_paths = _unique_paths(
        [_split_dependency_path(layer.realPath)[0] for layer in layers if layer.realPath]
    )
    asset_paths = _unique_paths([_split_dependency_path(a)[0] for a in assets])
    return layer_paths, asset_paths, list(unresolved)


def _build_path_mapping(
    layer_paths: list[str],
    asset_paths: list[str],
    input_dir: str,
    output_dir: str,
) -> dict:
    mapping: dict[str, str] = {}
    for src in layer_paths + asset_paths:
        mapping[src] = _norm(_classify_destination(src, input_dir, output_dir))
    return mapping


def _copy_binary_dependency(src: str, dst: str) -> None:
    src, _ = _split_dependency_path(src)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def _write_rewritten_layer(
    src_layer_path: str,
    dst_layer_path: str,
    path_mapping: dict,
) -> None:
    """Copy ``src_layer_path`` to ``dst_layer_path`` and rewrite every
    ``@...@`` asset reference so it points at the relocated dependency.

    This uses ``UsdUtils.ModifyAssetPaths`` which covers sublayers,
    references, payloads, variants and shader asset attributes in one pass.
    """
    from pxr import Sdf, UsdUtils

    src_layer = Sdf.Layer.FindOrOpen(src_layer_path)
    if src_layer is None:
        raise RuntimeError(f"Cannot open layer: {src_layer_path}")

    dst_dir = os.path.dirname(dst_layer_path)
    os.makedirs(dst_dir, exist_ok=True)

    # Work on an in-memory copy so the original file on disk is untouched.
    extension = os.path.splitext(dst_layer_path)[1].lstrip(".").lower()
    scratch_format = "usda" if extension == "usda" else "usdc"
    scratch_layer = Sdf.Layer.CreateAnonymous(
        f".__delivery__.{scratch_format}",
        args={"format": scratch_format},
    )
    scratch_layer.TransferContent(src_layer)

    def _modifier(asset_path: str) -> str:
        from pxr import Ar

        if not asset_path:
            return asset_path

        abs_src = src_layer.ComputeAbsolutePath(asset_path)
        abs_src_norm = ""
        package_inner_path = ""
        if abs_src:
            abs_src_norm, package_inner_path = _split_dependency_path(abs_src)

        if abs_src_norm in path_mapping:
            abs_dst_outer = path_mapping[abs_src_norm]
            rel = os.path.relpath(abs_dst_outer, dst_dir).replace("\\", "/")
            if not rel.startswith("."):
                rel = "./" + rel
            if package_inner_path:
                rel = Ar.JoinPackageRelativePath(rel, package_inner_path)
            return rel

        return asset_path

    UsdUtils.ModifyAssetPaths(scratch_layer, _modifier)
    scratch_layer.Export(dst_layer_path)


def package_bake_delivery(
    input_dir: str,
    output_dir: str | None = None,
    *,
    logger=None,
) -> str:
    """Package the bake run at ``input_dir`` into a standalone folder.

    Parameters
    ----------
    input_dir:
        Directory containing ``teleop_baked_with_deformable.usda`` plus the
        other layers produced by ``teleop_main.py``.
    output_dir:
        Destination directory. If ``None`` the delivery is written next to
        the input directory using ``<input>_delivery``.
    logger:
        Optional logger with an ``info`` method. Prints to stdout otherwise.

    Returns
    -------
    str
        Absolute path of the generated delivery directory.
    """

    def _log(msg: str) -> None:
        if logger is not None and hasattr(logger, "info"):
            logger.info(msg)
        else:
            print(msg, flush=True)

    input_dir = _norm(input_dir)
    entry_path = os.path.join(input_dir, ENTRY_USD_NAME)
    if not os.path.exists(entry_path):
        raise FileNotFoundError(f"Entry USD missing: {entry_path}")

    if output_dir is None:
        output_dir = input_dir.rstrip(os.sep) + "_delivery"
    output_dir = _norm(output_dir)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    _log(f"[Delivery] input_dir  = {input_dir}")
    _log(f"[Delivery] output_dir = {output_dir}")

    layer_paths, asset_paths, unresolved = _compute_dependencies(entry_path)
    _log(
        f"[Delivery] dependencies: {len(layer_paths)} layers, "
        f"{len(asset_paths)} assets, {len(unresolved)} unresolved"
    )
    for u in unresolved:
        _log(f"[Delivery] unresolved (kept as-is): {u}")

    mapping = _build_path_mapping(layer_paths, asset_paths, input_dir, output_dir)

    for asset_src in asset_paths:
        asset_dst = mapping[asset_src]
        _copy_binary_dependency(asset_src, asset_dst)

    for layer_src in layer_paths:
        layer_dst = mapping[layer_src]
        _write_rewritten_layer(layer_src, layer_dst, mapping)

    # Keep additional artifacts that help the customer: baked animation
    # layer (even though ``flattened_playback_over.usda`` inlines the time
    # samples) plus the diagnostic JSON.
    extras = [
        "rigid_body_over.usda",
        "deformable_animation_over.usda",
        "deformable_bake_debug.json",
    ]
    for extra in extras:
        src = os.path.join(input_dir, extra)
        dst = os.path.join(output_dir, extra)
        if not os.path.exists(src) or os.path.exists(dst):
            continue
        if extra.endswith(".usda") or extra.endswith(".usd") or extra.endswith(".usdc"):
            _write_rewritten_layer(src, dst, mapping)
        else:
            _copy_binary_dependency(src, dst)

    _log(f"[Delivery] done. Entry point: {os.path.join(output_dir, ENTRY_USD_NAME)}")
    return output_dir


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="Bake run directory containing teleop_baked_with_deformable.usda",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Delivery output directory (default: <input>_delivery)",
    )
    args = parser.parse_args()

    output_dir = package_bake_delivery(args.input, args.output)
    print(output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
