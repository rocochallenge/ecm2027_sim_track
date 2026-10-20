"""Sub-module for rigid articulated assets."""

__all__ = [
    "GeneralAsset",
    "GeneralAssetCfg",
]


def __getattr__(name: str):
    """Load legacy asset implementations only when explicitly requested.

    Importing ``general_asset_cfg`` installs its historical InteractiveScene
    monkey patch. Standard objects, including schema-aware deformables, should
    not install that patch merely by importing this package.
    """
    if name == "GeneralAsset":
        from .general_asset import GeneralAsset

        globals()[name] = GeneralAsset
        return GeneralAsset
    if name == "GeneralAssetCfg":
        from .general_asset_cfg import GeneralAssetCfg

        globals()[name] = GeneralAssetCfg
        return GeneralAssetCfg
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
