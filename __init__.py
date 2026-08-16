if __package__:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    from .spectrum_h3_compat import install_spectrum_h3_compat

    install_spectrum_h3_compat()
else:  # pragma: no cover - pytest may import the repository root as top-level __init__
    # Direct import was never a supported ComfyUI loading mode. Keep pytest's
    # repository-root collection side-effect free rather than importing an
    # ambiguous top-level `nodes` module (ComfyUI itself also owns that name).
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
