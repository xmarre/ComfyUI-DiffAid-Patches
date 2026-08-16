if __package__:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    from .spectrum_h3_compat import install_spectrum_h3_compat
else:  # pragma: no cover - direct pytest/import fallback
    from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    from spectrum_h3_compat import install_spectrum_h3_compat

install_spectrum_h3_compat()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
