"""Private scanpy plotting helpers that grassp reuses, resolved for the installed scanpy.

scanpy 1.13 moved its matplotlib plotting code from ``scanpy.plotting`` into
``scanpy.plotting.legacy`` to make room for a second, holoviews-based backend. Attribute access
such as ``scanpy.pl._utils`` still reaches the legacy modules through a module ``__getattr__``,
but only while the default preset is active, and ``import scanpy.plotting._utils`` fails
outright. Import the helpers from here instead of from scanpy.
"""

from __future__ import annotations

try:  # scanpy >= 1.13
    from scanpy.plotting.legacy import _utils as pl_utils
    from scanpy.plotting.legacy._tools import scatterplots
except ImportError:  # scanpy < 1.13
    from scanpy.plotting import _utils as pl_utils
    from scanpy.plotting._tools import scatterplots

_color_vector = scatterplots._color_vector
_components_to_dimensions = scatterplots._components_to_dimensions
_get_color_source_vector = scatterplots._get_color_source_vector
_get_palette = scatterplots._get_palette

if hasattr(pl_utils, "set_default_colors_for_categorical_obs"):  # scanpy >= 1.12
    set_default_colors_for_categorical_obs = pl_utils.set_default_colors_for_categorical_obs
else:  # scanpy < 1.12 had the same function under a private name
    set_default_colors_for_categorical_obs = (
        pl_utils._set_default_colors_for_categorical_obs  # pylint: disable=no-member
    )
