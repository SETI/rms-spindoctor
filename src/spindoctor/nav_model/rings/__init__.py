"""Ring feature domain model for planetary navigation.

This subpackage defines typed domain objects for ring feature data, rendering, and
filtering: immutable dataclasses checked when the configuration loads, and
rendering behavior on ``RingFeature``.

Architecture overview:

- ``ring_types``: Pure frozen dataclasses for orbital parameters. No rendering
  dependencies; safe for lightweight import.
- ``ring_render_context``: Immutable bundle of rendering dependencies passed to
  ``RingFeature.render()``.
- ``ring_render_result``: Lightweight result object returned by rendering.
- ``ring_feature``: Core domain object. Owns backplane-based rendering.
- ``ring_filter``: Four-pass filter pipeline deciding which features to render.
- ``ring_math``: Pure mathematical functions for fade and anti-aliasing.
"""

from .ring_feature import RingFeature
from .ring_filter import RingFeatureFilter
from .ring_math import compute_antialiasing, compute_edge_fade, compute_fade_integral
from .ring_render_context import RingsRenderContext
from .ring_render_result import RingRenderResult
from .ring_types import (
    RingBaseOrbitMode,
    RingEdgeData,
    RingFeatureType,
    RingPerturbationMode,
)

__all__ = [
    'RingBaseOrbitMode',
    'RingEdgeData',
    'RingFeature',
    'RingFeatureFilter',
    'RingFeatureType',
    'RingPerturbationMode',
    'RingRenderResult',
    'RingsRenderContext',
    'compute_antialiasing',
    'compute_edge_fade',
    'compute_fade_integral',
]
