"""The bias structure: a per-image pedestal offset and shallow gradients, in DN."""

from __future__ import annotations

from typing import Any

import numpy as np

from spindoctor.support.types import NDArrayFloatType

__all__ = ['add_bias_structure']


def add_bias_structure(
    dn: NDArrayFloatType,
    *,
    pedestal_sigma_dn: float,
    row_gradient_dn: float,
    col_gradient_dn: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Add a bias pedestal offset and low-order bias gradients (DN) in place.

    A per-image pedestal offset (a single seeded draw) rides on the flat bias
    level, and shallow row and column gradients model the read-out bias
    structure.  A no-op when every amplitude is zero.

    Parameters:
        dn: The DN image, modified in place.
        pedestal_sigma_dn: Standard deviation of the per-image pedestal (DN).
        row_gradient_dn: Peak-to-peak row (v-axis) bias gradient (DN).
        col_gradient_dn: Peak-to-peak column (u-axis) bias gradient (DN).
        rng: The effect's seeded generator.

    Returns:
        The realized pedestal offset and gradients for the truth record.
    """
    if pedestal_sigma_dn <= 0.0 and row_gradient_dn <= 0.0 and col_gradient_dn <= 0.0:
        return {'active': False}
    size_v, size_u = dn.shape
    pedestal_dn = 0.0
    if pedestal_sigma_dn > 0.0:
        pedestal_dn = float(rng.normal(0.0, pedestal_sigma_dn))
        dn += pedestal_dn
    if row_gradient_dn > 0.0:
        row_ramp = np.linspace(-0.5 * row_gradient_dn, 0.5 * row_gradient_dn, size_v)
        dn += row_ramp[:, None]
    if col_gradient_dn > 0.0:
        col_ramp = np.linspace(-0.5 * col_gradient_dn, 0.5 * col_gradient_dn, size_u)
        dn += col_ramp[None, :]
    return {
        'active': True,
        'pedestal_dn': pedestal_dn,
        'row_gradient_dn': row_gradient_dn,
        'col_gradient_dn': col_gradient_dn,
    }
