"""The typed ``newhorizons_lorri_star_photometry`` configuration section.

The section holds what
:class:`~spindoctor.obs.obs_inst_newhorizons_lorri.ObsNewHorizonsLORRI` needs to find the
faintest catalog star a frame can show, and the width of a star in the image.  LORRI has
no filter and one gain state; its binning mode changes the electrons per DN, the noise
of a pixel, and the width of a star in pixels, so those values are given per binning
mode.  The values live in the packaged YAML only, so no field has a default.
"""

from dataclasses import dataclass
from typing import Literal

from spindoctor.schema import SCHEMA_CONFIG, NonNeg, Positive, RuleError

__all__ = [
    'NewHorizonsLORRIBinning',
    'NewHorizonsLORRIStarPhotometry',
    'NewHorizonsLORRIStarPhotometrySection',
]

#: A LORRI binning mode: ``1X1`` for a 1024-row frame, ``4X4`` for a 256-row frame.
NewHorizonsLORRIBinning = Literal['1X1', '4X4']


@dataclass(frozen=True)
class NewHorizonsLORRIStarPhotometry:
    """The star photometry and star width of the LORRI camera.

    A star of V magnitude ``m`` deposits
    ``flux_electrons_per_s_vmag0 * 10**(-0.4 * m) * texp / electrons_per_dn[binning]`` DN.
    The dimmest visible star deposits ``min_detectable_dn[binning]``, and the catalog
    search reaches ``search_depth_mag`` past it.

    Parameters:
        flux_electrons_per_s_vmag0: Electrons per second a V = 0 star deposits.
        electrons_per_dn: Electrons per DN in each binning mode.
        min_detectable_dn: DN of the dimmest star a frame of each binning mode shows.
        search_depth_mag: Magnitudes past the dimmest visible star the catalog search
            reaches.
        star_psf_sigma: Gaussian sigma of a star, in the pixels of a frame of each binning
            mode.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    flux_electrons_per_s_vmag0: Positive
    electrons_per_dn: dict[NewHorizonsLORRIBinning, Positive]
    min_detectable_dn: dict[NewHorizonsLORRIBinning, Positive]
    search_depth_mag: NonNeg
    star_psf_sigma: dict[NewHorizonsLORRIBinning, Positive]

    def __post_init__(self) -> None:
        """Refuse tables that do not give the same binning modes.

        A frame reads all three tables for its binning mode, so a mode one table gives
        and another leaves out would be refused only when a frame of that mode arrives.
        """
        modes = set(self.electrons_per_dn)
        for key, table in (
            ('min_detectable_dn', self.min_detectable_dn),
            ('star_psf_sigma', self.star_psf_sigma),
        ):
            if set(table) != modes:
                raise RuleError(
                    f'gives the binning modes {sorted(table)}, but electrons_per_dn gives '
                    f'{sorted(modes)}; each table gives the same modes',
                    at=(key,),
                    reads=((key,), ('electrons_per_dn',)),
                )


@dataclass(frozen=True)
class NewHorizonsLORRIStarPhotometrySection:
    """The ``newhorizons_lorri_star_photometry`` section.

    Parameters:
        lorri: The LORRI camera's star photometry.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    lorri: NewHorizonsLORRIStarPhotometry
