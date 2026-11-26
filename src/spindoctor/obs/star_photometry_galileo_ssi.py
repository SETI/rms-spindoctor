"""The typed ``galileo_ssi_star_photometry`` configuration section.

The section holds what :class:`~spindoctor.obs.obs_inst_galileo_ssi.ObsGalileoSSI` needs
to find the faintest catalog star a frame can show, and the width of a star in the image.
The camera reads out either a full frame or a 2x2 summed frame; summation changes the
noise of a pixel and the width of a star in pixels, so those values are given per readout
mode.  The values live in the packaged YAML only, so no field has a default.
"""

from dataclasses import dataclass
from typing import Literal

from spindoctor.schema import SCHEMA_CONFIG, Float, NonNeg, Positive, RuleError

__all__ = [
    'GalileoSSIFilter',
    'GalileoSSIGainState',
    'GalileoSSIReadoutMode',
    'GalileoSSIStarPhotometry',
    'GalileoSSIStarPhotometrySection',
]

#: A Galileo SSI gain state, named by the full-well signal it allows, as the PDS3 label's
#: ``GAIN_MODE_ID`` spells it.
GalileoSSIGainState = Literal['10K', '40K', '100K', '400K']

#: A Galileo SSI readout mode: ``FULL`` for a full 800 x 800 frame, ``SUMMED`` for a frame
#: read out in 2x2 summation, 400 x 400.
GalileoSSIReadoutMode = Literal['FULL', 'SUMMED']

#: A Galileo SSI filter, as the label's ``FILTER_NAME`` spells it.
GalileoSSIFilter = Literal[
    'CLEAR', 'VIOLET', 'GREEN', 'RED', 'IR-7270', 'IR-7560', 'IR-8890', 'IR-9680'
]


@dataclass(frozen=True)
class GalileoSSIStarPhotometry:
    """The star photometry and star width of the Galileo SSI camera.

    A star of V magnitude ``m`` deposits
    ``flux_electrons_per_s_vmag0 * 10**(-0.4 * (m + filter_mag_offset[filter])) * texp /
    electrons_per_dn[gain]`` DN.  The dimmest visible star deposits
    ``min_detectable_dn[mode]``, and the catalog search reaches ``search_depth_mag`` past
    it.

    Parameters:
        flux_electrons_per_s_vmag0: Electrons per second a V = 0 star deposits through
            the clear filter.
        electrons_per_dn: Electrons per DN of each gain state.
        min_detectable_dn: DN of the dimmest star a frame of each readout mode shows.
        search_depth_mag: Magnitudes past the dimmest visible star the catalog search
            reaches.
        filter_mag_offset: Magnitudes each filter dims a star by, against the clear
            filter.
        star_psf_sigma: Gaussian sigma of a star, in the pixels of a frame of each readout
            mode.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    flux_electrons_per_s_vmag0: Positive
    electrons_per_dn: dict[GalileoSSIGainState, Positive]
    min_detectable_dn: dict[GalileoSSIReadoutMode, Positive]
    search_depth_mag: NonNeg
    filter_mag_offset: dict[GalileoSSIFilter, Float]
    star_psf_sigma: dict[GalileoSSIReadoutMode, Positive]

    def __post_init__(self) -> None:
        """Refuse tables that do not give the same readout modes.

        A frame reads both tables for its readout mode, so a mode one table gives and the
        other leaves out would be refused only when a frame of that mode arrives.
        """
        if set(self.star_psf_sigma) != set(self.min_detectable_dn):
            raise RuleError(
                f'gives the readout modes {sorted(self.star_psf_sigma)}, but '
                f'min_detectable_dn gives {sorted(self.min_detectable_dn)}; each table gives '
                'the same modes',
                at=('star_psf_sigma',),
                reads=(('star_psf_sigma',), ('min_detectable_dn',)),
            )


@dataclass(frozen=True)
class GalileoSSIStarPhotometrySection:
    """The ``galileo_ssi_star_photometry`` section.

    Parameters:
        ssi: The SSI camera's star photometry.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    ssi: GalileoSSIStarPhotometry
