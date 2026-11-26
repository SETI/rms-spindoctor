"""The typed ``voyager_iss_star_photometry`` configuration section.

The section holds what :class:`~spindoctor.obs.obs_inst_voyager_iss.ObsVoyagerISS` needs
to find the faintest catalog star a frame can show, and the width of a star in the image,
for each of the four Voyager cameras.  Each camera has its own entry: no two cameras
share optics or a vidicon.  The values live in the packaged YAML only, so no field has a
default.
"""

from dataclasses import dataclass
from typing import Literal

from spindoctor.schema import SCHEMA_CONFIG, Float, NonNeg, Positive

__all__ = [
    'VoyagerISSFilter',
    'VoyagerISSGainState',
    'VoyagerISSStarPhotometry',
    'VoyagerISSStarPhotometrySection',
]

#: A Voyager ISS gain state, as the PDS3 label's ``GAIN_MODE_ID`` spells it.
VoyagerISSGainState = Literal['LOW', 'HIGH']

#: A filter of either Voyager camera, as the label spells it.
VoyagerISSFilter = Literal[
    'CLEAR', 'VIOLET', 'BLUE', 'GREEN', 'ORANGE', 'UV', 'SODIUM', 'CH4_JS', 'CH4_U'
]


@dataclass(frozen=True)
class VoyagerISSStarPhotometry:
    """The star photometry and star width of one Voyager camera.

    A vidicon counts no electrons, so the rate is in DN.  A star of V magnitude ``m``
    deposits ``flux_dn_per_s_vmag0 * 10**(-0.4 * (m + filter_mag_offset[filter])) * texp
    * dn_gain_factor[gain]`` DN.  The dimmest visible star deposits ``min_detectable_dn``,
    and the catalog search reaches ``search_depth_mag`` past it.

    Parameters:
        flux_dn_per_s_vmag0: Raw DN per second a V = 0 star deposits through the clear
            filter in the low gain state.
        dn_gain_factor: The factor each gain state multiplies a star's DN by, against
            the low gain state.
        min_detectable_dn: Raw DN of the dimmest star the frame shows.
        search_depth_mag: Magnitudes past the dimmest visible star the catalog search
            reaches.
        filter_mag_offset: Magnitudes each filter dims a star by, against the clear
            filter.
        star_psf_sigma: Gaussian sigma, in pixels of the geometrically corrected image,
            of a star.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    flux_dn_per_s_vmag0: Positive
    dn_gain_factor: dict[VoyagerISSGainState, Positive]
    min_detectable_dn: Positive
    search_depth_mag: NonNeg
    filter_mag_offset: dict[VoyagerISSFilter, Float]
    star_psf_sigma: Positive


@dataclass(frozen=True)
class VoyagerISSStarPhotometrySection:
    """The ``voyager_iss_star_photometry`` section: one entry per camera.

    Parameters:
        vg1_nac: The Voyager 1 narrow-angle camera.
        vg1_wac: The Voyager 1 wide-angle camera.
        vg2_nac: The Voyager 2 narrow-angle camera.
        vg2_wac: The Voyager 2 wide-angle camera.
    """

    __pydantic_config__ = SCHEMA_CONFIG

    vg1_nac: VoyagerISSStarPhotometry
    vg1_wac: VoyagerISSStarPhotometry
    vg2_nac: VoyagerISSStarPhotometry
    vg2_wac: VoyagerISSStarPhotometry
