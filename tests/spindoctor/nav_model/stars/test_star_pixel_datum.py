"""The half-pixel datum between a star record and the pixel it lands on.

A star record carries oops uv (``MutableStar``); every star technique
measures its centroids in array indices, because each one builds its
coordinate array with ``np.arange`` over the slice it indexes the image
with.  These tests pin the conversion between the two at the seam where the
model hands a predicted position to a technique, and at the seam where the
simulator's scene builds a record.

The anchor is external to the pipeline: ``oops`` puts a default ``FlatFOV``
optical axis at ``uv_los``, and a star image that is symmetric about the
middle of an ``N``-pixel axis centroids at index ``(N - 1) / 2``.  Neither
half is a restatement of the code under test, so a pipeline that reads uv as
an index fails here by the half pixel that separates them.
"""

from typing import Any, cast

import numpy as np
import pytest
from oops.fov.flatfov import FlatFOV
from psfmodel import GaussianPSF

from spindoctor.feature.geometry import StarGeometry
from spindoctor.nav_model.stars.nav_model_stars import NavModelStars
from spindoctor.nav_orchestrator.nav_context import NavContext
from spindoctor.nav_technique._star_helpers import local_centroid
from spindoctor.sim.star_records import star_record_from_params
from spindoctor.support.types import STAR_UV_DATUM_PX, MutableStar

# Odd, so the FOV's optical axis falls on the centre of one pixel rather than
# on the boundary between two -- a boundary would leave the brightest-pixel
# search a tie to break and put the centroid box off centre.
_FOV_SIZE = 129
_STAR_SIGMA_PX = 1.5
_STAR_PEAK_DN = 1000.0


def _boresight_uv() -> float:
    """Return the FOV optical axis in oops uv, straight from ``oops``."""
    fov = FlatFOV(1e-5, (_FOV_SIZE, _FOV_SIZE))
    return float(fov.uv_los.vals[0])


def _boresight_index() -> float:
    """Return the index of the pixel at the middle of the detector axis."""
    return (_FOV_SIZE - 1) / 2.0


def _star_image(v_index: float, u_index: float) -> np.ndarray:
    """Return a frame holding one round Gaussian star at a pixel index."""
    vs = np.arange(_FOV_SIZE, dtype=np.float64)[:, np.newaxis]
    us = np.arange(_FOV_SIZE, dtype=np.float64)[np.newaxis, :]
    r2 = (vs - v_index) ** 2 + (us - u_index) ** 2
    return _STAR_PEAK_DN * np.exp(-r2 / (2.0 * _STAR_SIGMA_PX**2))


class _StarRecord:
    """Star record stand-in carrying the fields the star model reads."""

    def __init__(self, v: float, u: float) -> None:
        self.catalog_name = 'UCAC4'
        self.unique_number = 1
        self.pretty_name = '1'
        self.name = '1'
        self.v = v
        self.u = u
        self.move_v = 0.0
        self.move_u = 0.0
        self.vmag: float | None = 5.0
        self.photometry_corrected = False
        self.photometry_saturated = False
        self.spectral_class = 'G0'
        self.psf_size = (11, 11)
        self.ra_pm = 0.0
        self.dec_pm = 0.0
        self.conflicts = ''
        self.dn = 1.0


class _StarsConfig:
    """Stand-in for ``config.stars`` exposing only the gates the model reads."""

    max_smear = 100.0
    min_predicted_snr = 0.0


class _Config:
    """Stand-in for ``Config`` exposing only its ``stars`` section."""

    stars = _StarsConfig()


class _Obs:
    """Observation stand-in for a square detector with a chosen extfov margin."""

    def __init__(self, *, extfov_margin: int) -> None:
        self.extfov_margin_v = extfov_margin
        self.extfov_margin_u = extfov_margin
        self.data_shape_v = _FOV_SIZE
        self.data_shape_u = _FOV_SIZE
        self.extdata_shape_vu = (
            _FOV_SIZE + 2 * extfov_margin,
            _FOV_SIZE + 2 * extfov_margin,
        )

    def star_psf(self) -> GaussianPSF:
        """Return the PSF the model reads its Gaussian sigma from."""
        return GaussianPSF(sigma=_STAR_SIGMA_PX)

    def star_max_usable_vmag(self) -> float:
        """Return the limiting magnitude, set well past the test star."""
        return 12.0


class _Context:
    """NavContext stand-in carrying the masks the star model consults."""

    def __init__(self, shape_vu: tuple[int, int]) -> None:
        self.image_noise_sigma = 1.0
        self.saturation_mask_ext = np.zeros(shape_vu, dtype=bool)
        self.cosmic_ray_mask_ext = np.zeros(shape_vu, dtype=bool)


def _predicted_vu(star: _StarRecord, *, extfov_margin: int) -> tuple[float, float]:
    """Return the position the star model predicts for one record."""
    obs = _Obs(extfov_margin=extfov_margin)
    model = NavModelStars('stars', cast(Any, obs), config=cast(Any, _Config()))
    model._stars = [cast(MutableStar, star)]
    context = _Context(obs.extdata_shape_vu)
    feature = model.to_features(cast(NavContext, context))[0]
    geometry = feature.geometry
    assert isinstance(geometry, StarGeometry)
    return geometry.predicted_vu


def test_boresight_star_predicts_the_index_its_image_centroids_to() -> None:
    """A star at the FOV optical axis predicts the pixel its light falls on.

    The star record carries the uv ``oops`` reports for the optical axis and
    the frame holds a star symmetric about the middle pixel of the detector,
    which is where that axis points.  The model's prediction must therefore
    equal the centroid a technique measures, with no margin in the way.
    """
    star = _StarRecord(v=_boresight_uv(), u=_boresight_uv())
    predicted = _predicted_vu(star, extfov_margin=0)
    image = _star_image(_boresight_index(), _boresight_index())
    measured, _peak = local_centroid(
        image,
        predicted,
        search_window_px=4.0,
        centroid_box_half_px=4,
        image_noise_sigma=1.0,
        detection_sigma=5.0,
    )
    assert measured is not None
    assert measured[0] == pytest.approx(_boresight_index(), abs=1e-9)
    assert predicted[0] == pytest.approx(measured[0], abs=1e-9)
    assert predicted[1] == pytest.approx(measured[1], abs=1e-9)


def test_the_margin_does_not_disturb_the_datum() -> None:
    """Padding the frame moves the prediction with the light, not against it.

    The margin counts whole rows and columns of padding, so it carries no
    datum of its own.  Rather than restate that as arithmetic, this pads the
    image by the same margin and asks the same question the unpadded case
    asks: the prediction must land on the centroid of the star's light in the
    padded frame.  A pipeline that applied the datum to the padded position
    instead of the unpadded one, or applied it twice, fails here and not in
    the unpadded test.
    """
    margin = 10
    star = _StarRecord(v=_boresight_uv(), u=_boresight_uv())
    predicted = _predicted_vu(star, extfov_margin=margin)
    padded = np.zeros((_FOV_SIZE + 2 * margin, _FOV_SIZE + 2 * margin), dtype=np.float64)
    padded[margin : margin + _FOV_SIZE, margin : margin + _FOV_SIZE] = _star_image(
        _boresight_index(), _boresight_index()
    )
    measured, _peak = local_centroid(
        padded,
        predicted,
        search_window_px=4.0,
        centroid_box_half_px=4,
        image_noise_sigma=1.0,
        detection_sigma=5.0,
    )
    assert measured is not None
    assert measured[0] == pytest.approx(_boresight_index() + margin, abs=1e-9)
    assert predicted[0] == pytest.approx(measured[0], abs=1e-9)
    assert predicted[1] == pytest.approx(measured[1], abs=1e-9)


def test_scene_star_index_becomes_a_uv_record() -> None:
    """A scene states the pixel a star is drawn on; the record states uv.

    The simulator's builder is the boundary between the scene's convention
    and the record's, so a star the scene puts on pixel ``(40, 60)`` comes
    back as a record half a pixel higher on both axes -- the same numbers the
    catalog reduction would produce for a star landing on that pixel.
    """
    star = star_record_from_params(
        {'v': 40.0, 'u': 60.0, 'vmag': 5.0}, index=0, default_v=0.0, default_u=0.0
    )
    assert star.v == pytest.approx(40.0 + STAR_UV_DATUM_PX, abs=1e-12)
    assert star.u == pytest.approx(60.0 + STAR_UV_DATUM_PX, abs=1e-12)
