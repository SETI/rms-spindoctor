"""Tests for ``spindoctor.navigate_image_files.navigate_image_files``.

The driver wires the per-image batch loader together with the
orchestrator and the metadata curator.  These tests exercise the
happy / image-load-failure / status=failed paths against a fake
observation class so no holdings are required.

Also covers the annotation-compositing summary-PNG renderer
(``write_summary_png`` and the rendering helper now exposed via
``spindoctor.support.summary_png.grayscale_to_rgb_with_quantile_stretch``)
end-to-end against a synthetic ``Annotations`` collection.
"""

from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from filecache import FCPath
from PIL import Image

from spindoctor.annotation import Annotation, Annotations
from spindoctor.dataset.dataset import ImageFile, ImageFiles
from spindoctor.nav_orchestrator.image_classifier_result import NavImageClassifierResult
from spindoctor.nav_orchestrator.nav_result import NavResult
from spindoctor.nav_orchestrator.provenance import Provenance
from spindoctor.navigate_image_files import (
    _summary_metadata_from_obs_result,
    build_timing_section,
    navigate_image_files,
    write_summary_png,
)
from spindoctor.support.status_reason import NavStatusReason
from spindoctor.support.summary_png import (
    grayscale_to_rgb_with_quantile_stretch as _grayscale_to_rgb_with_quantile_stretch,
)


class _FakeSnapshot:
    """Minimal stand-in for ObsSnapshotInst used by the driver tests."""

    def __init__(
        self, *, blank: bool = False, midtime: float = 100.0, shutter_mode: str | None = None
    ) -> None:
        """Build a fake snapshot carrying one deterministic 32x32 image.

        Parameters:
            blank: True for an all-zero image, which the orchestrator's image
                classifier rejects as carrying no data; False for a fixed-seed
                noise field around a mean of 100, which it accepts.
            midtime: The observation midtime in TDB seconds past J2000.
            shutter_mode: The shutter mode the image was taken in, or ``None``
                for a host whose labels carry no such field.
        """
        rng = np.random.default_rng(seed=99)
        if blank:
            self.data = np.zeros((32, 32), np.float64)
        else:
            self.data = rng.standard_normal(size=(32, 32)) + 100.0
        # The orchestrator reads ``obs.extdata`` (extfov-shaped); this
        # fake uses zero extfov margin so ``extdata`` is the same shape
        # as ``data``.
        self.extdata = self.data
        self._sensor_mask = np.ones(self.data.shape, bool)
        self.midtime = midtime
        # Stands in for ObsInst.camera, written to observation.camera.
        self.camera = 'NAC'
        # Stands in for ObsInst.shutter_mode, written to
        # observation.shutter_mode when the host exposes one.
        self.shutter_mode = shutter_mode

    def get_public_metadata(self) -> dict[str, Any]:
        """Describe the image the way a real snapshot does.

        The driver captions the summary PNG from this, so a stand-in that
        cannot answer is incomplete. It used to be production's job to survive
        that, which meant a real snapshot with a broken accessor shipped an
        unlabelled PNG and said nothing.
        """
        return {
            'image_name': 'N0000000000_1_CALIB.IMG',
            'filters': ('CL1',),
            'exposure_time': 0.05,
        }

    def extfov_data_sensor_mask(self) -> np.ndarray:
        """Report every pixel of the extended FOV as live sensor.

        Returns:
            A boolean array of ``extdata``'s shape, all True: this fake uses
            zero extfov margin, so no pixel falls outside the detector.
        """
        return self._sensor_mask


@pytest.fixture(autouse=True)
def _fakes_report_as_simulated(fakes_report_as_simulated: None) -> None:
    """Apply the shared simulated-instrument report to every test in this module."""


def _make_fake_obs_class(
    *,
    blank: bool = False,
    raise_on_load: BaseException | None = None,
    shutter_mode: str | None = None,
) -> type:
    """Build a fresh per-test ``obs_class`` shim with controllable behavior.

    Each call returns a brand-new class that closes over the requested
    behavior, so nothing is shared between tests and the shims are safe under
    parallel execution.

    Parameters:
        blank: True for a class whose images are all zero, so the driver takes
            its no-data path.
        raise_on_load: An exception ``from_file`` raises instead of returning a
            snapshot, so the driver takes its image-load-failure path; ``None``
            to load successfully.
        shutter_mode: The shutter mode every loaded snapshot reports, or
            ``None`` for a host whose labels carry no such field.

    Returns:
        A class exposing the one classmethod the driver calls, ``from_file``,
        which takes a path and returns a ``_FakeSnapshot``.
    """
    captured_blank = blank
    captured_raise = raise_on_load
    captured_shutter_mode = shutter_mode

    class _FakeObsClass:
        """The observation class the driver loads each image through."""

        @classmethod
        def from_file(cls, path: Any, **kwargs: Any) -> _FakeSnapshot:
            """Return the configured fake snapshot, ignoring the path.

            Parameters:
                path: The image path the driver resolved; unread, since the
                    snapshot's contents are fixed at class-construction time.
                kwargs: Any further loader options the driver passes; unread.

            Returns:
                A ``_FakeSnapshot`` built with this class's captured settings.

            Raises:
                BaseException: whatever ``raise_on_load`` supplied, when the
                    class was built to fail on load.
            """
            if captured_raise is not None:
                raise captured_raise
            return _FakeSnapshot(blank=captured_blank, shutter_mode=captured_shutter_mode)

    return _FakeObsClass


def _make_image_files(tmp_path: Path, *, camera: str | None = None) -> ImageFiles:
    """Build an ImageFiles batch with a single placeholder image."""
    img_path = tmp_path / 'fake_image.IMG'
    img_path.write_bytes(b'\x00')
    label_path = tmp_path / 'fake_image.LBL'
    label_path.write_bytes(b'\x00')
    return ImageFiles(
        image_files=[
            ImageFile(
                image_file_url=FCPath(str(img_path)),
                label_file_url=FCPath(str(label_path)),
                results_path_stub='fake_image',
                camera=camera,
            )
        ]
    )


def test_navigate_image_files_no_features_path(tmp_path: Path) -> None:
    """A clean image with no models selected yields a status='failed' result.

    ``nav_models='!*'`` selects no model, so nothing emits a feature and the
    orchestrator reports ``NO_FEATURES_EXTRACTED``.  Selecting none is what
    makes the premise true rather than merely stated: this fake observation
    is not one the real scene models can read, and left unselected they build
    against it and raise.  Until an unexpected exception became fatal that
    raise was absorbed and the run reported ``no_features_extracted`` anyway
    -- the same reason this test asserts, arrived at by an entirely different
    route, which is the confusion the fatal path exists to remove.
    """
    obs_class = _make_fake_obs_class()
    image_files = _make_image_files(tmp_path)
    success, metadata = navigate_image_files(
        obs_class,
        image_files,
        FCPath(str(tmp_path / 'results')),
        nav_models=['!*'],
        write_output_files=False,
    )
    assert success is False
    assert metadata['status'] == 'failed'
    assert metadata['confidence'] == 0.0
    # The fake observation class is not in the instrument registry.
    assert metadata['observation']['instrument'] == 'unknown'
    assert metadata['observation']['image_shape'] == [32, 32]
    assert metadata['timing']['elapsed_s'] >= 0.0
    assert metadata['timing']['start_iso8601'].endswith('Z')
    assert metadata['timing']['end_iso8601'].endswith('Z')
    assert 'navigation_result' in metadata
    nav_result = metadata['navigation_result']
    assert nav_result['status_reason'] == 'no_features_extracted'
    assert 'internal_error' not in nav_result


def test_navigate_image_files_writes_metadata(tmp_path: Path) -> None:
    """``write_output_files=True`` writes the metadata JSON to disk."""
    obs_class = _make_fake_obs_class()
    image_files = _make_image_files(tmp_path)
    results_root = tmp_path / 'results'
    success, _metadata = navigate_image_files(
        obs_class,
        image_files,
        FCPath(str(results_root)),
        write_output_files=True,
    )
    metadata_path = results_root / 'fake_image_metadata.json'
    assert metadata_path.exists()
    assert success is False  # no real-scene models registered


def test_navigate_image_files_records_camera(tmp_path: Path) -> None:
    """The observation's camera is written to observation.camera."""
    obs_class = _make_fake_obs_class()
    image_files = _make_image_files(tmp_path)
    _success, metadata = navigate_image_files(
        obs_class,
        image_files,
        FCPath(str(tmp_path / 'results')),
        write_output_files=False,
    )
    assert metadata['observation']['camera'] == 'NAC'


def test_navigate_image_files_records_shutter_mode(tmp_path: Path) -> None:
    """A host that exposes a shutter mode writes observation.shutter_mode."""
    obs_class = _make_fake_obs_class(shutter_mode='BOTSIM')
    image_files = _make_image_files(tmp_path)
    _success, metadata = navigate_image_files(
        obs_class,
        image_files,
        FCPath(str(tmp_path / 'results')),
        write_output_files=False,
    )
    assert metadata['observation']['shutter_mode'] == 'BOTSIM'


def test_navigate_image_files_omits_absent_shutter_mode(tmp_path: Path) -> None:
    """A host that exposes no shutter mode leaves the field out entirely."""
    obs_class = _make_fake_obs_class()
    image_files = _make_image_files(tmp_path)
    _success, metadata = navigate_image_files(
        obs_class,
        image_files,
        FCPath(str(tmp_path / 'results')),
        write_output_files=False,
    )
    assert 'shutter_mode' not in metadata['observation']


def test_navigate_image_files_load_error_records_index_camera(tmp_path: Path) -> None:
    """A SPICE-kernel failure keeps the camera the index supplied."""
    obs_class = _make_fake_obs_class(
        raise_on_load=RuntimeError('SPICE(NOFRAMECONNECT) -- insufficient information')
    )
    image_files = _make_image_files(tmp_path, camera='WAC')
    _success, metadata = navigate_image_files(
        obs_class,
        image_files,
        FCPath(str(tmp_path / 'results')),
        write_output_files=False,
    )
    assert metadata['status_error'] == 'missing_spice_data'
    assert metadata['observation']['camera'] == 'WAC'


def test_navigate_image_files_load_error_without_index_details(tmp_path: Path) -> None:
    """An image with no index row records no camera."""
    obs_class = _make_fake_obs_class(raise_on_load=OSError('boom'))
    image_files = _make_image_files(tmp_path, camera=None)
    _success, metadata = navigate_image_files(
        obs_class,
        image_files,
        FCPath(str(tmp_path / 'results')),
        write_output_files=False,
    )
    assert 'camera' not in metadata['observation']


def test_navigate_image_files_prefers_observation_camera(tmp_path: Path) -> None:
    """On a successful load the observation's own camera wins over the index."""
    obs_class = _make_fake_obs_class()
    image_files = _make_image_files(tmp_path, camera='WAC')
    _success, metadata = navigate_image_files(
        obs_class,
        image_files,
        FCPath(str(tmp_path / 'results')),
        write_output_files=False,
    )
    # The fake snapshot reports NAC; the index row here says WAC.
    assert metadata['observation']['camera'] == 'NAC'


def test_navigate_image_files_blank_image_yields_no_signal(tmp_path: Path) -> None:
    """A blank image yields ``status_reason == 'no_signal_in_image'``."""
    obs_class = _make_fake_obs_class(blank=True)
    image_files = _make_image_files(tmp_path)
    success, metadata = navigate_image_files(
        obs_class,
        image_files,
        FCPath(str(tmp_path / 'results')),
        write_output_files=False,
    )
    assert success is False
    assert metadata['navigation_result']['status_reason'] == 'no_signal_in_image'


def test_navigate_image_files_image_load_failure_records_status(tmp_path: Path) -> None:
    """An OSError during ``from_file`` records ``status='error'`` metadata."""
    obs_class = _make_fake_obs_class(raise_on_load=OSError('cannot read fixture image'))
    image_files = _make_image_files(tmp_path)
    success, metadata = navigate_image_files(
        obs_class,
        image_files,
        FCPath(str(tmp_path / 'results')),
        write_output_files=False,
    )
    assert success is False
    assert metadata['status'] == 'error'
    assert metadata['status_error'] == 'image_read_error'
    assert 'cannot read fixture image' in metadata['status_exception']
    assert metadata['observation']['instrument'] == 'unknown'
    # The image never loaded, so no shape is recorded; timing still is.
    assert 'image_shape' not in metadata['observation']
    assert metadata['timing']['elapsed_s'] >= 0.0


def test_navigate_image_files_spice_load_failure_records_missing_kernel(
    tmp_path: Path,
) -> None:
    """A SPICE-data error is classified as ``status_error='missing_spice_data'``."""
    obs_class = _make_fake_obs_class(
        raise_on_load=RuntimeError('SPICE(SPKINSUFFDATA) coverage missing')
    )
    image_files = _make_image_files(tmp_path)
    success, metadata = navigate_image_files(
        obs_class,
        image_files,
        FCPath(str(tmp_path / 'results')),
        write_output_files=False,
    )
    assert success is False
    assert metadata['status_error'] == 'missing_spice_data'


def test_navigate_image_files_writes_summary_png(tmp_path: Path) -> None:
    """``write_output_files=True`` writes a non-empty summary PNG to disk."""
    obs_class = _make_fake_obs_class()
    image_files = _make_image_files(tmp_path)
    results_root = tmp_path / 'results'
    navigate_image_files(
        obs_class,
        image_files,
        FCPath(str(results_root)),
        write_output_files=True,
    )
    png_path = results_root / 'fake_image_summary.png'
    assert png_path.exists()
    with Image.open(png_path) as img:
        assert img.mode == 'RGB'
        assert img.size == (32, 32)


# ---------------------------------------------------------------------------
# _grayscale_to_rgb_with_quantile_stretch
# ---------------------------------------------------------------------------


def test_grayscale_to_rgb_stretch_shape_and_dtype() -> None:
    """Output is uint8 RGB with the input's spatial shape."""
    image = np.linspace(0.0, 1.0, num=64, dtype=np.float64).reshape(8, 8)
    rgb = _grayscale_to_rgb_with_quantile_stretch(image)
    assert rgb.shape == (8, 8, 3)
    assert rgb.dtype == np.uint8
    np.testing.assert_array_equal(rgb[..., 0], rgb[..., 1])
    np.testing.assert_array_equal(rgb[..., 0], rgb[..., 2])


def test_grayscale_to_rgb_stretch_handles_constant_image() -> None:
    """A constant image renders as an all-zero RGB field.

    The 0.001 / 0.999 quantiles collapse to the constant value, so the
    stretch helper bumps ``white`` to ``nextafter(black, inf)``; the
    subsequent ``(value - black) / (white - black)`` evaluates to 0
    everywhere and ``(0 * 255).astype(uint8)`` is uniformly zero.
    """
    image = np.full((4, 4), 5.0, dtype=np.float64)
    rgb = _grayscale_to_rgb_with_quantile_stretch(image)
    expected = np.zeros((4, 4, 3), dtype=np.uint8)
    np.testing.assert_array_equal(rgb, expected)


def test_grayscale_to_rgb_stretch_treats_non_finite_as_zero() -> None:
    """NaN / inf samples are masked out before percentile selection."""
    image = np.array([[0.0, np.nan], [1.0, np.inf]], dtype=np.float64)
    rgb = _grayscale_to_rgb_with_quantile_stretch(image)
    assert rgb.shape == (2, 2, 3)
    # Non-finite pixels are remapped to zero before the stretch — so they
    # appear at or near the black end (0).
    assert int(rgb[0, 1, 0]) == 0


def test_grayscale_to_rgb_stretch_preserves_few_bright_pixels() -> None:
    """A handful of bright pixels keep their relative brightness ordering.

    A 1024 x 1024 dark-sky image with 8 bright stars at distinct
    intensities would clip every star to 255 under a fixed 0.001 / 0.999
    quantile stretch (0.1 % of 1 M = 1 048 pixels excluded as "white,"
    far more than the 8 brights).  The adaptive helper limits the clip
    count to half the bright-outlier count, so the brightest few
    saturate but the dimmer half keeps distinct gray values that
    preserve their brightness ordering.
    """
    rng = np.random.default_rng(seed=42)
    image = rng.normal(loc=0.0, scale=0.001, size=(1024, 1024)).astype(np.float64)
    bright_intensities = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    bright_positions = [
        (100, 100),
        (200, 200),
        (300, 300),
        (400, 400),
        (500, 500),
        (600, 600),
        (700, 700),
        (800, 800),
    ]
    for (v, u), intensity in zip(bright_positions, bright_intensities, strict=True):
        image[v, u] = intensity
    rgb = _grayscale_to_rgb_with_quantile_stretch(image)
    bright_grays = [int(rgb[v, u, 0]) for v, u in bright_positions]
    # The two dimmest stars keep distinct, sub-saturation gray values
    # (under the old fixed-0.999 stretch they would both clip to 255).
    assert bright_grays[0] < bright_grays[1] < 255
    # Brightness ordering monotonically rises through every bright pixel.
    assert bright_grays == sorted(bright_grays)


def test_grayscale_to_rgb_stretch_keeps_default_clip_when_many_brights() -> None:
    """With many bright pixels the original 0.1 % clip is unchanged.

    A bright disc filling a quarter of the FOV has tens of thousands of
    bright pixels — far more than the 0.1 % default clip count.  The
    adaptive helper falls back to the default behavior in that regime so
    a body-fills-FOV scene's visualization is unchanged from before this
    fix.
    """
    image = np.zeros((512, 512), dtype=np.float64)
    # Bright disc: 256 x 256 = 65536 bright pixels (~25 % of the image,
    # vastly more than the 0.1 % default clip count of 262).
    image[128:384, 128:384] = 100.0
    rgb = _grayscale_to_rgb_with_quantile_stretch(image)
    # Every disc pixel saturates to 255 because they are all the same
    # value and far above the 99.9 percentile of the full image.
    assert int(rgb[256, 256, 0]) == 255
    # Background is at the dark end.
    assert int(rgb[0, 0, 0]) == 0


# ---------------------------------------------------------------------------
# write_summary_png — direct fixture-driven exercise
# ---------------------------------------------------------------------------


class _FakeObsForRender:
    """Minimal stand-in for ObsSnapshot that satisfies the renderer + Annotations."""

    def __init__(self, *, image: np.ndarray) -> None:
        self.data = image
        self.extdata = image
        self._shape = image.shape
        self.data_shape_vu = image.shape
        self.extdata_shape_vu = image.shape
        self.extfov_margin_vu = (0, 0)
        self.extfov_margin_v = 0
        self.extfov_margin_u = 0

    def extract_offset_array(
        self, array: np.ndarray, _offset: tuple[float, float] | tuple[int, int] | None
    ) -> np.ndarray:
        # Zero extfov margin: the offset slice is the array itself.
        return array

    def get_public_metadata(self) -> dict[str, Any]:
        """Describe the image the way a real snapshot does.

        The renderer captions the summary PNG from this. A stand-in without it
        is an incomplete stand-in, and production carrying that incompleteness
        on its behalf is what let an unlabelled PNG ship as a finished product.
        """
        return {
            'image_name': 'N0000000000_1_CALIB.IMG',
            'filters': ('CL1', 'CL2'),
            'exposure_time': 0.12,
        }


def _make_render_result(
    *, annotations: Annotations, offset_px: tuple[float, float] | None = None
) -> NavResult:
    """Build a minimal NavResult that carries an annotation collection."""
    classifier = NavImageClassifierResult(
        image_class='clean',
        saturation_frac=0.0,
        missing_frac=0.0,
        noise_sigma=1.0,
        max_dn=10.0,
    )
    provenance = Provenance(
        spindoctor_version='0.0.0',
        image_et=0.0,
        pipeline_run_iso8601='2026-04-28T00:00:00Z',
    )
    if offset_px is not None:
        return NavResult.success(
            offset_px=offset_px,
            covariance_px2=np.eye(2),
            confidence=0.5,
            confidence_rank='medium',
            status_reason=NavStatusReason.OK,
            per_technique=[],
            feature_inventory=[],
            image_classifier=classifier,
            provenance=provenance,
            annotations=annotations,
        )
    return NavResult.failed(
        status_reason=NavStatusReason.NO_FEATURES_EXTRACTED,
        image_classifier=classifier,
        provenance=provenance,
        annotations=annotations,
    )


def test_write_summary_png_image_only_when_no_annotations(tmp_path: Path) -> None:
    """With an empty annotation collection the PNG carries the source image only."""
    rng = np.random.default_rng(seed=11)
    image = rng.standard_normal((20, 24)) + 50.0
    obs = _FakeObsForRender(image=image)
    result = _make_render_result(annotations=Annotations())
    png_path = FCPath(str(tmp_path / 'out.png'))
    write_summary_png(obs, result, png_path, _CapturingLogger())  # type: ignore[arg-type]
    with Image.open(BytesIO(png_path.read_bytes())) as img:
        assert img.mode == 'RGB'
        assert img.size == (24, 20)


def test_write_summary_png_composites_overlay(tmp_path: Path) -> None:
    """A red overlay is visible at the overlay's True pixels in the PNG."""
    rng = np.random.default_rng(seed=12)
    image = rng.standard_normal((16, 16)) + 10.0
    obs = _FakeObsForRender(image=image)
    overlay_mask = np.zeros((16, 16), dtype=bool)
    overlay_mask[5:8, 5:8] = True
    annotation = Annotation(
        obs=obs,  # type: ignore[arg-type]
        overlay=overlay_mask,
        overlay_color=(255, 0, 0),
    )
    annotations = Annotations()
    annotations.add_annotations(annotation)
    result = _make_render_result(annotations=annotations, offset_px=(0.0, 0.0))
    png_path = FCPath(str(tmp_path / 'overlay.png'))
    write_summary_png(obs, result, png_path, _CapturingLogger())  # type: ignore[arg-type]
    with Image.open(BytesIO(png_path.read_bytes())) as raw:
        img = np.asarray(raw)
    inside = img[6, 6]
    outside = img[0, 0]
    assert inside[0] == 255
    assert inside[1] == 0
    assert inside[2] == 0
    assert outside[0] == outside[1]
    assert outside[1] == outside[2]


class _CapturingLogger:
    """Minimal stand-in for pdslogger that records info messages."""

    def __init__(self) -> None:
        self.infos: list[str] = []

    def info(self, fmt: str, *args: Any) -> None:
        self.infos.append(fmt % args if args else fmt)


def test_navigate_image_files_rejects_multi_image_batch(tmp_path: Path) -> None:
    """A batch containing more than one image yields an error metadata block."""
    img_path = tmp_path / 'a.IMG'
    img_path.write_bytes(b'\x00')
    image_files = ImageFiles(
        image_files=[
            ImageFile(
                image_file_url=FCPath(str(img_path)),
                label_file_url=FCPath(str(img_path)),
                results_path_stub='a',
            ),
            ImageFile(
                image_file_url=FCPath(str(img_path)),
                label_file_url=FCPath(str(img_path)),
                results_path_stub='b',
            ),
        ]
    )
    obs_class = _make_fake_obs_class()
    success, metadata = navigate_image_files(
        obs_class,
        image_files,
        FCPath(str(tmp_path / 'results')),
        write_output_files=False,
    )
    assert success is False
    assert metadata['status_error'] == 'expected_one_image_per_batch'
    # Even the validation early-return records the instrument and timing.
    assert metadata['observation']['instrument'] == 'unknown'
    assert metadata['timing']['elapsed_s'] >= 0.0


def test_build_timing_section_formats_utc() -> None:
    """The timing section carries UTC ISO8601 strings and float seconds."""
    start = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)
    end = datetime(2026, 7, 11, 12, 0, 2, 500000, tzinfo=UTC)
    timing = build_timing_section(start, end)
    assert timing['start_iso8601'] == '2026-07-11T12:00:00Z'
    assert timing['end_iso8601'] == '2026-07-11T12:00:02.500000Z'
    assert timing['elapsed_s'] == 2.5


# ---------------------------------------------------------------------------
# _summary_metadata_from_obs_result — header assembly happy path
# ---------------------------------------------------------------------------


class _FakeObsForMetadata:
    """Obs stand-in exposing get_public_metadata for the header assembler."""

    def __init__(self, public: dict[str, Any]) -> None:
        self._public = public

    def get_public_metadata(self) -> dict[str, Any]:
        return self._public


class _FakeTechniqueResult:
    """Technique-result stand-in exposing only a technique name."""

    def __init__(self, technique_name: str) -> None:
        self.technique_name = technique_name


class _FakeNavResult:
    """NavResult stand-in exposing the fields the header assembler reads."""

    def __init__(
        self,
        *,
        status: str,
        per_technique: list[_FakeTechniqueResult],
        consensus_techniques: list[str],
        confidence: float,
        confidence_rank: str,
    ) -> None:
        self.status = status
        self.per_technique = per_technique
        self.consensus_techniques = consensus_techniques
        self.confidence = confidence
        self.confidence_rank = confidence_rank


def test_summary_metadata_joins_filters_with_plus() -> None:
    """Multiple filters are joined with ``+`` from the public metadata."""
    obs = _FakeObsForMetadata(
        {'image_name': 'N1.IMG', 'filters': ['CL1', 'IR3'], 'exposure_time': 2.0}
    )
    result = _FakeNavResult(
        status='success',
        per_technique=[_FakeTechniqueResult('RingEdgeNav')],
        consensus_techniques=['RingEdgeNav'],
        confidence=0.6,
        confidence_rank='medium',
    )
    meta = _summary_metadata_from_obs_result(obs, result)  # type: ignore[arg-type]
    assert meta.filter_name == 'CL1+IR3'


def test_summary_metadata_reads_exposure_as_float() -> None:
    """The exposure time is coerced to float from the public metadata."""
    obs = _FakeObsForMetadata({'image_name': 'N1.IMG', 'filters': [], 'exposure_time': 2})
    result = _FakeNavResult(
        status='success',
        per_technique=[_FakeTechniqueResult('StarRefineNav')],
        consensus_techniques=['StarRefineNav'],
        confidence=0.6,
        confidence_rank='medium',
    )
    meta = _summary_metadata_from_obs_result(obs, result)  # type: ignore[arg-type]
    assert meta.exposure_s == 2.0


def test_summary_metadata_tolerates_unparsable_exposure() -> None:
    """A non-numeric exposure leaves the field unknown instead of crashing."""
    obs = _FakeObsForMetadata({'image_name': 'N1.IMG', 'filters': [], 'exposure_time': 'bad'})
    result = _FakeNavResult(
        status='success',
        per_technique=[_FakeTechniqueResult('StarRefineNav')],
        consensus_techniques=['StarRefineNav'],
        confidence=0.6,
        confidence_rank='medium',
    )
    meta = _summary_metadata_from_obs_result(obs, result)  # type: ignore[arg-type]
    assert meta.exposure_s is None


def test_summary_metadata_uses_consensus_techniques() -> None:
    """Only the ensemble's consensus subset is reported, not every result."""
    obs = _FakeObsForMetadata({'image_name': 'N1.IMG', 'filters': ['CL1'], 'exposure_time': 1.0})
    result = _FakeNavResult(
        status='success',
        per_technique=[
            _FakeTechniqueResult('RingEdgeNav'),
            _FakeTechniqueResult('BodyLimbNav'),  # an outlier the ensemble rejected
        ],
        consensus_techniques=['RingEdgeNav'],
        confidence=0.7,
        confidence_rank='high',
    )
    meta = _summary_metadata_from_obs_result(obs, result)  # type: ignore[arg-type]
    assert meta.techniques == ('RingEdgeNav',)


def test_summary_metadata_falls_back_to_per_technique_when_no_consensus() -> None:
    """A success with no stamped consensus falls back to the per-technique set."""
    obs = _FakeObsForMetadata({'image_name': 'N1.IMG', 'filters': ['CL1'], 'exposure_time': 1.0})
    result = _FakeNavResult(
        status='success',
        per_technique=[_FakeTechniqueResult('BodyDiscCorrelateNav')],
        consensus_techniques=[],
        confidence=0.7,
        confidence_rank='high',
    )
    meta = _summary_metadata_from_obs_result(obs, result)  # type: ignore[arg-type]
    assert meta.techniques == ('BodyDiscCorrelateNav',)


def test_summary_metadata_failed_reports_no_techniques() -> None:
    """A failed nav reports no techniques regardless of what ran."""
    obs = _FakeObsForMetadata({'image_name': 'N1.IMG', 'filters': ['CL1'], 'exposure_time': 1.0})
    result = _FakeNavResult(
        status='failed',
        per_technique=[_FakeTechniqueResult('RingEdgeNav')],
        consensus_techniques=[],
        confidence=0.0,
        confidence_rank='failed',
    )
    meta = _summary_metadata_from_obs_result(obs, result)  # type: ignore[arg-type]
    assert meta.techniques == ()


def test_summary_metadata_fails_when_public_metadata_raises() -> None:
    """A public-metadata failure fails the image rather than captioning it blank.

    These are the labels the summary PNG carries, so absorbing the failure does
    not produce a PNG that says it is missing them: it produces an unlabelled
    one, which reads as a finished product rather than as a fault.
    """

    class _RaisingObs:
        abspath = Path('/holdings/N9.IMG')

        def get_public_metadata(self) -> dict[str, Any]:
            raise RuntimeError('no label')

    result = _FakeNavResult(
        status='failed',
        per_technique=[],
        consensus_techniques=[],
        confidence=0.0,
        confidence_rank='failed',
    )
    with pytest.raises(RuntimeError) as exc_info:
        _summary_metadata_from_obs_result(_RaisingObs(), result)  # type: ignore[arg-type]
    assert 'no label' in str(exc_info.value)
