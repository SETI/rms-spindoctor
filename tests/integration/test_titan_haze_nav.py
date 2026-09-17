"""Real-frame integration tests for the Titan haze model and technique.

Everything the haze model needs from an observation -- the inventory
bounding box, the body-center resolution and phase, the incidence
backplane over an unclipped envelope box, the occlusion backplanes, and
the real YBSC / Tycho-2 star queries with their nominal-to-extfov
conversion -- only exists against real SPICE kernels and holdings.  These
tests exercise that path on one unoccluded Cassini Titan frame from the
validation cohort and then run the whole pipeline end to end on it.

The module is gated by ``pytestmark = pytest.mark.integration`` and skips
itself when ``PDS3_HOLDINGS_DIR`` is not set.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from filecache import FCPath

pytestmark = pytest.mark.integration

if 'PDS3_HOLDINGS_DIR' not in os.environ:
    pytest.skip(
        'PDS3_HOLDINGS_DIR is not set; skipping Titan haze integration tests',
        allow_module_level=True,
    )

from spindoctor.dataset.dataset import ImageFile, ImageFiles  # noqa: E402  (guarded import)
from spindoctor.feature.feature_type import NavFeatureType  # noqa: E402
from spindoctor.feature.geometry import TitanHazeGeometry  # noqa: E402
from spindoctor.nav_model.nav_model_titan import NavModelTitan  # noqa: E402
from spindoctor.navigate_image_files import navigate_image_files  # noqa: E402
from spindoctor.obs import ObsCassiniISS  # noqa: E402

_IMAGE_ID = 'W1822132529_1'
"""Unoccluded, well-resolved Cassini Titan frame from the cohort list."""

_IMAGE_REL = 'volumes/COISS_2xxx/COISS_2099/data/1822057149_1822284412/W1822132529_1.IMG'
"""Holdings-relative path of the frame."""


def _image_url() -> FCPath:
    """Resolve the frame's holdings-relative path against ``PDS3_HOLDINGS_DIR``."""
    root = os.environ['PDS3_HOLDINGS_DIR'].rstrip('/')
    return FCPath(f'{root}/{_IMAGE_REL}')


@pytest.fixture(scope='module')
def titan_geometry() -> TitanHazeGeometry:
    """Build the Titan model on the real frame and return its feature geometry."""
    obs = ObsCassiniISS.from_file(_image_url())
    models = NavModelTitan.instances_for_obs(obs)
    assert len(models) == 1
    model = models[0]
    model.create_model()
    feature = model.to_features(None)[0]  # type: ignore[arg-type]
    assert isinstance(feature.geometry, TitanHazeGeometry)
    return feature.geometry


@pytest.fixture(scope='module')
def titan_reliability() -> float:
    """Reliability the model scores for the real frame's haze feature."""
    obs = ObsCassiniISS.from_file(_image_url())
    model = NavModelTitan.instances_for_obs(obs)[0]
    model.create_model()
    return model.to_features(None)[0].reliability  # type: ignore[arg-type]


@pytest.fixture(scope='module')
def nav_metadata(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Navigate the real frame end to end and return its navigation metadata."""
    tmp_path: Path = tmp_path_factory.mktemp('titan_nav')
    url = _image_url()
    image_files = ImageFiles(
        image_files=[
            ImageFile(
                image_file_url=url,
                label_file_url=url,
                results_path_stub=_IMAGE_ID,
            )
        ]
    )
    _success, metadata = navigate_image_files(
        ObsCassiniISS,
        image_files,
        FCPath(str(tmp_path)),
        write_output_files=False,
    )
    result = metadata.get('navigation_result')
    assert isinstance(result, dict)
    return result


# ---------------------------------------------------------------------------
# Model emission on the real frame
# ---------------------------------------------------------------------------


def test_model_emits_one_haze_feature() -> None:
    """Titan in the extfov inventory produces exactly one TITAN_LIMB feature."""
    obs = ObsCassiniISS.from_file(_image_url())
    model = NavModelTitan.instances_for_obs(obs)[0]
    model.create_model()
    features = model.to_features(None)  # type: ignore[arg-type]
    assert [f.feature_type for f in features] == [NavFeatureType.TITAN_LIMB]


def test_real_frame_scores_usable_reliability(titan_reliability: float) -> None:
    """A well-resolved unoccluded Titan clears the 0.30 type gate comfortably."""
    assert titan_reliability > 0.9


def test_envelope_radius_exceeds_the_solid_radius(
    titan_geometry: TitanHazeGeometry,
) -> None:
    """The haze envelope sits above the solid body by the configured height."""
    assert titan_geometry.r_env_px > titan_geometry.r_solid_px


def test_image_scale_is_positive(titan_geometry: TitanHazeGeometry) -> None:
    """The body-center resolution evaluated to a usable image scale."""
    assert titan_geometry.km_per_px > 0.0


def test_symmetry_axis_is_defined_at_moderate_phase(
    titan_geometry: TitanHazeGeometry,
) -> None:
    """A frame well away from zero phase yields a defined sub-solar direction."""
    assert titan_geometry.axis_degenerate is False


def test_phase_angle_is_populated(titan_geometry: TitanHazeGeometry) -> None:
    """The phase angle at the body center is recorded on the geometry."""
    assert 0.0 < titan_geometry.phase_deg < 180.0


def test_filters_are_recorded(titan_geometry: TitanHazeGeometry) -> None:
    """The image's filter names travel with the feature for later analysis."""
    assert len(titan_geometry.filters) == 2


def test_contaminant_mask_matches_the_extended_frame(
    titan_geometry: TitanHazeGeometry,
) -> None:
    """When present the mask is shipped on the extended-frame grid.

    This is the assertion that exercises the real YBSC and Tycho-2 star
    queries and their nominal-to-extfov conversion: a mask of the wrong
    shape, or a query that raised, would show up here.
    """
    obs = ObsCassiniISS.from_file(_image_url())
    if titan_geometry.contaminant_mask is None:
        pytest.skip('no contaminant is predicted in this frame')
    assert titan_geometry.contaminant_mask.shape == obs.extdata_shape_vu


def test_unoccluded_frame_reports_no_occlusion(titan_geometry: TitanHazeGeometry) -> None:
    """No nearer body or ring covers Titan in this frame."""
    obs = ObsCassiniISS.from_file(_image_url())
    model = NavModelTitan.instances_for_obs(obs)[0]
    model.create_model()
    assert model.metadata['occluded_fraction'] == 0.0


# ---------------------------------------------------------------------------
# Technique execution on the real frame
# ---------------------------------------------------------------------------


def test_technique_runs_on_the_real_frame(nav_metadata: dict[str, object]) -> None:
    """``TitanHazeNav`` produces a per-technique result for the frame."""
    per_technique = nav_metadata['per_technique']
    assert isinstance(per_technique, list)
    names = [entry['technique_name'] for entry in per_technique]
    assert 'TitanHazeNav' in names


def test_technique_result_is_not_spurious(nav_metadata: dict[str, object]) -> None:
    """The fit converges rather than reporting a failed gate."""
    per_technique = nav_metadata['per_technique']
    assert isinstance(per_technique, list)
    entry = next(e for e in per_technique if e['technique_name'] == 'TitanHazeNav')
    assert entry['spurious'] is False


def test_technique_result_names_no_failed_gate(nav_metadata: dict[str, object]) -> None:
    """Every symmetry and arc gate passes on this frame."""
    per_technique = nav_metadata['per_technique']
    assert isinstance(per_technique, list)
    entry = next(e for e in per_technique if e['technique_name'] == 'TitanHazeNav')
    assert entry['diagnostics']['gate_failed'] is None


def test_fitted_haze_radius_is_recorded(nav_metadata: dict[str, object]) -> None:
    """The fitted arc radius lands in the physically plausible range for Titan.

    Titan's solid radius is about 2575 km and the configured envelope adds
    700 km, so a fit that converged on the haze top must sit between them.
    """
    per_technique = nav_metadata['per_technique']
    assert isinstance(per_technique, list)
    entry = next(e for e in per_technique if e['technique_name'] == 'TitanHazeNav')
    radius_km = entry['diagnostics']['fitted_haze_radius_km']
    assert 2500.0 < radius_km < 3400.0


def test_frame_navigates_successfully(nav_metadata: dict[str, object]) -> None:
    """The frame resolves to a committed offset."""
    assert nav_metadata['status'] == 'success'


def test_haze_feature_reaches_the_metadata(nav_metadata: dict[str, object]) -> None:
    """The emitted haze feature is recorded in the per-image feature inventory."""
    inventory = nav_metadata['feature_inventory']
    assert isinstance(inventory, list)
    ids = [entry['feature_id'] for entry in inventory]
    assert 'titan_limb:TITAN' in ids


def test_haze_feature_breakdown_reaches_the_metadata(
    nav_metadata: dict[str, object],
) -> None:
    """The reliability breakdown is serialized alongside the feature entry."""
    inventory = nav_metadata['feature_inventory']
    assert isinstance(inventory, list)
    entry = next(e for e in inventory if e['feature_id'] == 'titan_limb:TITAN')
    assert 'titan_envelope_diameter_px' in entry['reliability_reasons']
