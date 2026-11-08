"""End-to-end smoke of the Section 7 realism-match runner on real frames.

It reads holdings and resolves SPICE geometry, so it carries only the
``integration`` marker: the default selection runs ``sim_integration`` tests,
which need neither, and a real-data test there would run whenever both
environment variables happen to be set.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.integration.sim_realism import run_realism_match

pytestmark = pytest.mark.integration

# CI sets PDS3_HOLDINGS_DIR but not OOPS_RESOURCES.
_HAS_REAL_DATA = bool(os.environ.get('PDS3_HOLDINGS_DIR')) and bool(
    os.environ.get('OOPS_RESOURCES')
)


@pytest.mark.skipif(not _HAS_REAL_DATA, reason='PDS3_HOLDINGS_DIR or OOPS_RESOURCES unset')
def test_realism_smoke_lorri(tmp_path: Path) -> None:
    """End-to-end smoke on the smallest cohort, writing to a temp root.

    LORRI has two star frames, so this exercises the real-frame load, the
    feature FOMs, aggregation, support labeling, and both writers without
    the full-cohort runtime.
    """
    from tests.integration.sim_realism_report import write_figures, write_summary

    results = run_realism_match(instruments=['nhlorri'], skip_fom7=True)
    comparison = results.comparisons['nhlorri']
    assert len(comparison.records) == 2
    assert 'star_ee50' in comparison.real.samples
    assert 'star_ee50' in comparison.sim.samples
    assert comparison.fom_support['fom3_limb'] == 'unsupported'
    assert comparison.fom_support['fom4_ring'] == 'unsupported'
    figures = write_figures(results, figures_root=tmp_path / 'figures')
    assert any('nhlorri_psf' in p.name for p in figures)
    summary_path = write_summary(results, results_root=tmp_path / 'results')
    payload = json.loads(summary_path.read_text())
    inst = payload['instruments']['nhlorri']
    assert inst['n_frames'] == 2
    assert 'star_ee50' in inst['divergences']
    assert inst['divergences']['star_ee50']['w1_normalized'] is not None


@pytest.mark.skipif(not _HAS_REAL_DATA, reason='PDS3_HOLDINGS_DIR or OOPS_RESOURCES unset')
def test_realism_reads_the_downlink_table_from_a_calibrated_vicar_label() -> None:
    """A CALIB frame's VICAR label marks its TABLE downlink for the matched scene.

    The calibrated WAC cohort has four frames, two downlinked through the 8-bit
    table; only those two records ask for the ``sqrt_lut`` quantization.
    """
    results = run_realism_match(instruments=['coiss_calib_wac'], skip_fom7=True)
    records = results.comparisons['coiss_calib_wac'].records
    assert len(records) == 4
    assert [record.quantization for record in records].count('sqrt_lut') == 2
