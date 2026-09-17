"""Campaign driver for the star-field FOV distortion and twist analysis.

Runs one or more instrument-and-camera cohorts (``configs/*.yaml``), measuring
every listed frame in a process pool, aggregating the per-frame twists and
residual distortion into an instrument summary, and writing a per-frame CSV, a
JSON summary, and figures.

Run after ``source /seti/newnav/setup.sh``, from the repository root::

    python util/fov_distortion/run.py util/fov_distortion/configs/coiss_nac.yaml
    python util/fov_distortion/run.py util/fov_distortion/configs/*.yaml --workers 8

Artifacts default to ``_work/fov_distortion/``.  Pass ``--report-figures`` to
also write the instrument twist and radial figures plus one representative
per-cohort sample figure into ``docs/fov_distortion_report/_figures/`` for the
documentation report.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'src'))

from util.fov_distortion.config import AnalysisConfig, load_config  # noqa: E402
from util.fov_distortion.measure import (  # noqa: E402
    FrameMeasurement,
    MeasureParams,
    measure_frame,
)
from util.fov_distortion.plots import (  # noqa: E402
    plot_frame_decomposition,
    plot_instrument_distortion_map,
    plot_instrument_radial,
    plot_instrument_twist,
)
from util.fov_distortion.results import InstrumentSummary, summarize_instrument  # noqa: E402

DEFAULT_OUT = REPO / '_work' / 'fov_distortion'
REPORT_FIGURES = REPO / 'docs' / 'fov_distortion_report' / '_figures'


def _init_worker() -> None:
    """Silence the navigation loggers and pin BLAS threads in each worker."""
    import os

    for var in (
        'OMP_NUM_THREADS',
        'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS',
        'NUMEXPR_NUM_THREADS',
    ):
        os.environ[var] = '1'
    import pdslogger

    from spindoctor.config import IMAGE_LOGGER, MAIN_LOGGER

    for logger in (IMAGE_LOGGER, MAIN_LOGGER):
        logger.remove_all_handlers()
        logger.add_handler(pdslogger.NULL_HANDLER)


def _measure_one(job: tuple[str, str, MeasureParams]) -> FrameMeasurement:
    """Pool worker: measure one frame."""
    url, inst_id, params = job
    return measure_frame(url, inst_id, params=params)


def _frame_row(frame: FrameMeasurement) -> dict[str, object]:
    """Flatten one frame measurement into a CSV row."""
    row: dict[str, object] = {
        'image_name': frame.image_name,
        'status': frame.status,
        'n_stars': len(frame.stars),
        'offset_v': '' if frame.offset_vu is None else round(frame.offset_vu[0], 4),
        'offset_u': '' if frame.offset_vu is None else round(frame.offset_vu[1], 4),
    }
    decomp = frame.decomposition
    if decomp is None:
        row.update(
            {
                'twist_deg': '',
                'sigma_twist_deg': '',
                'rms_raw_px': '',
                'rms_after_twist_px': '',
                'rms_after_radial_px': '',
                'radial_k1': '',
                'radial_k2': '',
                'rms_radial_px': '',
                'rms_nonradial_px': '',
            }
        )
        return row
    k_sim = decomp.radial.k_sim
    row.update(
        {
            'twist_deg': round(decomp.twist.rotation_deg, 5),
            'sigma_twist_deg': round(decomp.twist.sigma_rotation_deg, 5),
            'rms_raw_px': round(decomp.rms_raw_px, 4),
            'rms_after_twist_px': round(decomp.rms_after_twist_px, 4),
            'rms_after_radial_px': round(decomp.rms_after_radial_px, 4),
            'radial_k1': f'{k_sim[0]:.3e}',
            'radial_k2': f'{k_sim[1]:.3e}' if len(k_sim) > 1 else '',
            'rms_radial_px': round(decomp.radial.rms_radial_px, 4),
            'rms_nonradial_px': round(decomp.radial.rms_nonradial_px, 4),
        }
    )
    return row


def _summary_dict(summary: InstrumentSummary) -> dict[str, object]:
    """Serialize an instrument summary to a JSON-friendly dict."""
    out: dict[str, object] = {
        'inst_id': summary.inst_id,
        'label': summary.label,
        'n_frames_total': summary.n_frames_total,
        'n_frames_ok': summary.n_frames_ok,
        'median_floor_px': summary.median_floor_px,
    }
    if summary.consistency is not None:
        out['consistency'] = asdict(summary.consistency)
    if summary.recommendation is not None:
        out['recommendation'] = asdict(summary.recommendation)
    if summary.pooled_radial is not None:
        model = summary.pooled_radial.model
        out['aggregate_radial'] = {
            'powers': list(model.powers),
            'k_sim': list(model.k_sim),
            'coeffs_px': list(model.coeffs_px),
            'rho_ref_px': model.rho_ref_px,
            'rms_radial_px': model.rms_radial_px,
            'rms_nonradial_px': model.rms_nonradial_px,
        }
    return out


def _representative_frame(summary: InstrumentSummary) -> FrameMeasurement | None:
    """Pick the ok frame with the most stars for the sample figure."""
    if not summary.ok_frames:
        return None
    return max(summary.ok_frames, key=lambda f: len(f.stars))


def _write_sample_figure(frame: FrameMeasurement, params: MeasureParams, path: Path) -> None:
    """Reload the frame's pixels and draw the per-frame decomposition figure."""
    from filecache import FCPath

    from spindoctor.obs import inst_name_to_obs_class
    from spindoctor.obs.obs_snapshot_inst import ObsSnapshotInst

    obs = cast(
        ObsSnapshotInst,
        inst_name_to_obs_class(frame.inst_id).from_file(
            FCPath(frame.url).expandvars(), fast_distortion=params.fast_distortion
        ),
    )
    image = np.nan_to_num(np.asarray(obs.data, dtype=np.float64))
    plot_frame_decomposition(frame, image, str(path))


def run_cohort(
    config: AnalysisConfig,
    out_dir: Path,
    *,
    workers: int,
    limit: int | None,
    report_figures: bool,
    cohort_name: str,
) -> InstrumentSummary:
    """Measure and summarize one cohort, writing all artifacts.

    Parameters:
        config: The cohort configuration.
        out_dir: Directory for per-cohort artifacts.
        workers: Process-pool size.
        limit: Optional cap on the number of frames (for quick runs).
        report_figures: Also write figures into the docs report figure dir.
        cohort_name: Short name for output files (the config stem).

    Returns:
        The :class:`InstrumentSummary` for the cohort.
    """
    urls = config.images[:limit] if limit is not None else config.images
    jobs = [(url, config.inst_id, config.params) for url in urls]

    print(
        f'[{cohort_name}] {config.label}: measuring {len(jobs)} frame(s) on {workers} worker(s)',
        flush=True,
    )
    frames: list[FrameMeasurement] = []
    if workers == 1:
        for i, job in enumerate(jobs, 1):
            frames.append(_measure_one(job))
            if i % 10 == 0 or i == len(jobs):
                _ok = sum(1 for f in frames if f.decomposition is not None)
                print(f'[{cohort_name}]   {i}/{len(jobs)} done ({_ok} ok)', flush=True)
    else:
        with mp.Pool(workers, initializer=_init_worker) as pool:
            for i, frame in enumerate(pool.imap_unordered(_measure_one, jobs), 1):
                frames.append(frame)
                if i % 10 == 0 or i == len(jobs):
                    _ok = sum(1 for f in frames if f.decomposition is not None)
                    print(f'[{cohort_name}]   {i}/{len(jobs)} done ({_ok} ok)', flush=True)

    summary = summarize_instrument(
        config.inst_id,
        config.label,
        frames,
        radial_powers=config.params.radial_powers,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_frames_csv(frames, out_dir / f'{cohort_name}_frames.csv')
    (out_dir / f'{cohort_name}_summary.json').write_text(
        json.dumps(_summary_dict(summary), indent=2)
    )

    fig_dir = out_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    _write_cohort_figures(summary, config, fig_dir, cohort_name)
    if report_figures:
        REPORT_FIGURES.mkdir(parents=True, exist_ok=True)
        _write_cohort_figures(summary, config, REPORT_FIGURES, cohort_name)

    _print_verdict(summary)
    return summary


def _write_frames_csv(frames: list[FrameMeasurement], path: Path) -> None:
    """Write one CSV row per frame."""
    rows = [_frame_row(f) for f in frames]
    fieldnames = list(rows[0].keys()) if rows else ['image_name', 'status']
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_cohort_figures(
    summary: InstrumentSummary, config: AnalysisConfig, fig_dir: Path, cohort_name: str
) -> None:
    """Write the twist, radial, and sample figures for a cohort."""
    if summary.consistency is not None:
        plot_instrument_twist(summary, str(fig_dir / f'{cohort_name}_twist.png'))
    if summary.pooled_radial is not None:
        plot_instrument_radial(summary, str(fig_dir / f'{cohort_name}_radial.png'))
        plot_instrument_distortion_map(summary, str(fig_dir / f'{cohort_name}_distortion_map.png'))
    rep = _representative_frame(summary)
    if rep is not None:
        _write_sample_figure(rep, config.params, fig_dir / f'{cohort_name}_sample.png')


def _print_verdict(summary: InstrumentSummary) -> None:
    """Print the one-line verdict for a cohort."""
    if summary.consistency is None or summary.recommendation is None:
        print(f'[{summary.label}] no usable frames ({summary.n_frames_total} attempted)')
        return
    con = summary.consistency
    print(
        f'[{summary.label}] {summary.n_frames_ok}/{summary.n_frames_total} frames | '
        f'twist {con.weighted_mean_deg:+.4f} +/- {con.sigma_mean_deg:.4f} deg | '
        f'reduced chi-square {con.reduced_chi_square:.1f} | '
        f'{"CONSISTENT" if con.consistent else "INCONSISTENT"}'
    )
    print(f'    -> {summary.recommendation.rationale}')


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Parameters:
        argv: Argument vector; ``sys.argv[1:]`` if omitted.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('configs', nargs='+', type=Path, help='Cohort YAML config file(s).')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT, help='Artifact root directory.')
    parser.add_argument('--workers', type=int, default=4, help='Process-pool size.')
    parser.add_argument(
        '--limit', type=int, default=None, help='Cap frames per cohort (quick run).'
    )
    parser.add_argument(
        '--report-figures',
        action='store_true',
        help='Also write figures into docs/fov_distortion_report/_figures/.',
    )
    args = parser.parse_args(argv)

    summaries: list[dict[str, object]] = []
    for config_path in args.configs:
        config = load_config(config_path)
        cohort_name = Path(config_path).stem
        summary = run_cohort(
            config,
            args.out / cohort_name,
            workers=args.workers,
            limit=args.limit,
            report_figures=args.report_figures,
            cohort_name=cohort_name,
        )
        summaries.append({'cohort': cohort_name, **_summary_dict(summary)})

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'campaign_summary.json').write_text(json.dumps(summaries, indent=2))
    print(f'\nWrote campaign summary for {len(summaries)} cohort(s) to {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
