# SpinDoctor

<!-- pyml disable MD025 -->

[![GitHub release; latest by date](https://img.shields.io/github/v/release/SETI/rms-spindoctor)](https://github.com/SETI/rms-spindoctor/releases)
[![GitHub Release Date](https://img.shields.io/github/release-date/SETI/rms-spindoctor)](https://github.com/SETI/rms-spindoctor/releases)
[![Test Status](https://img.shields.io/github/actions/workflow/status/SETI/rms-spindoctor/run-tests.yml?branch=main)](https://github.com/SETI/rms-spindoctor/actions)
[![Documentation Status](https://readthedocs.org/projects/rms-spindoctor/badge/?version=latest)](https://rms-spindoctor.readthedocs.io/en/latest/?badge=latest)
[![Code coverage](https://img.shields.io/codecov/c/github/SETI/rms-spindoctor/main?logo=codecov)](https://codecov.io/gh/SETI/rms-spindoctor)

[![PyPI - Version](https://img.shields.io/pypi/v/rms-spindoctor)](https://pypi.org/project/rms-spindoctor)
[![PyPI - Format](https://img.shields.io/pypi/format/rms-spindoctor)](https://pypi.org/project/rms-spindoctor)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/rms-spindoctor)](https://pypi.org/project/rms-spindoctor)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/rms-spindoctor)](https://pypi.org/project/rms-spindoctor)

[![GitHub commits since latest release](https://img.shields.io/github/commits-since/SETI/rms-spindoctor/latest)](https://github.com/SETI/rms-spindoctor/commits/main/)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/m/SETI/rms-spindoctor)](https://github.com/SETI/rms-spindoctor/commits/main/)
[![GitHub last commit](https://img.shields.io/github/last-commit/SETI/rms-spindoctor)](https://github.com/SETI/rms-spindoctor/commits/main/)

[![Number of GitHub open issues](https://img.shields.io/github/issues-raw/SETI/rms-spindoctor)](https://github.com/SETI/rms-spindoctor/issues)
[![Number of GitHub closed issues](https://img.shields.io/github/issues-closed-raw/SETI/rms-spindoctor)](https://github.com/SETI/rms-spindoctor/issues)
[![Number of GitHub open pull requests](https://img.shields.io/github/issues-pr-raw/SETI/rms-spindoctor)](https://github.com/SETI/rms-spindoctor/pulls)
[![Number of GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed-raw/SETI/rms-spindoctor)](https://github.com/SETI/rms-spindoctor/pulls)

![GitHub License](https://img.shields.io/github/license/SETI/rms-spindoctor)
[![Number of GitHub stars](https://img.shields.io/github/stars/SETI/rms-spindoctor)](https://github.com/SETI/rms-spindoctor/stargazers)
![GitHub forks](https://img.shields.io/github/forks/SETI/rms-spindoctor)
<!-- start-after-point -->

# Introduction

SpinDoctor determines where a spacecraft camera was really pointing when it
took an image. It reads images from Cassini ISS, Voyager ISS, Galileo SSI, and
New Horizons LORRI, builds a model of the stars, rings, and bodies that should
appear in each one from the SPICE kernels, and measures how far the real image
has shifted from that model. From that measurement it records a corrected
pointing for the image.

The corrected pointing is what makes everything downstream possible. SpinDoctor
turns it into SPICE C kernels that any SPICE-based tool can use, into per-pixel
geometry backplanes, and into PDS4 archive bundles. It also reprojects and
mosaics navigated images onto ring and body grids.

SpinDoctor is for anyone who needs to know the geometry of an archived
planetary image more precisely than the mission's own reconstructed pointing
provides.

## Features

- **Multi-mission support**: Cassini ISS, Voyager ISS, Galileo SSI, and New
  Horizons LORRI imagery, read from VICAR files (Cassini, Voyager, Galileo) and
  FITS files (New Horizons)
- **Multiple navigation techniques**: star fields, body limbs and terminators,
  body discs, ring edges and annuli, and the solar symmetry of Titan's haze
- **Automated pointing corrections**: every applicable technique is run and
  their answers are reconciled into one correction with an uncertainty
- **Corrected-pointing C kernels**: writes SPICE C kernels carrying the
  corrected attitude, mirroring the original kernels the images were navigated
  against
- **Backplane generation**: per-pixel geometry products such as longitude,
  latitude, incidence, emission, phase, and ring radius
- **PDS4 bundle generation**: bundles with labels, collections, and browse
  products, checkable against the PDS4 schemas
- **Reprojection and mosaicing**: ring radius/longitude and body
  latitude/longitude reprojections, combined into mosaics, with interactive
  viewers
- **Run statistics**: a results index holding one row per navigated image, plus
  reports on success rates, technique usage, correction sizes, and how well the
  techniques agreed
- **Configurable processing**: every threshold and tolerance can be overridden
  from a configuration file, the environment, or the command line

## Installation

SpinDoctor requires Python 3.11 or higher.

Install the library and all command-line programs:

```bash
pip install rms-spindoctor
```

To install only the command-line programs, in their own isolated environment:

```bash
pipx install rms-spindoctor
```

Navigation also needs data that does not ship with the package: the SPICE
kernels for your mission, the image files themselves, and a star catalog.
Point SpinDoctor at them with environment variables:

```bash
export SPICE_PATH=/path/to/spice/kernels
export PDS3_HOLDINGS_DIR=/path/to/pds3/holdings
export UCAC4_PATH=/path/to/UCAC4
```

The
[Installation and Setup guide](https://rms-spindoctor.readthedocs.io/en/latest/user_guide/user_guide_installation.html)
lists every environment variable, the expected directory layouts, and where to
obtain each kind of data.

## Quick Start

Navigate a single Cassini image:

```bash
sd_offset coiss N1294562056 \
  --pds3-holdings-root /path/to/pds3 \
  --nav-results-root /path/to/nav_results
```

Navigate every Voyager image in one archive volume:

```bash
sd_offset vgiss \
  --volumes VGISS_5101 \
  --pds3-holdings-root /path/to/pds3 \
  --nav-results-root /path/to/nav_results
```

Every reported image gets a metadata document. Where navigation measured a
correction the document holds it, its uncertainty, and the corrected pointing;
where navigation failed it holds the status and the reason instead. An image the
navigator worked through
also gets a summary PNG showing the models drawn over the image; an image that
could not be read or that failed before navigation gets the metadata document
alone. See the
[navigation guide](https://rms-spindoctor.readthedocs.io/en/latest/user_guide/user_guide_navigation_running.html)
for `sd_offset`'s full option reference.

Write SPICE C kernels carrying the corrected pointing:

```bash
sd_create_ck coiss \
  --nav-results-root /path/to/nav_results \
  --kernel-dir /path/to/spice/kernels \
  --output-dir /path/to/ck_results
```

One corrected kernel is written for each original kernel the images were
navigated against, alongside a meta-kernel that furnishes the set and a CSV
report on every image considered. See the
[C kernel guide](https://rms-spindoctor.readthedocs.io/en/latest/user_guide/user_guide_ck_kernels.html).

Generate backplanes for navigated images:

```bash
sd_backplanes coiss_saturn \
  --nav-results-root /path/to/nav_results \
  --backplane-results-root /path/to/backplane_results \
  --volumes COISS_2001
```

See the
[backplanes guide](https://rms-spindoctor.readthedocs.io/en/latest/user_guide/user_guide_backplanes.html);
`sd_backplane_viewer` displays the result interactively.

Generate a PDS4 bundle, then check it against the PDS4 schemas, its own tables,
and itself:

```bash
sd_create_bundle labels coiss_saturn \
  --nav-results-root /path/to/nav_results \
  --backplane-results-root /path/to/backplane_results \
  --bundle-results-root /path/to/bundle_results \
  --volumes COISS_2001
sd_create_bundle summary coiss_saturn --bundle-results-root /path/to/bundle_results
sd_create_bundle check coiss_saturn --bundle-results-root /path/to/bundle_results
```

See the
[PDS4 bundle guide](https://rms-spindoctor.readthedocs.io/en/latest/user_guide/user_guide_pds4_bundle.html).

### Reprojection and mosaicing

Reproject a set of ring images and combine them into a mosaic:

```bash
sd_mosaic_rings coiss_saturn \
  --volumes COISS_2001 \
  --pds3-holdings-root /path/to/pds3 \
  --nav-results-root /path/to/nav_results \
  --planet SATURN \
  --radius-inner 139500 \
  --radius-outer 140220 \
  --output-dir /path/to/mosaic_results \
  --prefix saturn_fring_2004
```

Reproject body images onto a latitude/longitude grid:

```bash
sd_mosaic_body coiss_saturn \
  --volumes COISS_2001 \
  --pds3-holdings-root /path/to/pds3 \
  --nav-results-root /path/to/nav_results \
  --body-name MIMAS \
  --output-dir /path/to/mosaic_results \
  --prefix mimas_2004
```

Display a mosaic, or any individual reprojection file:

```bash
sd_mosaic_display_rings /path/to/mosaic_results/saturn_fring_2004_mosaic.fits
```

See the
[reprojection guide](https://rms-spindoctor.readthedocs.io/en/latest/user_guide/user_guide_reprojection.html)
for the full option reference for `sd_mosaic_rings`, `sd_mosaic_body`,
`sd_mosaic_display_rings`, and `sd_mosaic_display_body`, and more examples.

### Reviewing results

```bash
sd_consolidate_metadata coiss_saturn --nav-results-root /path/to/nav_results \
  --dest-dir /path/to/flat_results --copy-all
sd_results_index ingest --nav-results-root /path/to/nav_results \
  --results-index-db sqlite:///path/to/results_index.db
sd_stats_report --nav-results-root /path/to/nav_results \
  --output-dir /path/to/stats_report
```

`sd_consolidate_metadata` gathers each image's metadata document and summary
PNG into one flat directory
([guide](https://rms-spindoctor.readthedocs.io/en/latest/user_guide/user_guide_consolidate_metadata.html)).
`sd_results_index` builds the results index, a database holding one row per
navigated image, so later programs can read a whole mission's results in bulk
([guide](https://rms-spindoctor.readthedocs.io/en/latest/user_guide/user_guide_results_index.html)).
`sd_stats_report` summarizes how a run went
([guide](https://rms-spindoctor.readthedocs.io/en/latest/user_guide/user_guide_statistics.html)).

`sd_create_simulated_image` renders an image of a known geometry with stars,
bodies, and rings placed at a known offset, so you can compare what navigation
recovers against the truth that was planted
([guide](https://rms-spindoctor.readthedocs.io/en/latest/user_guide/user_guide_simulated_images.html)).

### Processing in the cloud

Cloud tasks is a work-queue package supplied by the Ring-Moon Systems Node,
distributed as [rms-cloud-tasks](https://rms-cloud-tasks.readthedocs.io). It
hands out batches of work to cloud compute instances and keeps track of which
batches have been done.

SpinDoctor's programs whose names end in `_cloud_tasks` are the workers that
the cloud task system starts on those instances, and you never run one
yourself: `sd_offset_cloud_tasks`, `sd_backplanes_cloud_tasks`,
`sd_create_bundle_cloud_tasks`, `sd_mosaic_cloud_tasks`, and
`sd_results_index_cloud_tasks`. `sd_offset`, `sd_backplanes`, and `sd_mosaic`
each write the task file their worker's queue is loaded from, via
`--output-cloud-tasks-file PATH`, and `sd_results_index divide` writes its own
via `--tasks-file PATH`.

Nothing limits how many images one run can process. For a mission with a very
large number of images you may still prefer to break the work into smaller
chunks, so that you can assess how each chunk turned out before starting the
next one. See the
[cloud tasks guide](https://rms-spindoctor.readthedocs.io/en/latest/user_guide/user_guide_cloud_tasks.html).

## Documentation

The full documentation is at
[rms-spindoctor.readthedocs.io](https://rms-spindoctor.readthedocs.io/en/latest/),
starting with the
[Quick Start](https://rms-spindoctor.readthedocs.io/en/latest/quick_start.html)
and the
[User Guide](https://rms-spindoctor.readthedocs.io/en/latest/user_guide/user_guide.html).
To build it locally, run `make html` in the `docs` directory; the result
appears in `docs/_build/html`.

## Contributing

Information on contributing to this package can be found in the [Contributing
Guide](https://github.com/SETI/rms-spindoctor/blob/main/CONTRIBUTING.md).

## Licensing

This code is licensed under the [Apache License v2.0](https://github.com/SETI/rms-spindoctor/blob/main/LICENSE).
