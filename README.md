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

SpinDoctor is a comprehensive navigation system designed for spacecraft imagery
processing. It provides tools to analyze images from various space missions
(Cassini, Voyager, Galileo, New Horizons) and determine precise positional
offsets by comparing observed images with theoretical models of celestial
bodies.

## Features

- **Multi-mission support**: Works with Cassini, Voyager, Galileo, and New
  Horizons imagery
- **Multiple navigation techniques**: Star-based, body-based, rings-based, and
  haze-symmetry (Titan) navigation
- **Automated offset calculation**: Determines precise pointing corrections
- **Visualization tools**: Creates annotated images with identified features
- **Configurable processing**: Customizable parameters for different scenarios
- **PDS4 bundle generation**: Creates PDS4-compliant bundles with labels,
  metadata, and browse products
- **Backplane generation**: Computes per-pixel geometry products (longitude,
  latitude, angles, etc.)
- **Run statistics**: Ingests navigation results into SQLite and generates
  reports on success rates, technique usage, offsets, and cross-technique
  agreement (`sd_results_index` / `sd_stats_report`)

## Installation

## Prerequisites

- Python 3.11 or higher
- SPICE toolkit and kernels for planetary data
- Dependencies listed in `requirements.txt`

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/SETI/rms-spindoctor.git
   cd rms-spindoctor
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up SPICE kernels:
   - Download the required SPICE kernels for your mission
   - Set the `SPICE_PATH` environment variable to point to your kernels directory:

     ```bash
     export SPICE_PATH=/path/to/your/spice/kernels
     ```

> **Note**: To fix mypy operability with editable pip installs:
>
> ```bash
> export SETUPTOOLS_ENABLE_FEATURES="legacy-editable"
> ```

## Quick Start

Process a single Cassini image using the installed CLI script:

```bash
sd_offset coiss N1234567890 \
  --pds3-holdings-root /path/to/pds3 \
  --nav-results-root /path/to/nav_results
```

Process all Voyager images within a single PDS3 volume:

```bash
sd_offset vgiss \
  --volumes VGISS_5101 \
  --pds3-holdings-root /path/to/pds3 \
  --nav-results-root /path/to/nav_results
```

Generate backplanes for processed images:

```bash
sd_backplanes coiss_saturn \
  --nav-results-root /path/to/nav_results \
  --backplane-results-root /path/to/backplane_results \
  --volumes COISS_2001
```

Generate PDS4 bundle files:

```bash
sd_create_bundle labels coiss_saturn \
  --nav-results-root /path/to/nav_results \
  --backplane-results-root /path/to/backplane_results \
  --bundle-results-root /path/to/bundle_results \
  --volumes COISS_2001
sd_create_bundle summary coiss_saturn --bundle-results-root /path/to/bundle_results
```

Check the bundle against the PDS4 schemas, its tables and itself:

```bash
sd_create_bundle check coiss_saturn --bundle-results-root /path/to/bundle_results
```

### Mosaicing

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

Display the resulting mosaic (or any individual reprojection file):

```bash
sd_mosaic_display_rings /path/to/mosaic_results/saturn_fring_2004_mosaic.fits
```

Reproject body images (e.g. Mimas):

```bash
sd_mosaic_body coiss_saturn \
  --volumes COISS_2001 \
  --pds3-holdings-root /path/to/pds3 \
  --nav-results-root /path/to/nav_results \
  --body-name MIMAS \
  --output-dir /path/to/mosaic_results \
  --prefix mimas_2004
```

See the
[Reprojection user guide](https://rms-spindoctor.readthedocs.io/en/latest/user_guide/user_guide_reprojection.html)
for full option references and more examples.

### Cloud Tasks variants

Each of the main batch drivers above has a queue-driven counterpart suffixed
with `_cloud_tasks`, which reads file lists from a
[cloud_tasks](https://github.com/SETI/rms-cloud-tasks) queue instead of
enumerating the dataset locally:

- `sd_offset_cloud_tasks` — navigation offsets
- `sd_backplanes_cloud_tasks` — backplane generation
- `sd_create_bundle_cloud_tasks` — PDS4 bundle labels pass
- `sd_mosaic_cloud_tasks` — mosaic reprojection pass; a single worker
  handles both ring and body tasks, with the mode carried in each task
  payload (mosaic combination is run separately via
  `sd_mosaic <mode> --skip-reproject`)

These workers accept only the environment flags needed to locate configuration
and results roots; the task payload carries the list of files plus any
per-task parameters. Each of `sd_offset`, `sd_backplanes`, and
`sd_mosaic_rings` / `sd_mosaic_body` can produce a ready-to-load task-queue
JSON file for its matching worker via `--output-cloud-tasks-file PATH`. The
per-feature user guides document the JSON schema each worker expects:

- `sd_offset_cloud_tasks`:
  [Navigation user guide](https://rms-spindoctor.readthedocs.io/en/latest/user_guide/user_guide_navigation.html)
- `sd_backplanes_cloud_tasks`:
  [Backplanes user guide](https://rms-spindoctor.readthedocs.io/en/latest/user_guide/user_guide_backplanes.html)
- `sd_mosaic_cloud_tasks`:
  [Reprojection user guide](https://rms-spindoctor.readthedocs.io/en/latest/user_guide/user_guide_reprojection.html)

## Documentation

Comprehensive documentation is available in the `docs` directory. To build
the documentation:

```bash
cd docs
make html
```

The built documentation will be available in `docs/_build/html`.

## Contributing

Information on contributing to this package can be found in the [Contributing
Guide](https://github.com/SETI/rms-spindoctor/blob/main/CONTRIBUTING.md).

## Licensing

This code is licensed under the [Apache License v2.0](https://github.com/SETI/rms-spindoctor/blob/main/LICENSE).
