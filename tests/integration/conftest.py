"""Shared fixtures for the integration test layer (Part 0 §56).

A single ``pds3_holdings_dir`` fixture replaces every ad-hoc
``pytest.skip(...)`` so every integration test gates on the same
environment variable consistently.

``rotation_fitting_config`` is the packaged configuration with Galileo SSI
camera-rotation fitting turned on.  No packaged configuration fits a rotation,
so the tests that read a record of a navigation that fitted one hand this
configuration to their one image and leave the default configuration as it is.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from spindoctor.config import Config
from spindoctor.schema import Layer


@pytest.fixture(scope='session')
def pds3_holdings_dir() -> Path:
    """Return the PDS3 holdings root, or skip the test when unset."""
    p = os.environ.get('PDS3_HOLDINGS_DIR')
    if not p:
        pytest.skip('PDS3_HOLDINGS_DIR unset; integration tests skipped')
    return Path(p)


@pytest.fixture(scope='session')
def rotation_fitting_config() -> Config:
    """Return the packaged configuration with Galileo SSI rotation fitting turned on.

    No packaged configuration fits a camera rotation, so a navigation that fits one
    is produced by handing this configuration to that one image, through its
    ``extra_params``.  Every other setting is the packaged default, and the
    process-wide default configuration is left untouched.
    """
    override = Layer({'galileo_ssi': {'fit_camera_rotation': True}}, 'the rotation override')
    config = Config(overrides=[override])
    config.read_config()
    return config
