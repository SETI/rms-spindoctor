"""Warn when a scene's published instrument config repeats its planted truth.

The boundary filter keeps a scene's truth-side blocks out of the navigator's
view, but ``instrument_config`` is published knowledge and crosses unfiltered.
So an author who copies a planted value into it, such as the truth-side read
noise into ``instrument_config.noise.read_noise_dn``, hands the navigator the
scene's realization instead of the instrument's published parameters, and the
navigator's SNR model adapts to the planted noise.  No error follows, only an
optimistic recovery.

:func:`check_truth_echo` compares the two sides and warns with
:class:`SimSceneTruthEchoWarning`, naming both sites, for each truth value that
the published config repeats exactly:

- ``noise.<key>`` (and ``noise.vidicon.<key>``) against
  ``instrument_config.noise.<key>``, which share their key names.
- ``optics.psf.sigma_v`` and ``optics.psf.sigma_u`` against
  ``instrument_config.star_psf_sigma``, the navigator's PSF sigma.

A coincidence can be deliberate, as when a scene renders at the published
parameters on purpose.  The scene then lists the truth paths in its
``published_equals_truth`` key, and those paths don't warn.  Each listed path
must be one that coincides, so a stale opt-out fails validation.
"""

import warnings
from collections.abc import Iterator, Mapping
from typing import Any

from spindoctor.sim.scene_schema import SimSceneValidationError

__all__ = [
    'SimSceneTruthEchoWarning',
    'check_truth_echo',
    'stale_truth_declarations',
    'truth_echoes',
]


class SimSceneTruthEchoWarning(UserWarning):
    """A scene's published ``instrument_config`` repeats one of its truth values."""


def _noise_echoes(
    truth: Mapping[str, Any], published: Mapping[str, Any], prefix: str
) -> Iterator[tuple[str, str]]:
    """Yield the ``(truth path, published path)`` pairs where two noise blocks agree."""
    for key, value in truth.items():
        if value is None or key not in published:
            continue
        other = published[key]
        if isinstance(value, Mapping):
            if isinstance(other, Mapping):
                yield from _noise_echoes(value, other, f'{prefix}.{key}')
        elif not isinstance(other, Mapping) and other == value:
            yield f'noise{prefix}.{key}', f'instrument_config.noise{prefix}.{key}'


def truth_echoes(sim_params: Mapping[str, Any]) -> list[tuple[str, str]]:
    """List each truth value that the scene's ``instrument_config`` repeats.

    Parameters:
        sim_params: A scene mapping whose block types have been validated.

    Returns:
        ``(truth path, published path)`` pairs, such as
        ``('noise.read_noise_dn', 'instrument_config.noise.read_noise_dn')``, in
        the order the noise block and then the PSF list them.  Empty when the
        scene doesn't carry an ``instrument_config``.
    """
    published = sim_params.get('instrument_config')
    if not isinstance(published, Mapping):
        return []
    echoes: list[tuple[str, str]] = []
    noise = sim_params.get('noise')
    published_noise = published.get('noise')
    if isinstance(noise, Mapping) and isinstance(published_noise, Mapping):
        echoes.extend(_noise_echoes(noise, published_noise, ''))
    optics = sim_params.get('optics')
    psf = optics.get('psf') if isinstance(optics, Mapping) else None
    sigma = published.get('star_psf_sigma')
    if isinstance(psf, Mapping) and sigma is not None:
        for key in ('sigma_v', 'sigma_u'):
            if psf.get(key) is not None and psf[key] == sigma:
                echoes.append((f'optics.psf.{key}', 'instrument_config.star_psf_sigma'))
    return echoes


def _declared_paths(sim_params: Mapping[str, Any], source: str) -> list[str]:
    """Return the scene's ``published_equals_truth`` list, empty when the key is unset.

    Raises:
        spindoctor.sim.scene_schema.SimSceneValidationError: If the value isn't a
            list of strings.
    """
    declared = sim_params.get('published_equals_truth')
    if declared is None:
        return []
    if not isinstance(declared, list) or not all(isinstance(path, str) for path in declared):
        raise SimSceneValidationError(
            f'{source}: published_equals_truth must be a list of truth paths such as '
            f"'noise.read_noise_dn'; got {declared!r}"
        )
    return declared


def stale_truth_declarations(sim_params: Mapping[str, Any], *, source: str) -> list[str]:
    """List the ``published_equals_truth`` paths whose two values don't coincide.

    Parameters:
        sim_params: A scene mapping whose block types have been validated.
        source: Label used in the error message.

    Returns:
        The listed truth paths that ``instrument_config`` doesn't repeat, sorted.
        Empty when every listed path coincides or the key is unset.

    Raises:
        spindoctor.sim.scene_schema.SimSceneValidationError: If
            ``published_equals_truth`` isn't a list of strings.
    """
    declared = _declared_paths(sim_params, source)
    return sorted(set(declared) - {truth for truth, _ in truth_echoes(sim_params)})


def _echo_consequence(truth: str) -> str:
    """Say what a published copy of ``truth`` does, for the echo warning."""
    # The navigator doesn't read the vidicon keys, so a vidicon echo only
    # mislabels the published config; every other compared key feeds it.
    if truth.startswith('noise.vidicon.'):
        return "the published config carries the scene's realization instead of the camera's"
    return 'the navigator reads the scene truth as published knowledge'


def check_truth_echo(sim_params: Mapping[str, Any], *, source: str) -> None:
    """Warn for each truth value that ``instrument_config`` repeats unannounced.

    A ``noise.vidicon`` echo's warning says the published config carries the
    scene's realization; every other echo's warning says the navigator reads the
    scene truth as published knowledge.

    Parameters:
        sim_params: A scene mapping whose block types have been validated.
        source: Label used in the warning and error messages.

    Raises:
        spindoctor.sim.scene_schema.SimSceneValidationError: If
            ``published_equals_truth`` isn't a list of strings, or lists a path
            whose values don't coincide.
    """
    declared = _declared_paths(sim_params, source)
    stale = stale_truth_declarations(sim_params, source=source)
    if stale:
        raise SimSceneValidationError(
            f"{source}: published_equals_truth lists {stale}, but instrument_config doesn't "
            f'repeat those truth values; remove them from the list'
        )
    for truth, published in truth_echoes(sim_params):
        if truth in declared:
            continue
        warnings.warn(
            f'{source}: {published} repeats the planted {truth}, so '
            f'{_echo_consequence(truth)}; list {truth!r} in '
            f'published_equals_truth if they are equal on purpose',
            SimSceneTruthEchoWarning,
            stacklevel=3,
        )
