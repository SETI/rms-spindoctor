"""Render every committed simulator scene and digest each render's exact bytes.

A change meant to leave every render as it was (a speedup, a restructuring) is
checked by digesting the renders before and after it on one machine and comparing
the two files.  The renders covered are every scene of the catalog, twice (as the
render-diff images render it, without its planted offset, and as the navigator
reads it, with the offset), every point of every sweep, the developer-guide gallery
scenes, and the seven render-budget scenes of ``tests/integration/test_sim_perf.py``.
Each digest covers the float image and every array and scalar of the truth
metadata, recursively, so a change in the last bit of any value changes it.

Run it from the root of the checkout whose renders are digested; it imports that
checkout's ``src`` and ``tests``.  Rendering everything takes about five minutes on
one core.

Usage::

    python util/render_identity/render_digests.py OUT.json [--only PREFIX]
        [--compare BASE.json]

``--only`` digests only the renders whose names start with ``PREFIX`` (``catalog/``,
``sweep/``, ``gallery/``, or ``perf/``), and exits with status 1 if no name does.
``--compare`` then lists every render whose digest differs from ``BASE.json`` and
every render of ``BASE.json`` within the prefix that this run did not make, and
exits with status 1 if there is either.  Floating-point results can differ between
machines and library builds, so compare only digests taken on one machine with one
environment.

``main_c6815cfc.json`` beside this script holds the digests of main at commit
c6815cfc, taken on the development machine (NumPy 2.5.1, SciPy 1.17.0, Python
3.12).  On that machine a checkout descended from that commit can compare with it
directly; anywhere else, or once main's renders have moved, digest the base commit
first.
"""

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from tests.integration import sim_doc_images  # noqa: E402
from tests.integration.sim_sweep import (  # noqa: E402
    build_sweep_params,
    iter_sweep_paths,
    load_sweep,
)

from spindoctor.sim.render.driver import clear_render_caches, render_scene  # noqa: E402
from spindoctor.sim.scene.io import iter_scene_paths, load_sim_scene  # noqa: E402


def _field(digest: Any, tag: bytes, payload: bytes) -> None:
    """Feed one tagged field, its length first, to a hash.

    Parameters:
        digest: The hash object.
        tag: One byte saying what the field holds.
        payload: The field's bytes.
    """
    digest.update(tag + len(payload).to_bytes(8, 'little') + payload)


def _feed(digest: Any, value: Any) -> None:
    """Feed a value's exact bytes, and its structure, to a hash.

    Every value goes in as a type tag followed by its length or item count, so the
    stream reads back one way only: ``(3, 131)`` and ``(31, 31)`` feed different bytes.

    Parameters:
        digest: The hash object.
        value: An array, a scalar, a mapping, a sequence, or a dataclass.
    """
    if isinstance(value, np.ndarray):
        _field(digest, b'A', str(value.dtype).encode())
        _field(digest, b'S', repr(value.shape).encode())
        _field(digest, b'B', np.ascontiguousarray(value).tobytes())
    elif isinstance(value, dict):
        _field(digest, b'D', str(len(value)).encode())
        for key in sorted(value, key=str):
            _field(digest, b'K', repr(key).encode())
            _feed(digest, value[key])
    elif isinstance(value, list | tuple):
        _field(digest, b'L' if isinstance(value, list) else b'T', str(len(value)).encode())
        for item in value:
            _feed(digest, item)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        _field(digest, b'C', type(value).__qualname__.encode())
        _feed(digest, {f.name: getattr(value, f.name) for f in dataclasses.fields(value)})
    elif isinstance(value, float):
        _field(digest, b'F', np.float64(value).tobytes())
    elif isinstance(value, np.floating | np.integer | np.bool_):
        _field(digest, b'N', str(value.dtype).encode())
        _field(digest, b'B', np.asarray(value).tobytes())
    elif hasattr(value, '__dict__'):
        _field(digest, b'O', type(value).__qualname__.encode())
        _feed(digest, vars(value))
    else:
        _field(digest, b'R', repr(value).encode())


def _render_digests(params: dict[str, Any], **kwargs: Any) -> list[str]:
    """Render a scene from cold caches and digest its image and its truth metadata.

    Parameters:
        params: The scene.
        **kwargs: Keyword arguments for the render.

    Returns:
        The image's digest and the metadata's, 16 hex digits each.
    """
    clear_render_caches()
    image, metadata = render_scene(params, **kwargs)
    image_digest = hashlib.sha256()
    _feed(image_digest, image)
    metadata_digest = hashlib.sha256()
    _feed(metadata_digest, metadata)
    return [image_digest.hexdigest()[:16], metadata_digest.hexdigest()[:16]]


def _jobs() -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Every render to digest: its name, its scene, and its render arguments.

    Returns:
        The renders, in a fixed order.
    """
    jobs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for path in iter_scene_paths(ROOT / 'tests/integration/sim_scenes'):
        params = load_sim_scene(path)
        name = f'catalog/{path.parent.name}/{path.stem}'
        jobs.append((name + '#display', params, {'ignore_offset': True}))
        jobs.append((name + '#nav', params, {}))
    for path in iter_sweep_paths(ROOT / 'tests/integration/sim_sweeps'):
        for value, params in build_sweep_params(load_sweep(path)):
            jobs.append((f'sweep/{path.stem}/{value!r}', params, {}))
    for name, params, _ in sim_doc_images._GUI_GALLERY:
        jobs.append((f'gallery/{name}', params, {}))
    spec = importlib.util.spec_from_file_location(
        'sim_perf', ROOT / 'tests/integration/test_sim_perf.py'
    )
    assert spec is not None
    assert spec.loader is not None
    perf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(perf)
    for label, builder, size in (
        ('star', perf._psf_detector_scene, 512),
        ('star', perf._psf_detector_scene, 1024),
        ('body', perf._body_psf_detector_scene, 512),
        ('body', perf._body_psf_detector_scene, 1024),
        ('haze', perf._haze_body_psf_detector_scene, 512),
        ('haze', perf._haze_body_psf_detector_scene, 1024),
        ('ring', perf._ring_psf_detector_scene, 1024),
    ):
        jobs.append((f'perf/{label}{size}', builder(size), {}))
    return jobs


def _compare(base: dict[str, Any], results: dict[str, list[str]], *, only: str) -> int:
    """Compare digests with a base file's and print every difference.

    Parameters:
        base: The base file's digests.
        results: The digests of this run.
        only: The prefix this run was limited to; base renders outside it are not
            expected in the run.

    Returns:
        The exit status: 1 when a render differs from the base or a base render the
        run selected is missing from it, else 0.
    """
    selected = {name for name in base if name.startswith(only)}
    missing = sorted(selected - set(results))
    common = sorted(selected & set(results))
    differ = [name for name in common if base[name] != results[name]]
    for name in differ:
        print(f'DIFFERS {name}')
    for name in missing:
        print(f'MISSING {name}')
    print(
        f'{len(common)} compared, {len(differ)} differ, {len(missing)} missing, '
        f'{len(set(results) - set(base))} new'
    )
    return 1 if differ or missing else 0


def main() -> int:
    """Digest the renders, write them, and compare them with a base file if asked.

    Returns:
        The exit status: 1 when no render matches ``--only``, a render fails, a
        compared render differs, or a selected base render is missing; else 0.
    """
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('out', type=Path, help='the JSON file to write the digests to')
    parser.add_argument('--only', default='', help='digest only names with this prefix')
    parser.add_argument('--compare', type=Path, help='a digest file to compare with')
    args = parser.parse_args()

    results: dict[str, list[str]] = {}
    start = time.time()
    for name, params, kwargs in _jobs():
        if not name.startswith(args.only):
            continue
        try:
            results[name] = _render_digests(params, **kwargs)
        except Exception as exc:  # A failing render is reported, not fatal.
            results[name] = ['error', f'{type(exc).__name__}: {exc}'[:200]]
    if not results:
        print(f'no render name starts with {args.only!r}')
        return 1
    args.out.write_text(json.dumps(results, indent=1, sort_keys=True) + '\n')
    print(f'{len(results)} renders in {time.time() - start:.0f} s')

    failed = [name for name, value in results.items() if value[0] == 'error']
    for name in failed:
        print(f'FAILED {name}: {results[name][1]}')
    if args.compare is None:
        return 1 if failed else 0
    status = _compare(json.loads(args.compare.read_text()), results, only=args.only)
    return 1 if status or failed else 0


if __name__ == '__main__':
    sys.exit(main())
