"""The published Cassini F ring bundle, read as an independent pointing answer.

The F ring bundle carries, beside every reprojected product, a supplementary
text file recording the attitude that project navigated the frame to and how it
navigated it.  For these images that is the only per-frame pointing available
that this pipeline had no hand in, which makes it the one thing a run of ours
can be checked against.

It is an answer rather than truth, and its own file says which kind it is.  A
frame labeled ``Stars`` was fixed against a catalog and is worth believing to
well under a pixel; ``Ring and/or Satellite Models`` and ``Manual`` are weaker,
and a disagreement with one of those is a disagreement rather than an error.

The file quotes the boresight twice, as a right ascension and declination to six
significant figures and as a C-matrix to ten decimals.  Six significant figures
is not a fixed precision in pixels: it is three decimal places for a right
ascension under 100 degrees and two above it, so the quantization runs from
about a third of a NAC pixel to about 1.5 of one -- past the tolerance a
comparison would use.  The C-matrix is therefore read wherever it is there, and
in the published bundle it is there in every file, so the rounded pair is a
fallback that has not yet been needed rather than a path anything travels.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from filecache import FCPath
from util.nav_verification.boresight import vector_from_ra_dec

__all__ = [
    'DEFAULT_BUNDLE_DIR',
    'NAC_PLATE_SCALE_URAD',
    'BundlePointing',
    'bundle_image_names',
    'read_bundle_pointing',
    'suppl_path',
]

# Cassini ISS narrow angle camera, microradians per pixel: the nominal scale the
# project quotes, which is what the bundle's own products were built against.
# It is not identical to what the instrument kernel's distortion model implies at
# the field center, and the disagreement is a few parts in a thousand -- enough
# to matter to an absolute scale, not to the pixel differences measured here.
NAC_PLATE_SCALE_URAD = 5.9946

DEFAULT_BUNDLE_DIR = FCPath('/data/fring-bundles/pds4/data_reproj_img')

_RA = re.compile(r'Navigated Boresight RA\s*=\s*([-\d.]+)')
_DEC = re.compile(r'Navigated Boresight Dec\s*=\s*([-\d.]+)')
_TYPE = re.compile(r'Navigation Type\s*=\s*(.+)')
_CMATRIX = re.compile(
    r'C-Matrix\s*=\s*' + r'\s+'.join([r'([-\d.eE+]+)'] * 9),
)


def _number(text: str) -> float:
    """The number a matched field holds, or NaN when it does not hold one.

    The patterns match a run of characters a number is spelled with rather than
    a number, so what they capture can be something like ``1.2.3``, and the
    exponent form they accept can name a value too large to be one.

    Parameters:
        text: The text the pattern captured.

    Returns:
        The value, or NaN when the text does not parse as a number.
    """
    try:
        return float(text)
    except ValueError:
        return math.nan


@dataclass(frozen=True)
class BundlePointing:
    """What the bundle says about one frame's pointing.

    Parameters:
        boresight: The camera boresight as a unit vector in J2000.
        navigation_type: What the bundle says the frame was navigated by, as
            written -- ``Stars``, ``Ring and/or Satellite Models``, ``Manual``,
            or a combination of them.
        from_cmatrix: Whether the boresight came from the quoted C-matrix.  A
            False here means it came from the rounded right ascension and
            declination instead, which is quantized between about a third of a
            NAC pixel and about 1.5 of one depending on the right ascension.
    """

    boresight: np.ndarray
    navigation_type: str
    from_cmatrix: bool


def suppl_path(
    image_name: str,
    *,
    observation_id: str,
    bundle_dir: str | Path | FCPath = DEFAULT_BUNDLE_DIR,
) -> FCPath:
    """Where the bundle keeps one frame's supplementary file.

    The bundle names a product by its image number with the camera letter moved
    to the end and lowercased, under a directory named for the observation, so
    ``N1492052683`` of ``ISS_006RI_LPHRLFMOV001_PRIME`` is
    ``iss_006ri_lphrlfmov001_prime/1492052683n_reproj_img_suppl.txt``.

    Parameters:
        image_name: The image, with or without its version suffix.
        observation_id: The observation the frame belongs to.
        bundle_dir: The bundle's reprojected-image collection.

    Returns:
        The path the file would have, whether or not it is there.

    Raises:
        ValueError: If the image name does not begin with a camera letter
            followed by its number.
    """
    bare = image_name.split('_')[0]
    if len(bare) < 2 or not bare[0].isalpha() or not bare[1:].isdigit():
        raise ValueError(f'{image_name!r} does not name a Cassini ISS image')
    return (
        FCPath(bundle_dir)
        / observation_id.lower()
        / f'{bare[1:]}{bare[0].lower()}_reproj_img_suppl.txt'
    )


def read_bundle_pointing(path: str | Path | FCPath) -> BundlePointing | None:
    """Read one supplementary file.

    The bundle is another project's product, so a field of it can hold
    something that is not a finite number.  A value like that is declined
    rather than carried into a boresight, where it would spread a NaN through
    every number measured from it: a matrix that holds one is passed over for
    the rounded pair, and a pair that holds one is no answer.

    Parameters:
        path: The supplementary file.

    Returns:
        What the file says about the frame, or None if it records no boresight
        this can use.

    Raises:
        FileNotFoundError: If there is no such file.
        OSError: If the file is there but cannot be read.
    """
    text = FCPath(path).read_text(errors='replace')
    nav_type = _TYPE.search(text)
    navigation_type = nav_type.group(1).strip() if nav_type else 'unstated'

    matrix = _CMATRIX.search(text)
    if matrix:
        row = np.array([_number(matrix.group(i)) for i in (7, 8, 9)], dtype=float)
        norm = float(np.linalg.norm(row))
        if np.isfinite(row).all() and math.isfinite(norm) and norm > 0.0:
            return BundlePointing(row / norm, navigation_type, True)

    ra, dec = _RA.search(text), _DEC.search(text)
    if not (ra and dec):
        return None
    ra_deg, dec_deg = _number(ra.group(1)), _number(dec.group(1))
    if not (math.isfinite(ra_deg) and math.isfinite(dec_deg)):
        return None
    return BundlePointing(vector_from_ra_dec(ra_deg, dec_deg), navigation_type, False)


def bundle_image_names(
    *, observation_id: str, bundle_dir: str | Path | FCPath = DEFAULT_BUNDLE_DIR
) -> list[str]:
    """Every image the bundle holds a frame for, in one observation.

    A comparison needs this to tell a frame the pass navigated badly from one it
    never attempted.  Counting only the records a pass wrote would let a run
    that skipped a hundred frames report a better rate for having skipped them.

    Parameters:
        observation_id: The observation.
        bundle_dir: The bundle's reprojected-image collection.

    Returns:
        The image names, with the camera letter restored to the front and
        upper-cased, sorted.  An observation the bundle does not hold is an
        empty list rather than an error, since a caller comparing a pass the
        bundle never covered is asking a reasonable question.
    """
    directory = FCPath(bundle_dir) / observation_id.lower()
    if not directory.is_dir():
        return []
    names = []
    for path in directory.glob('*_reproj_img_suppl.txt'):
        product = path.name.split('_')[0]
        if len(product) > 1 and product[:-1].isdigit():
            names.append(f'{product[-1].upper()}{product[:-1]}')
    return sorted(names)
