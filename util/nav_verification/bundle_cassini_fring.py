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

Most of the bundle's wide angle camera C-matrices are not in the frame the standard
Cassini kernels define.  Their navigation offsets were measured with a geometry library
that applied a fixed rotation to every wide angle frame, and the matrices were written
without it, so what they say is about 0.8 of a wide angle pixel from where the camera
pointed.  The reader puts that rotation back before returning a wide angle boresight,
except on the frames whose offsets were measured after the library dropped the rotation,
and says which it did.  The narrow angle camera's matrices are read as published.
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
    'WAC_AS_PUBLISHED',
    'WAC_OFFSET_RESTORED',
    'WAC_PLATE_SCALE_URAD',
    'BundlePointing',
    'bundle_image_names',
    'plate_scale_urad',
    'read_bundle_pointing',
    'restore_old_oops_wac_offset',
    'suppl_path',
]

# Cassini ISS narrow angle camera, microradians per pixel: the nominal scale the
# project quotes, which is what the bundle's own products were built against.
# It is not identical to what the instrument kernel's distortion model implies at
# the field center, and the disagreement is a few parts in a thousand -- enough
# to matter to an absolute scale, not to the pixel differences measured here.
NAC_PLATE_SCALE_URAD = 5.9946

# Cassini ISS wide angle camera, microradians per pixel: the field-center scale of the
# instrument kernel's FOV as oops reads it, the mean of its two axes.
WAC_PLATE_SCALE_URAD = 59.7165

# Each frame width a Cassini ISS summing mode produces (full frame, 2x2, 4x4), against
# its summing factor: a summed frame's pixels are proportionally wider.
_SUMMING_BY_COLUMNS = {1024: 1, 512: 2, 256: 4}

DEFAULT_BUNDLE_DIR = FCPath(
    '/data/fring-bundles/cassini_iss_fring_mosaics_rsfrench2025/data_reproj_img'
)

# The wide angle camera frame of the geometry library the bundle was navigated with.
#
# From 2017 until 2025, Cassini ISS observations read through rms-oops put CASSINI_ISS_WAC
# on a Navigation frame of (-7, +4.4) narrow angle pixels -- "Rob's determination of WAC -
# NAC in units of NAC pixels" -- on top of CASSINI_ISS_WAC-NO_OFFSET, the library's own
# camera frame, which is the SPICE wide angle frame turned half way round the boresight.
# Commit 08066b302a (SETI/rms-oops#124, 2024-03-04) turned off the module-level default,
# but from_file still initialized through the class method, whose offset_wac stayed True
# until commit c725d6fc24 (2025-02-20, first released in v0.1.1).  One narrow angle pixel
# is arctan(tan(FOV_REF_ANGLE) / 512) with the 0.175 degree FOV_REF_ANGLE of
# cas_iss_v10.ti (its FOV_CROSS_ANGLE, which sizes the second shift, is the same).  The
# library this pipeline uses has no such offset.
#
# The bundle's navigation offsets were measured in the rotated frame, but its wide angle
# C-matrices were written from the frame without the rotation.  The rotation predicts
# that the bundle's boresight sits (-0.699, -0.440) wide angle pixels from ours along the
# camera axes.  On the 786 star-navigated wide angle frames of the full F ring run that
# agree within a pixel as published, the measured median is (-0.686, -0.433); with the
# rotation restored as this reader does, it is (+0.003, +0.001) over the 869 that then
# agree (#759).  The cause is filed against the bundle as rfrenchseti/f-ring#14.
_OLD_OOPS_WAC_SHIFT_NAC_PX = (-7.0, 4.4)
_NAC_FOV_REF_ANGLE_DEG = 0.175
_NAC_HALF_WIDTH_PX = 512.0

# The bundle's wide angle frames whose offsets were measured after the library dropped the
# rotation, so that their matrices are already in the standard frame and are read as
# published.  Which frame an offset was measured in is a matter of when it was written,
# and nothing in a supplementary file records that, so the frames are named.
#
# Every wide angle image in the bundle was in the f-ring repository's mosaic lists by
# 2023-07-03 except twelve, which f-ring commit 1895396 added on 2025-07-27 and which were
# therefore navigated without the rotation: the eleven 2007 frames of the rev 039 to 043
# FMONITOR observations and W1598925706 of 2008.  W1555949214, one of the twelve, is not
# listed: its manual offset misses the image stars by 0.84 px as published, and restoring
# the rotation brings it to 0.21 px, which is the same move as correcting that manual
# offset by hand.  Each of the other seven with usable image stars fits them as
# published, a median 0.1 to 0.3 px off, and 0.7 to 0.9 px off with the rotation
# restored.
#
# A manual offset is chosen on a reprojection made with the library installed at the
# time, so a manual offset entered after the change needs no restoration either.  All 43
# manually navigated wide angle frames were checked against the image stars, and ten
# outside the twelve are listed: the nine of ISS_134RI_SPKMVDFHP001_PRIME and W1726805460.
# Eight of the ten fit the stars as published.  W1656603760 is closer as published (0.59
# against 1.39 px restored), and W1656595136 is about half a pixel off either way (0.54
# against 0.49) and is kept with the rest of its observation.  The other 31 outside the
# twelve fit the stars better with the rotation restored, or have too few stars to say,
# and are restored like every other frame.
_AS_PUBLISHED_WAC_IMAGES = frozenset(
    {
        # Added to the mosaic lists in f-ring 1895396 (2025-07-27).
        'W1550952562',
        'W1550959642',
        'W1552853136',
        'W1555706573',
        'W1555707053',
        'W1555944144',
        'W1555945704',
        'W1555947654',
        'W1555951164',
        'W1555952724',
        'W1598925706',
        # Manual offsets evidently entered after the change.
        'W1656595136',
        'W1656595752',
        'W1656596368',
        'W1656598216',
        'W1656601296',
        'W1656601912',
        'W1656602528',
        'W1656603144',
        'W1656603760',
        'W1726805460',
    }
)

# What the reader says it did, in the words a comparison report prints.
WAC_OFFSET_RESTORED = 'WAC frame: the old oops WAC frame offset restored'
WAC_AS_PUBLISHED = (
    'WAC frame whose offset was measured without the old oops WAC offset: read as published'
)

# The rotation between the geometry library's camera frame and the SPICE camera frame: a
# half turn about the boresight, which is its own inverse.
_ROT180 = np.diag([-1.0, -1.0, 1.0])

_RA = re.compile(r'Navigated Boresight RA\s*=\s*([-\d.]+)')
_DEC = re.compile(r'Navigated Boresight Dec\s*=\s*([-\d.]+)')
_TYPE = re.compile(r'Navigation Type\s*=\s*(.+)')
_CMATRIX = re.compile(
    r'C-Matrix\s*=\s*' + r'\s+'.join([r'([-\d.eE+]+)'] * 9),
)
_PRODUCT = re.compile(r'Source Data Product ID\s*=\s*(\d+)([nw])_calib')


def plate_scale_urad(camera: str, columns: int) -> float:
    """The angle one pixel of a Cassini ISS frame spans.

    Parameters:
        camera: ``NAC`` or ``WAC``.
        columns: How many columns the frame has; a summed frame has fewer, and
            each of its pixels spans proportionally more sky.

    Returns:
        The plate scale in microradians per pixel.

    Raises:
        ValueError: If the camera is neither of the two, or the column count is
            not one a summing mode produces.
    """
    full = {'NAC': NAC_PLATE_SCALE_URAD, 'WAC': WAC_PLATE_SCALE_URAD}.get(camera)
    if full is None:
        raise ValueError(f'a Cassini ISS camera is NAC or WAC, not {camera!r}')
    summing = _SUMMING_BY_COLUMNS.get(columns)
    if summing is None:
        raise ValueError(f'no summing mode leaves a frame {columns} columns wide')
    return full * summing


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
        boresight: The camera boresight as a unit vector in J2000, in the frame the
            standard Cassini kernels define.
        navigation_type: What the bundle says the frame was navigated by, as
            written -- ``Stars``, ``Ring and/or Satellite Models``, ``Manual``,
            or a combination of them.
        from_cmatrix: Whether the boresight came from the quoted C-matrix.  A
            False here means it came from the rounded right ascension and
            declination instead, which is quantized between about a third of a
            NAC pixel and about 1.5 of one depending on the right ascension.
        note: What the reader did to the bundle's answer before returning it:
            ``WAC_OFFSET_RESTORED`` for a wide angle frame whose C-matrix was
            rotated into the standard frame, ``WAC_AS_PUBLISHED`` for a wide angle
            frame whose matrix is in that frame already, and None for a narrow angle
            frame, which is returned as published.
    """

    boresight: np.ndarray
    navigation_type: str
    from_cmatrix: bool
    note: str | None = None


def _axis_rotation(angle: float, axis: int) -> np.ndarray:
    """The rotation about one axis that the geometry library's Navigation frame composes.

    Parameters:
        angle: The rotation angle in radians.
        axis: The axis it turns about, 0, 1, or 2.

    Returns:
        The 3x3 matrix, with the library's sign convention: it rotates a vector given in
        the reference frame into the rotated frame.
    """
    first, second = (axis + 1) % 3, (axis + 2) % 3
    matrix = np.eye(3)
    matrix[first, first] = matrix[second, second] = math.cos(angle)
    matrix[first, second] = math.sin(angle)
    matrix[second, first] = -math.sin(angle)
    return matrix


def _old_oops_wac_restoration() -> np.ndarray:
    """The rotation that carries a bundle wide angle C-matrix into the standard frame.

    The old library's wide angle frame was a Navigation frame N on top of its own camera
    frame, which is the SPICE camera frame turned half way round the boresight.  A
    C-matrix C written from the frame without N becomes ``R180 @ N @ R180 @ C``.

    Returns:
        The 3x3 matrix to apply on the left of such a C-matrix.
    """
    nac_pixel = math.atan(math.tan(math.radians(_NAC_FOV_REF_ANGLE_DEG)) / _NAC_HALF_WIDTH_PX)
    # Navigation((a, b)) turns by a about the Y axis and then by b about the X axis.
    about_y, about_x = (shift * nac_pixel for shift in _OLD_OOPS_WAC_SHIFT_NAC_PX)
    navigation = _axis_rotation(about_x, 0) @ _axis_rotation(about_y, 1)
    restoration: np.ndarray = _ROT180 @ navigation @ _ROT180
    return restoration


_WAC_RESTORATION = _old_oops_wac_restoration()


def restore_old_oops_wac_offset(cmatrix: np.ndarray) -> np.ndarray:
    """Put back the wide angle frame offset a bundle C-matrix was written without.

    Parameters:
        cmatrix: A wide angle C-matrix as the bundle publishes it, rotating J2000 into
            the camera frame, as a 3x3 array.

    Returns:
        The same attitude in the frame the standard Cassini kernels define for the
        wide angle camera, as a new 3x3 array.  Its boresight lies about 0.7 wide angle
        pixels along the camera's first axis and 0.44 along its second from the
        published one.
    """
    restored: np.ndarray = _WAC_RESTORATION @ np.asarray(cmatrix, dtype=float).reshape(3, 3)
    return restored


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

    The camera is the one the file's source product ID names.  A wide angle frame's
    C-matrix has the old geometry library's frame offset restored, as
    ``restore_old_oops_wac_offset`` describes, unless the frame is one of the 21 whose
    navigation offsets were measured without that offset, when it is read as published.
    The offset is a rotation of the whole attitude, which the rounded pair does not
    carry, so a wide angle frame that needs it and quotes no usable matrix is no answer.
    A file that names no wide angle source product is read as a narrow angle frame.

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

    product = _PRODUCT.search(text)
    note = None
    if product is not None and product.group(2) == 'w':
        as_published = f'W{product.group(1)}' in _AS_PUBLISHED_WAC_IMAGES
        note = WAC_AS_PUBLISHED if as_published else WAC_OFFSET_RESTORED

    matrix = _CMATRIX.search(text)
    if matrix:
        elements = np.array([_number(matrix.group(i)) for i in range(1, 10)], dtype=float)
        cmatrix = elements.reshape(3, 3)
        # The restored boresight mixes all three rows, so a value that is not a number
        # anywhere in the matrix reaches it and is declined below.
        if note == WAC_OFFSET_RESTORED:
            cmatrix = restore_old_oops_wac_offset(cmatrix)
        row = cmatrix[2]
        norm = float(np.linalg.norm(row))
        if np.isfinite(row).all() and math.isfinite(norm) and norm > 0.0:
            return BundlePointing(row / norm, navigation_type, True, note)

    if note == WAC_OFFSET_RESTORED:
        return None
    ra, dec = _RA.search(text), _DEC.search(text)
    if not (ra and dec):
        return None
    ra_deg, dec_deg = _number(ra.group(1)), _number(dec.group(1))
    if not (math.isfinite(ra_deg) and math.isfinite(dec_deg)):
        return None
    return BundlePointing(vector_from_ra_dec(ra_deg, dec_deg), navigation_type, False, note)


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
    names = []
    # The listing is the only request: a directory that is not there lists nothing,
    # or raises on a store that treats a missing path as an error.
    try:
        for path in directory.glob('*_reproj_img_suppl.txt'):
            product = path.name.split('_')[0]
            if len(product) > 1 and product[:-1].isdigit():
                names.append(f'{product[-1].upper()}{product[:-1]}')
    except FileNotFoundError:
        return []
    return sorted(names)
