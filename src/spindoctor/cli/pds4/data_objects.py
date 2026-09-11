"""The data objects a backplane FITS holds, as its PDS4 data label describes them.

A backplane FITS is an empty primary HDU followed by one image HDU per array: the
body identity map, when some body claimed a pixel, and then one float plane per
backplane that has a valid pixel.  A data label describes the header of every HDU
as a ``Header`` and every image past the primary as an ``Array_2D_Image``, each at
the byte offset the file holds it at.  :func:`describe_backplane_fits` reads those
offsets, and everything else the label states about an array, from the file itself
rather than from the configuration that asked for it, so that the label describes
the file in the bundle: a plane the writer dropped is not described, and one it
wrote is described as it was written.

What the backplane writer does not write is refused rather than described.  A label
that described such a file would be describing it wrongly -- an integer declared at
the wrong width, a scaled array declared as its raw values, an array past the end of
a truncated file -- and a label that is wrong about its file is worse than no label.
"""

import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyUserWarning
from filecache import FCPath

from spindoctor.cli.backplanes.writer import BODY_ID_MAP_HDU_NAME
from spindoctor.config import Config

FITS_DATA_TYPES: dict[int, str] = {-32: 'IEEE754MSBSingle', 32: 'SignedMSB4'}
"""The PDS4 ``data_type`` of an array's elements, by the ``BITPIX`` of its HDU.

The two element types the backplane writer uses: 32-bit floats for every plane and
32-bit signed integers for the body identity map.  Both are the most significant byte
first, because FITS stores every value big-endian.
"""

BODY_ID_MAP_DESCRIPTION = (
    'The body that claimed each pixel: 0 where no body claimed it, and otherwise the '
    'NAIF ID of the body that did.'
)
"""What the label says the body identity map's values are.

The map declares no missing value: its 0 marks a pixel no body claimed, which is a
fact about the pixel rather than a missing measurement, and no NAIF ID is 0.
"""

_LOCAL_IDENTIFIER = re.compile(r'[a-z_][a-z0-9_.-]*')
"""What a lower-case HDU name has to be to serve as an array's local identifier.

A PDS4 ``local_identifier`` is an XML ``ID``: a name that begins with a letter or an
underscore and holds only letters, digits, underscores, hyphens and periods.
"""

_SCALING_KEYWORDS = ('BSCALE', 'BZERO')
"""The header keywords that make an array's stored values other than its values."""


class UndescribableFitsError(ValueError):
    """A backplane FITS holds something its data label could not truthfully describe."""


@dataclass(frozen=True)
class FitsArray:
    """One image HDU's array, as the data label's ``Array_2D_Image`` describes it.

    Attributes:
        local_identifier: The HDU's name in lower case, by which the label's display
            settings refer to the array.
        offset: Where the array's first element is, in bytes from the start of the file.
        data_type: The PDS4 ``data_type`` of its elements, from the HDU's ``BITPIX``.
        unit: The HDU's ``BUNIT``, or None when the header has none.
        lines: The number of elements down the frame, the HDU's ``NAXIS2``.
        samples: The number of elements across the frame, the HDU's ``NAXIS1``.
        missing_constant: The value a float array holds wherever it measured nothing,
            spelled as the label states it; None for an integer array.
        description: What the array's values are, for the body identity map; None for
            every other array.
    """

    local_identifier: str
    offset: int
    data_type: str
    unit: str | None
    lines: int
    samples: int
    missing_constant: str | None
    description: str | None


@dataclass(frozen=True)
class FitsHdu:
    """One HDU of the file, as the data label's ``Header`` and array describe it.

    Attributes:
        name: The HDU's name, ``PRIMARY`` for the first.
        header_offset: Where the HDU's header begins, in bytes from the start of the
            file.
        header_length: How many bytes the header takes, padding included, so that
            the HDU's data begin this far past ``header_offset``.
        array: The array the HDU holds; None for the primary HDU, which holds none.
    """

    name: str
    header_offset: int
    header_length: int
    array: FitsArray | None


@dataclass(frozen=True)
class BackplaneFitsObjects:
    """Every HDU of one backplane FITS, in the order the file holds them.

    Attributes:
        hdus: One entry per HDU, the primary first.
    """

    hdus: tuple[FitsHdu, ...]

    @property
    def arrays(self) -> tuple[FitsArray, ...]:
        """The arrays the HDUs hold, in the order the file holds them."""
        return tuple(hdu.array for hdu in self.hdus if hdu.array is not None)


def describe_backplane_fits(
    fits_path: str | Path | FCPath, *, masked_value: float
) -> BackplaneFitsObjects:
    """Return the data objects a backplane FITS holds, read from the file.

    Every HDU gets its header's offset and length.  Every image HDU past the primary
    gets an array at the offset of its data, whose element type comes from its
    ``BITPIX`` through :data:`FITS_DATA_TYPES`, whose unit is its ``BUNIT`` when it has
    one, and whose lines and samples are its ``NAXIS2`` and ``NAXIS1``.  Its local
    identifier is its name in lower case.  A float array declares ``masked_value`` as
    its missing constant; an integer array declares none, and the body identity map
    carries :data:`BODY_ID_MAP_DESCRIPTION` instead.

    The file is read as it is stored, with no scaling applied, so that what is
    described is the bytes in the file.

    Parameters:
        fits_path: The FITS to describe.
        masked_value: The value a float plane holds wherever it measured nothing, the
            configuration's ``backplanes.masked_value``, taken to be one
            :func:`unusable_masked_value` accepts.

    Returns:
        The file's HDUs, the primary first, each with its header and its array.

    Raises:
        UndescribableFitsError: If astropy cannot read the file as FITS without an
            error or a warning, which a truncated file draws; its primary HDU holds
            data; an HDU past the primary is not an image; or an image HDU is not
            two-dimensional, has a ``BITPIX`` that :data:`FITS_DATA_TYPES` does not
            map, carries ``BSCALE`` or ``BZERO``, or has a name that in lower case is
            not an XML name or is another image HDU's.  The message names the file,
            and the HDU by its index and name.
        FileNotFoundError: If there is no file at ``fits_path``.
        astropy.io.fits.verify.VerifyError: If a header card the description reads,
            ``EXTNAME`` or ``BUNIT``, is one astropy cannot parse.
    """
    fcpath = FCPath(fits_path)
    missing_constant = _missing_constant(masked_value)
    with fcpath.open('rb') as fits_file:
        try:
            # A file astropy reads only with a warning -- one cut short, or with a
            # header that does not verify -- is not one whose offsets a label can
            # state, so a warning it draws while reading is a refusal like an error.
            with warnings.catch_warnings():
                warnings.simplefilter('error', AstropyUserWarning)
                hdul = fits.open(fits_file, do_not_scale_image_data=True, lazy_load_hdus=False)
        except (OSError, AstropyUserWarning) as exc:
            raise UndescribableFitsError(
                f'{fcpath} is not a FITS file astropy reads cleanly: {exc}'
            ) from exc
        with hdul:
            hdus = tuple(
                _describe_hdu(
                    hdu,
                    where=f'HDU {index} ({hdu.name or "unnamed"}) of {fcpath}',
                    is_primary=index == 0,
                    missing_constant=missing_constant,
                )
                for index, hdu in enumerate(hdul)
            )
    identifiers = [array.local_identifier for hdu in hdus if (array := hdu.array) is not None]
    repeated = sorted({name for name in identifiers if identifiers.count(name) > 1})
    if repeated:
        raise UndescribableFitsError(
            f'{fcpath} holds more than one image HDU named {", ".join(repeated)} in lower '
            'case, and each array needs a local identifier of its own'
        )
    return BackplaneFitsObjects(hdus=hdus)


def unusable_masked_value(config: Config) -> str | None:
    """Return why the configured masked value cannot be a label's missing constant.

    Every float array of every data label declares ``backplanes.masked_value`` as its
    missing constant, and the backplane writer fills every unmeasured pixel of a float
    plane with it, so it has to be a number a 32-bit float holds exactly: one that is
    not a number, not finite, or not exactly representable would be declared as a value
    no masked pixel holds.  It is the same for every image, so the drivers ask this
    once, before any image is processed.

    Parameters:
        config: The configuration whose ``backplanes.masked_value`` is checked.

    Returns:
        None when the value is usable, and otherwise a sentence naming the value and
        what is wrong with it: that it is not a number, not a finite number, or not
        one a 32-bit float holds exactly.
    """
    value = config.backplanes.masked_value
    if isinstance(value, bool) or not isinstance(value, int | float):
        return f'the configured masked value {value!r} is not a number'
    if not math.isfinite(value):
        return f'the configured masked value {value!r} is not a finite number'
    with np.errstate(over='ignore'):
        stored = float(np.float32(value))
    if stored != value:
        return (
            f'the configured masked value {value!r} is not one a 32-bit float holds '
            'exactly, so no float plane can hold it'
        )
    return None


def _missing_constant(masked_value: float) -> str:
    """Return the masked value as a label states it.

    The value is taken to be one :func:`unusable_masked_value` accepts, which the
    drivers establish once before any image is processed.

    Parameters:
        masked_value: The configuration's masked value.

    Returns:
        The shortest decimal spelling of the 32-bit float every masked pixel of a
        float plane holds.
    """
    return repr(float(np.float32(masked_value)))


def _describe_hdu(
    hdu: fits.hdu.base._BaseHDU,
    *,
    where: str,
    is_primary: bool,
    missing_constant: str,
) -> FitsHdu:
    """Return one HDU's header and array, refusing what the writer does not write.

    Parameters:
        hdu: The HDU, from a file opened with no scaling applied.
        where: The HDU's index and name and the file's path, for a refusal's message.
        is_primary: Whether this is the file's first HDU.
        missing_constant: The masked value, spelled as the label states it.

    Returns:
        The HDU's header, and its array when it is an image past the primary.

    Raises:
        UndescribableFitsError: If the HDU is one a backplane FITS does not hold, as
            :func:`describe_backplane_fits` lists.
    """
    info = hdu.fileinfo()
    header_offset = int(info['hdrLoc'])
    data_offset = int(info['datLoc'])
    header = hdu.header
    naxis = int(header['NAXIS'])
    if is_primary:
        if naxis != 0:
            raise UndescribableFitsError(
                f'{where} holds data (NAXIS = {naxis}), where a backplane FITS keeps its '
                'primary HDU empty'
            )
        return FitsHdu(
            name=hdu.name,
            header_offset=header_offset,
            header_length=data_offset - header_offset,
            array=None,
        )

    if not isinstance(hdu, fits.ImageHDU):
        raise UndescribableFitsError(f'{where} is a {type(hdu).__name__}, not an image')
    if naxis != 2:
        raise UndescribableFitsError(f'{where} is not two-dimensional (NAXIS = {naxis})')
    bitpix = int(header['BITPIX'])
    if bitpix not in FITS_DATA_TYPES:
        raise UndescribableFitsError(
            f'{where} has BITPIX = {bitpix}, where a backplane array has one of '
            f'{", ".join(str(value) for value in FITS_DATA_TYPES)}'
        )
    scaling = [keyword for keyword in _SCALING_KEYWORDS if keyword in header]
    if scaling:
        raise UndescribableFitsError(
            f'{where} carries {" and ".join(scaling)}, so its stored values are not its values'
        )
    local_identifier = hdu.name.lower()
    if _LOCAL_IDENTIFIER.fullmatch(local_identifier) is None:
        raise UndescribableFitsError(
            f'{where} has a name that in lower case is not a local identifier, which '
            'begins with a letter or an underscore and holds only letters, digits, '
            'underscores, hyphens and periods'
        )
    unit = header.get('BUNIT')
    return FitsHdu(
        name=hdu.name,
        header_offset=header_offset,
        header_length=data_offset - header_offset,
        array=FitsArray(
            local_identifier=local_identifier,
            offset=data_offset,
            data_type=FITS_DATA_TYPES[bitpix],
            unit=None if unit is None else str(unit),
            lines=int(header['NAXIS2']),
            samples=int(header['NAXIS1']),
            missing_constant=missing_constant if bitpix < 0 else None,
            description=BODY_ID_MAP_DESCRIPTION if hdu.name == BODY_ID_MAP_HDU_NAME else None,
        ),
    )
