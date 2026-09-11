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
Whether a file is cut short is decided from its length and the offsets astropy
reads, not from the warnings astropy emits, which also flag files the FITS standard
allows and are not a refusal here.
"""

import io
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
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

FITS_RECORD = 2880
"""The length of a FITS logical record: every header and every data unit fills whole
records, so a whole FITS file is a whole number of them."""

_LOCAL_IDENTIFIER = re.compile(r'[a-z_][a-z0-9_.-]*')
"""What a lower-case HDU name has to be to serve as an array's local identifier.

A PDS4 ``local_identifier`` is an XML ``ID``: a name that begins with a letter or an
underscore and holds only letters, digits, underscores, hyphens and periods.
"""

_SCALING_KEYWORDS = ('BSCALE', 'BZERO')
"""The header keywords that make an array's stored values other than its values."""

SUPPLEMENTAL_FILE_IDENTIFIER = 'navigation-details'
"""The ``local_identifier`` the data label gives its supplemental file.

``data.lblx`` states it through a template variable set from this constant, and no
array may take it, since a local identifier is its label's only one of its name.
"""

LABEL_LOCAL_IDENTIFIERS = frozenset({SUPPLEMENTAL_FILE_IDENTIFIER})
"""Every ``local_identifier`` the data label defines of its own, none of which an
array may take."""


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
        description: What the array holds.  For a float plane: its name, the oops
            backplane method it came from where the configuration names one, its unit
            where it has one, and that a pixel the plane does not cover holds the
            missing constant.  For the body identity map,
            :data:`BODY_ID_MAP_DESCRIPTION`.  None for any other integer array.
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
    fits_path: str | Path | FCPath,
    *,
    masked_value: float,
    methods: Mapping[str, str] | None = None,
) -> BackplaneFitsObjects:
    """Return the data objects a backplane FITS holds, read from the file.

    Every HDU gets its header's offset and length.  Every image HDU past the primary
    gets an array at the offset of its data, whose element type comes from its
    ``BITPIX`` through :data:`FITS_DATA_TYPES`, whose unit is its ``BUNIT`` when it has
    one, and whose lines and samples are its ``NAXIS2`` and ``NAXIS1``.  Its local
    identifier is its name in lower case.  A float array declares ``masked_value`` as
    its missing constant; an integer array declares none, and the body identity map
    carries :data:`BODY_ID_MAP_DESCRIPTION` instead.  Every float array carries a
    description of what it holds: its name, the oops backplane method it came from
    when ``methods`` names one, its unit, and that a pixel the plane does not cover
    holds the missing constant.

    The file is read as it is stored, with no scaling applied, so that what is
    described is the bytes in the file.  A warning astropy emits while reading it is
    not a refusal: whether the file is whole is decided from its length and the
    offsets of its HDUs.

    Parameters:
        fits_path: The FITS to describe.
        masked_value: The value a float plane holds wherever it measured nothing, the
            configuration's ``backplanes.masked_value``, taken to be one
            :func:`unusable_masked_value` accepts.
        methods: The oops backplane method each plane was computed with, by the plane's
            configured name, as :func:`configured_methods` gives them; a plane it does
            not name is described without one.

    Returns:
        The file's HDUs, the primary first, each with its header and its array.

    Raises:
        UndescribableFitsError: If astropy cannot read the file as FITS, which a
            header cut short at the end of a record draws; the file is cut short
            otherwise -- an HDU's data run past its end, or its length is not a whole
            number of :data:`FITS_RECORD`-byte records, which a header cut short within
            a record, and dropped by astropy, leaves; its primary HDU holds data; an
            HDU past the primary is not an image, which a tile-compressed image is,
            being a binary table on disk; or an image HDU is not
            two-dimensional, has a ``BITPIX`` that :data:`FITS_DATA_TYPES` does not
            map, carries ``BSCALE`` or ``BZERO``, or has a name that in lower case is
            not an XML name, is another image HDU's, or is one of
            :data:`LABEL_LOCAL_IDENTIFIERS`.  The message names the file, and
            the HDU by its index and name where the refusal is of one HDU, with the
            byte counts where it is of the file's length.
        FileNotFoundError: If there is no file at ``fits_path``.
        astropy.io.fits.verify.VerifyError: If a header card the description reads,
            ``EXTNAME`` or ``BUNIT``, is one astropy cannot parse.
    """
    fcpath = FCPath(fits_path)
    missing_constant = _missing_constant(masked_value)
    with fcpath.open('rb') as fits_file:
        file_size = fits_file.seek(0, io.SEEK_END)
        fits_file.seek(0)
        try:
            # astropy decompresses a tile-compressed image and hands back an ImageHDU
            # subclass with the image's header, where the file holds a binary table;
            # without decompression the HDU is the table it is on disk, and is refused
            # as an extension that is not an image.
            hdul = fits.open(
                fits_file,
                do_not_scale_image_data=True,
                lazy_load_hdus=False,
                disable_image_compression=True,
            )
        except OSError as exc:
            raise UndescribableFitsError(
                f'{fcpath} is not a FITS file astropy can read: {exc}'
            ) from exc
        with hdul:
            _refuse_data_past_the_end(hdul, file_size=file_size, fcpath=fcpath)
            hdus = tuple(
                _describe_hdu(
                    hdu,
                    where=_where(index, hdu, fcpath),
                    is_primary=index == 0,
                    missing_constant=missing_constant,
                    methods={} if methods is None else methods,
                )
                for index, hdu in enumerate(hdul)
            )
    # astropy drops, with a warning, an extension whose header the file cuts short
    # within a record, and every HDU it keeps then ends within the file; what the cut
    # leaves is a length that is not a whole number of records.
    if file_size % FITS_RECORD != 0:
        raise UndescribableFitsError(
            f'{fcpath} is {file_size} bytes, not a whole number of {FITS_RECORD}-byte records'
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


def configured_methods(config: Config) -> dict[str, str]:
    """Return the oops backplane method each configured plane is computed with.

    Parameters:
        config: The configuration whose ``backplanes.bodies`` and ``backplanes.rings``
            entries are read.

    Returns:
        Each entry's ``method`` by its ``name``, for every entry that names both.
    """
    entries = [*config.backplanes.bodies, *config.backplanes.rings]
    return {
        str(entry['name']): str(entry['method'])
        for entry in entries
        if 'name' in entry and 'method' in entry
    }


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


def _where(index: int, hdu: fits.hdu.base._BaseHDU, fcpath: FCPath) -> str:
    """Return how a refusal names one HDU of a file.

    Parameters:
        index: The HDU's position in the file, from zero.
        hdu: The HDU.
        fcpath: The file.

    Returns:
        The HDU's index and name, ``unnamed`` for an HDU with none, and the file.
    """
    return f'HDU {index} ({hdu.name or "unnamed"}) of {fcpath}'


def _data_bytes(header: fits.Header) -> int:
    """Return how many bytes an HDU's data take, before their padding.

    Parameters:
        header: The HDU's header.

    Returns:
        ``|BITPIX| / 8 * GCOUNT * (PCOUNT + NAXIS1 * ... * NAXISn)``, the FITS
        standard's size of a data unit, and 0 for an HDU with no axes.
    """
    naxis = int(header['NAXIS'])
    if naxis == 0:
        return 0
    elements = math.prod(int(header[f'NAXIS{axis}']) for axis in range(1, naxis + 1))
    group_count = int(header.get('GCOUNT', 1))
    parameter_count = int(header.get('PCOUNT', 0))
    return abs(int(header['BITPIX'])) // 8 * group_count * (parameter_count + elements)


def _refuse_data_past_the_end(hdul: fits.HDUList, *, file_size: int, fcpath: FCPath) -> None:
    """Refuse a file holding an HDU whose data run past its end.

    astropy reads such an HDU with a warning and goes on, so the offsets it reports are
    held to the file's length here.

    Parameters:
        hdul: The file's HDUs, as astropy reads them.
        file_size: The file's length in bytes.
        fcpath: The file, for a refusal's message.

    Raises:
        UndescribableFitsError: If an HDU's data run past the end of the file, naming
            the HDU, where its data end and where the file does.
    """
    for index, hdu in enumerate(hdul):
        data_end = int(hdu.fileinfo()['datLoc']) + _data_bytes(hdu.header)
        if data_end > file_size:
            raise UndescribableFitsError(
                f'{_where(index, hdu, fcpath)} has data running to byte {data_end}, and '
                f'the file ends at byte {file_size}'
            )


def _array_description(
    hdu_name: str,
    local_identifier: str,
    *,
    is_float: bool,
    method: str | None,
    unit: str | None,
    missing_constant: str,
) -> str | None:
    """Return what the label says an array holds.

    Parameters:
        hdu_name: The HDU's name as the file gives it.
        local_identifier: The array's identifier, that name in lower case.
        is_float: Whether the array holds 32-bit floats.
        method: The oops backplane method the plane came from, when the configuration
            names one.
        unit: The HDU's ``BUNIT``, when it has one.
        missing_constant: The masked value, spelled as the label states it.

    Returns:
        :data:`BODY_ID_MAP_DESCRIPTION` for the body identity map.  For a float plane,
        a sentence naming the plane and, where there are ones, its method and its unit,
        and a sentence saying a pixel the plane does not cover holds the missing
        constant.  None for any other integer array.
    """
    if hdu_name == BODY_ID_MAP_HDU_NAME:
        return BODY_ID_MAP_DESCRIPTION
    if not is_float:
        return None
    source = '' if method is None else f', from the oops backplane method {method},'
    measure = '' if unit is None else f', in {unit}'
    return (
        f'The {local_identifier} backplane{source} evaluated at each pixel of the image'
        f'{measure}. A pixel the plane does not cover holds the missing constant, '
        f'{missing_constant}.'
    )


def _describe_hdu(
    hdu: fits.hdu.base._BaseHDU,
    *,
    where: str,
    is_primary: bool,
    missing_constant: str,
    methods: Mapping[str, str],
) -> FitsHdu:
    """Return one HDU's header and array, refusing what the writer does not write.

    Parameters:
        hdu: The HDU, from a file opened with no scaling applied.
        where: The HDU's index and name and the file's path, for a refusal's message.
        is_primary: Whether this is the file's first HDU.
        missing_constant: The masked value, spelled as the label states it.
        methods: The oops backplane method each plane was computed with, by name.

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
    if local_identifier in LABEL_LOCAL_IDENTIFIERS:
        raise UndescribableFitsError(
            f'{where} has the name {local_identifier} in lower case, which the data '
            'label gives to another of its objects'
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
            description=_array_description(
                hdu.name,
                local_identifier,
                is_float=bitpix < 0,
                method=methods.get(local_identifier),
                unit=None if unit is None else str(unit),
                missing_constant=missing_constant,
            ),
        ),
    )
