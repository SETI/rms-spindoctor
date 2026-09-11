"""The data objects a backplane FITS holds, as its PDS4 data label describes them.

A backplane FITS, as :func:`~spindoctor.cli.backplanes.writer.write_fits` writes it, is
an empty primary HDU followed by one image HDU per array: the body identity map, when
some body claimed a pixel, and then one float plane per backplane that has a valid
pixel.  A data label describes the header of every HDU as a ``Header`` and every image
past the primary as an ``Array_2D_Image``, each at the byte offset the file holds it
at.  :func:`describe_backplane_fits` reads those offsets, and everything else the label
states about an array, from the file itself rather than from the configuration that
asked for it, so that the label describes the file in the bundle: a plane the writer
dropped is not described, and one it wrote is described as it was written.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

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
        missing_constant: The value a float plane holds wherever it measured nothing,
            spelled as the label states it; None for the body identity map.
        description: What the array holds.  For the body identity map,
            :data:`BODY_ID_MAP_DESCRIPTION`.  For a float plane: its name, the oops
            backplane method it came from where the configuration names one, its unit
            where it has one, and that a pixel the plane does not cover holds the
            missing constant.
    """

    local_identifier: str
    offset: int
    data_type: str
    unit: str | None
    lines: int
    samples: int
    missing_constant: str | None
    description: str


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
    fits_path: Path,
    *,
    masked_value: float,
    methods: Mapping[str, str] | None = None,
) -> BackplaneFitsObjects:
    """Return the data objects a backplane FITS holds, read from the file.

    Every HDU gets its header's offset and length.  Every image HDU past the primary
    gets an array at the offset of its data, whose element type comes from its
    ``BITPIX`` through :data:`FITS_DATA_TYPES`, whose unit is its ``BUNIT`` when it has
    one, and whose lines and samples are its ``NAXIS2`` and ``NAXIS1``.  Its local
    identifier is its name in lower case.  The body identity map declares no missing
    constant and carries :data:`BODY_ID_MAP_DESCRIPTION`.  Every other array is a float
    plane, which declares ``masked_value`` as its missing constant and carries a
    description of what it holds: its name, the oops backplane method it came from
    when ``methods`` names one, its unit, and that a pixel the plane does not cover
    holds the missing constant.

    Parameters:
        fits_path: The local FITS to describe, as the backplane writer writes one.
        masked_value: The value a float plane holds wherever it measured nothing, the
            configuration's ``backplanes.masked_value``.
        methods: The oops backplane method each plane was computed with, by the plane's
            configured name, as :func:`configured_methods` gives them; a plane it does
            not name is described without one.

    Returns:
        The file's HDUs, the primary first, each with its header and its array.
    """
    missing_constant = _missing_constant(masked_value)
    with fits.open(fits_path) as hdul:
        hdus = tuple(
            _describe_hdu(
                hdu,
                is_primary=index == 0,
                missing_constant=missing_constant,
                methods={} if methods is None else methods,
            )
            for index, hdu in enumerate(hdul)
        )
    return BackplaneFitsObjects(hdus=hdus)


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

    Parameters:
        masked_value: The configuration's masked value.

    Returns:
        The shortest decimal spelling of the 32-bit float every masked pixel of a
        float plane holds.
    """
    return repr(float(np.float32(masked_value)))


def _plane_description(
    local_identifier: str, *, method: str | None, unit: str | None, missing_constant: str
) -> str:
    """Return what the label says a float plane holds.

    Parameters:
        local_identifier: The plane's identifier, its HDU name in lower case.
        method: The oops backplane method the plane came from, when the configuration
            names one.
        unit: The HDU's ``BUNIT``, when it has one.
        missing_constant: The masked value, spelled as the label states it.

    Returns:
        A sentence naming the plane and, where there are ones, its method and its unit,
        and a sentence saying a pixel the plane does not cover holds the missing
        constant.
    """
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
    is_primary: bool,
    missing_constant: str,
    methods: Mapping[str, str],
) -> FitsHdu:
    """Return one HDU's header and, past the primary, its array.

    Parameters:
        hdu: The HDU.
        is_primary: Whether this is the file's first HDU, which holds no array.
        missing_constant: The masked value, spelled as the label states it.
        methods: The oops backplane method each plane was computed with, by name.

    Returns:
        The HDU's header, and its array when it is an image past the primary.
    """
    info = hdu.fileinfo()
    header_offset = int(info['hdrLoc'])
    data_offset = int(info['datLoc'])
    if is_primary:
        return FitsHdu(
            name=hdu.name,
            header_offset=header_offset,
            header_length=data_offset - header_offset,
            array=None,
        )
    header = hdu.header
    local_identifier = hdu.name.lower()
    bunit = header.get('BUNIT')
    unit = None if bunit is None else str(bunit)
    if hdu.name == BODY_ID_MAP_HDU_NAME:
        missing: str | None = None
        description = BODY_ID_MAP_DESCRIPTION
    else:
        missing = missing_constant
        description = _plane_description(
            local_identifier,
            method=methods.get(local_identifier),
            unit=unit,
            missing_constant=missing_constant,
        )
    return FitsHdu(
        name=hdu.name,
        header_offset=header_offset,
        header_length=data_offset - header_offset,
        array=FitsArray(
            local_identifier=local_identifier,
            offset=data_offset,
            data_type=FITS_DATA_TYPES[int(header['BITPIX'])],
            unit=unit,
            lines=int(header['NAXIS2']),
            samples=int(header['NAXIS1']),
            missing_constant=missing,
            description=description,
        ),
    )
