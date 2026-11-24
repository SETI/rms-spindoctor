from .dataset import DataSet
from .dataset_pds3_cassini_iss import (
    DataSetPDS3CassiniISS,
    DataSetPDS3CassiniISSCruise,
    DataSetPDS3CassiniISSSaturn,
)
from .dataset_pds3_galileo_ssi import DataSetPDS3GalileoSSI
from .dataset_pds3_newhorizons_lorri import DataSetPDS3NewHorizonsLORRI
from .dataset_pds3_voyager_iss import DataSetPDS3VoyagerISS
from .dataset_sim import DataSetSim

_DATASET_NAME_TO_CLASS_MAPPING: dict[str, type[DataSet]] = {
    'coiss': DataSetPDS3CassiniISS,
    'coiss_pds3': DataSetPDS3CassiniISS,
    'coiss_cruise': DataSetPDS3CassiniISSCruise,
    'coiss_cruise_pds3': DataSetPDS3CassiniISSCruise,
    'coiss_saturn': DataSetPDS3CassiniISSSaturn,
    'coiss_saturn_pds3': DataSetPDS3CassiniISSSaturn,
    'gossi': DataSetPDS3GalileoSSI,
    'gossi_pds3': DataSetPDS3GalileoSSI,
    'nhlorri': DataSetPDS3NewHorizonsLORRI,
    'nhlorri_pds3': DataSetPDS3NewHorizonsLORRI,
    'vgiss': DataSetPDS3VoyagerISS,
    'vgiss_pds3': DataSetPDS3VoyagerISS,
    'sim': DataSetSim,
}


_DATASET_NAME_TO_INST_NAME_MAPPING = {
    'coiss': 'coiss',
    'coiss_pds3': 'coiss',
    'coiss_cruise': 'coiss',
    'coiss_cruise_pds3': 'coiss',
    'coiss_saturn': 'coiss',
    'coiss_saturn_pds3': 'coiss',
    'gossi': 'gossi',
    'gossi_pds3': 'gossi',
    'nhlorri': 'nhlorri',
    'nhlorri_pds3': 'nhlorri',
    'vgiss': 'vgiss',
    'vgiss_pds3': 'vgiss',
    'sim': 'sim',
}


assert sorted(_DATASET_NAME_TO_CLASS_MAPPING.keys()) == sorted(
    _DATASET_NAME_TO_INST_NAME_MAPPING.keys()
), 'Dataset names are inconsistent'


def dataset_names() -> list[str]:
    """Return a list of all dataset names."""
    return sorted(_DATASET_NAME_TO_CLASS_MAPPING.keys())


def pds4_config_keys() -> tuple[str, ...]:
    """Return the key of each registered dataset's bundle in the ``pds4`` config section.

    Returns:
        Each key a registered dataset class gives as its ``PDS4_CONFIG_KEY``, once each
        and sorted.  A dataset that makes no bundle gives none.
    """
    return tuple(
        sorted(
            {
                cls.PDS4_CONFIG_KEY
                for cls in _DATASET_NAME_TO_CLASS_MAPPING.values()
                if cls.PDS4_CONFIG_KEY is not None
            }
        )
    )


def dataset_name_to_class(name: str) -> type[DataSet]:
    """Convert a dataset name to the corresponding class.

    Parameters:
        name: The name of the dataset.

    Returns:
        The class corresponding to the dataset name.

    Raises:
        KeyError: If the dataset name is not found; the message lists the
            valid names.
    """
    try:
        return _DATASET_NAME_TO_CLASS_MAPPING[name.lower()]
    except KeyError:
        valid = ', '.join(sorted(_DATASET_NAME_TO_CLASS_MAPPING))
        raise KeyError(f'unknown dataset name {name!r}; valid names: {valid}') from None


def dataset_name_to_inst_name(name: str) -> str:
    """Convert a dataset name to the corresponding instrument name.

    Parameters:
        name: The name of the dataset.

    Returns:
        The instrument name corresponding to the dataset name.

    Raises:
        KeyError: If the dataset name is not found; the message lists the
            valid names.
    """
    try:
        return _DATASET_NAME_TO_INST_NAME_MAPPING[name.lower()]
    except KeyError:
        valid = ', '.join(sorted(_DATASET_NAME_TO_INST_NAME_MAPPING))
        raise KeyError(f'unknown dataset name {name!r}; valid names: {valid}') from None


__all__ = [
    'DataSet',
    'DataSetPDS3CassiniISS',
    'DataSetPDS3CassiniISSCruise',
    'DataSetPDS3CassiniISSSaturn',
    'DataSetPDS3GalileoSSI',
    'DataSetPDS3NewHorizonsLORRI',
    'DataSetPDS3VoyagerISS',
    'DataSetSim',
    'dataset_name_to_class',
    'dataset_name_to_inst_name',
    'dataset_names',
    'pds4_config_keys',
]
