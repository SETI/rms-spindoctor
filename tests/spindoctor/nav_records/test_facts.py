"""Turning one navigation metadata document into the facts a consumer reads.

What these pin is mostly what the flattening does *not* do: it does not key an
image by its name, does not round the offset it stores, does not merge the two
reason vocabularies, does not reduce a matrix to part of itself, and does not
read a file that is some other kind of document at all.

A navigation document is written by the navigator and by nothing else, so the
values here are the ones it writes: an absent field, a recorded null, a matrix
it did record, an epoch it did not.  A file that is not a navigation document at
all is refused whole, which is its own subject below.
"""

from typing import Any

import pytest
from tests.spindoctor.conftest import (
    metadata_document,
    technique,
)

from spindoctor.nav_records import facts as facts_module
from spindoctor.nav_records.facts import (
    DocumentOrigin,
    MetadataDocumentError,
    facts_from_document,
)

SOURCE = DocumentOrigin(
    root_url='/data/nav-results',
    results_path_stub='COISS_2001/data/1294561143_1295221348/N1294561202_1_CALIB',
    source_file='/data/nav-results/x_metadata.json',
    mtime_ns=1234567890123456789,
    size_bytes=4096,
)

TWIST_COVARIANCE = [
    [0.0961, 0.0100, 0.0025],
    [0.0100, 0.0784, -0.0050],
    [0.0025, -0.0050, 0.0009],
]
"""A 3x3 covariance of the kind a twist-fitted result records.

Its third row and column are the rotation's, and its two off-diagonal entries
there are the offset-to-rotation cross terms: the numbers that say how much of
the offset uncertainty is the twist's, which no per-axis sigma and no rotation
sigma states.  Deliberately not symmetric-by-accident about anything, so a
column that stored the wrong entry produces the wrong number rather than the
right one.
"""


def test_the_document_reader_attaches_no_record() -> None:
    """One producer of the carried record, and this is not it.

    A set of facts carrying a record came from a source that read a document and
    still holds it.  This function is handed the parsed document by that source
    and by nothing else, so filling the field here would put a second producer
    beside it and make "carries a record" stop meaning "came from a document".

    """
    rows = facts_from_document(metadata_document(), SOURCE)
    assert rows.record is None


def test_the_stub_and_root_key_the_row() -> None:
    """The key comes from where the file is, never from the document."""
    rows = facts_from_document(metadata_document(), SOURCE)
    assert rows.image['results_path_stub'] == SOURCE.results_path_stub


def test_the_root_is_recorded_as_given() -> None:
    """The other half of the key is the root the walk was told to read."""
    rows = facts_from_document(metadata_document(), SOURCE)
    assert rows.image['root_url'] == SOURCE.root_url


def test_the_subtree_is_the_stubs_first_segment() -> None:
    """A stub under a directory yields it without string surgery in SQL."""
    rows = facts_from_document(metadata_document(), SOURCE)
    assert rows.image['subtree'] == 'COISS_2001'


def test_a_bare_basename_stub_has_no_subtree() -> None:
    """The simulated dataset produces a stub with no separator, and no subtree."""
    source = DocumentOrigin(
        root_url='/data/nav-results',
        results_path_stub='sim_scene_000042',
        source_file='/data/nav-results/sim_scene_000042_metadata.json',
        mtime_ns=1,
        size_bytes=2,
    )
    rows = facts_from_document(metadata_document(instrument='sim'), source)
    assert rows.image['subtree'] is None


def test_the_stored_offset_is_the_top_level_one() -> None:
    """The authoritative offset is the top-level field, not the display copy."""
    document = metadata_document(offset=[3.14159265358979, -2.71828182845905])
    document['navigation_result']['offset_px'] = [3.142, -2.718]
    rows = facts_from_document(document, SOURCE)
    assert rows.image['offset_dv'] == 3.14159265358979


def test_the_stored_offset_is_not_rounded() -> None:
    """A fifteen-digit offset round-trips through the row unchanged."""
    document = metadata_document(offset=[3.14159265358979, -2.71828182845905])
    document['navigation_result']['offset_px'] = [3.142, -2.718]
    rows = facts_from_document(document, SOURCE)
    assert rows.image['offset_du'] == -2.71828182845905


def test_the_stored_confidence_is_the_top_level_one() -> None:
    """Confidence follows the offset: the value, not the rounded display copy."""
    document = metadata_document(confidence=0.876543210987654)
    document['navigation_result']['confidence'] = 0.877
    rows = facts_from_document(document, SOURCE)
    assert rows.image['confidence'] == 0.876543210987654


def test_a_missing_top_level_offset_is_null() -> None:
    """A document with no offset stores none, whatever the display copy says."""
    document = metadata_document(status='failed', status_reason='no_features', offset=None)
    document['navigation_result']['offset_px'] = [9.0, 9.0]
    rows = facts_from_document(document, SOURCE)
    assert rows.image['offset_dv'] is None


def test_a_document_naming_a_status_keeps_it() -> None:
    """The control: the field the ladder reads survives into the column verbatim."""
    rows = facts_from_document(metadata_document(status='conflicted'), SOURCE)
    assert rows.image['status'] == 'conflicted'


def test_status_error_is_stored_verbatim() -> None:
    """The selection filter matches this token exactly, so nothing may touch it."""
    document = metadata_document(
        status='error', status_reason=None, status_error='missing_spice_data', offset=None
    )
    rows = facts_from_document(document, SOURCE)
    assert rows.image['status_error'] == 'missing_spice_data'


def test_the_traceback_is_stored_whole() -> None:
    """Its newlines and every frame, so the row answers what the log would."""
    traceback = 'Traceback (most recent call last):\n  File "obs.py", line 1\nOSError: gone'
    document = metadata_document(
        status='error',
        status_reason=None,
        status_error='image_read_error',
        status_traceback=traceback,
        offset=None,
    )
    rows = facts_from_document(document, SOURCE)
    assert rows.image['status_traceback'] == traceback


def test_a_document_with_no_traceback_stores_none() -> None:
    """Every navigated image is one, and NULL is what a query for them asks on."""
    rows = facts_from_document(metadata_document(), SOURCE)
    assert rows.image['status_traceback'] is None


def test_status_error_does_not_reach_the_reason_column() -> None:
    """The two vocabularies stay in their own columns rather than merging."""
    document = metadata_document(
        status='error', status_reason=None, status_error='missing_spice_data', offset=None
    )
    rows = facts_from_document(document, SOURCE)
    assert rows.image['status_reason'] is None


def test_status_reason_does_not_reach_the_error_column() -> None:
    """And the reverse, so a filter on one never matches a value of the other."""
    document = metadata_document(status='failed', status_reason='no_features', offset=None)
    rows = facts_from_document(document, SOURCE)
    assert rows.image['status_error'] is None


@pytest.mark.parametrize(
    ('sigma', 'expected'),
    [
        ([0.5, 0.75], (0.5, 0.75)),
        ([0.5, 0.75, 1.0], (None, None)),
        ([0.5], (None, None)),
        ([], (None, None)),
    ],
    ids=['pair', 'three', 'one', 'empty'],
)
def test_a_recorded_sigma_pair_is_refused_whole_unless_it_is_a_pair(
    sigma: Any, expected: tuple[float | None, float | None]
) -> None:
    """Two of three recorded numbers are not the pair anybody wrote.

    The per-axis uncertainty an operator reads off a report has to be the one
    the navigation recorded; taking the first two of three would report an
    uncertainty from a value of some other shape as though it were that pair.

    Parameters:
        sigma: The recorded ``navigation_result.sigma_px``.
        expected: The ``(sigma_dv, sigma_du)`` the columns must hold.
    """
    document = metadata_document()
    document['navigation_result']['sigma_px'] = sigma
    rows = facts_from_document(document, SOURCE)
    assert (rows.image['sigma_dv'], rows.image['sigma_du']) == expected


@pytest.mark.parametrize(
    ('offset', 'expected'),
    [([1.5, -2.5], (1.5, -2.5)), ([1.5, -2.5, 9.0], (None, None)), ([1.5], (None, None))],
    ids=['pair', 'three', 'one'],
)
def test_a_techniques_offset_is_refused_whole_unless_it_is_a_pair(
    offset: Any, expected: tuple[float | None, float | None]
) -> None:
    """The same rule for the per-technique estimates a report compares.

    Parameters:
        offset: The recorded ``per_technique[].offset_px``.
        expected: The ``(offset_dv, offset_du)`` the row must hold.
    """
    entry = technique('StarFieldFromCatalogNav', (0.0, 0.0))
    entry['offset_px'] = offset
    document = metadata_document(per_technique=[entry])
    rows = facts_from_document(document, SOURCE)
    assert (rows.techniques[0]['offset_dv'], rows.techniques[0]['offset_du']) == expected


def test_a_two_by_two_covariance_is_stored_as_written() -> None:
    """The common case: an untwisted fit records four numbers and keeps four."""
    document = metadata_document()
    document['navigation_result']['covariance_px2'] = [[0.0961, 0.0100], [0.0100, 0.0784]]
    rows = facts_from_document(document, SOURCE)
    assert rows.image['covariance_px2'] == [[0.0961, 0.0100], [0.0100, 0.0784]]


def test_a_twist_fitted_covariance_is_stored_whole() -> None:
    """The 3x3 case: nine numbers recorded, nine stored, in the same places."""
    document = metadata_document()
    document['navigation_result']['covariance_px2'] = TWIST_COVARIANCE
    rows = facts_from_document(document, SOURCE)
    assert rows.image['covariance_px2'] == TWIST_COVARIANCE


def test_the_offset_to_rotation_cross_terms_survive() -> None:
    """The terms a stored offset block loses, and that no sigma states.

    ``sigma_rotation_deg`` is the rotation's own uncertainty and says nothing
    about how much of the offset uncertainty is the twist's, so a reader that
    wants to propagate the twist into an offset needs these two entries and
    cannot reconstruct them from anything else the document records.
    """
    document = metadata_document()
    document['navigation_result']['covariance_px2'] = TWIST_COVARIANCE
    rows = facts_from_document(document, SOURCE)
    cross_terms = [rows.image['covariance_px2'][0][2], rows.image['covariance_px2'][1][2]]
    assert cross_terms == [0.0025, -0.0050]


def test_a_recorded_covariance_of_integers_is_stored_as_numbers() -> None:
    """JSON has one number type, so an identity matrix arrives as integers."""
    document = metadata_document()
    document['navigation_result']['covariance_px2'] = [[1, 0], [0, 1]]
    rows = facts_from_document(document, SOURCE)
    assert rows.image['covariance_px2'] == [[1.0, 0.0], [0.0, 1.0]]


def test_a_covariance_no_reader_could_use_is_stored_as_nothing() -> None:
    """A recorded null is no matrix, and every unsuccessful navigation records one."""
    document = metadata_document()
    document['navigation_result']['covariance_px2'] = None
    rows = facts_from_document(document, SOURCE)
    assert rows.image['covariance_px2'] is None


def test_a_techniques_covariance_is_stored_whole() -> None:
    """A technique's own matrix is kept for the same reason the fused one is."""
    entry = technique('BodyLimbNav', (1.0, 2.0))
    entry['covariance_px2'] = TWIST_COVARIANCE
    rows = facts_from_document(metadata_document(per_technique=[entry]), SOURCE)
    assert rows.techniques[0]['covariance_px2'] == TWIST_COVARIANCE


def test_the_recorded_epoch_is_copied_out_of_the_provenance_block() -> None:
    """The one field an image's epoch comes from, copied as the document wrote it.

    It is also the column a date filter and a range report compare against.
    """
    rows = facts_from_document(metadata_document(image_et=170002800.0), SOURCE)
    assert rows.image['image_et'] == 170002800.0


def test_the_derived_date_is_rendered_from_the_recorded_epoch() -> None:
    """The date a ``--start-date`` bound compares against, to the day."""
    rows = facts_from_document(metadata_document(image_et=170002800.0), SOURCE)
    assert rows.image['image_date'] == '2005-05-22'


def test_an_image_that_never_loaded_records_no_epoch() -> None:
    """No provenance block at all, which is what a fatal load error writes.

    An epoch is the observation's midtime, so a document written because the
    observation never loaded has none, and a row with no epoch is what the
    index then holds.  The image is placed nowhere in time, so a date bound
    passes it over.
    """
    document = metadata_document(status='error', status_error='missing_spice_data', offset=None)
    document['navigation_result'].pop('provenance')
    rows = facts_from_document(document, SOURCE)
    assert rows.image['image_et'] is None


def test_an_image_that_never_loaded_has_no_derived_date() -> None:
    """The date column that a ``--start-date`` bound compares against is NULL."""
    document = metadata_document(status='error', status_error='missing_spice_data', offset=None)
    document['navigation_result'].pop('provenance')
    rows = facts_from_document(document, SOURCE)
    assert rows.image['image_date'] is None


def test_the_rotation_columns_are_read() -> None:
    """A twist-fitted result records a rotation, and the index carries it."""
    document = metadata_document()
    document['navigation_result']['rotation_deg'] = 0.125
    document['navigation_result']['sigma_rotation_deg'] = 0.004
    rows = facts_from_document(document, SOURCE)
    assert rows.image['rotation_deg'] == 0.125


def test_the_image_number_is_ingested() -> None:
    """The range filter compares a column, so the number is computed here."""
    rows = facts_from_document(metadata_document(), SOURCE)
    assert rows.image['image_number'] == 1454725799


def test_the_file_metrics_come_from_the_walk() -> None:
    """The incremental skip compares these two against the next listing."""
    rows = facts_from_document(metadata_document(), SOURCE)
    assert (rows.image['mtime_ns'], rows.image['size_bytes']) == (
        SOURCE.mtime_ns,
        SOURCE.size_bytes,
    )


def test_the_technique_flags_are_booleans() -> None:
    """A boolean column holds a boolean; an integer flag is a type error later."""
    document = metadata_document(
        per_technique=[technique('BodyLimbNav', (1.0, 1.0), spurious=True)]
    )
    rows = facts_from_document(document, SOURCE)
    assert rows.techniques[0]['spurious'] is True


def test_the_child_rows_carry_the_image_key() -> None:
    """A child row names the image by the pair that keys it."""
    document = metadata_document(per_technique=[technique('BodyLimbNav', (1.0, 1.0))])
    rows = facts_from_document(document, SOURCE)
    assert rows.techniques[0]['results_path_stub'] == SOURCE.results_path_stub


def test_the_feature_inventory_is_aggregated() -> None:
    """Per-feature detail is not retained; the counts per source are."""
    rows = facts_from_document(metadata_document(), SOURCE)
    gated = next(row for row in rows.feature_sources if row['source_model'] == 'stars')
    assert gated['n_gated'] == 1


def test_a_techniques_source_names_come_back_sorted() -> None:
    """The names are collected into a set, whose order is neither of the two useful ones.

    It is not the order the feature ids were recorded in, and it is not sorted
    order.  Both storages hand this list to one consumer, so the order it holds
    has to be one the two can both produce, and a sorted list of distinct names
    is that order.
    """
    entry = technique('BodyLimbNav', (1.0, 1.0))
    entry['feature_ids'] = [
        'body_disc:TITAN',
        'body_disc:RHEA',
        'ring_edge:SATURN:feature_135_ieg:IEG',
        'body_disc:ENCELADUS',
        'body_disc:MIMAS',
        'body_disc:DIONE',
    ]
    rows = facts_from_document(metadata_document(per_technique=[entry]), SOURCE)
    assert rows.techniques[0]['source_names'] == [
        'DIONE',
        'ENCELADUS',
        'MIMAS',
        'RHEA',
        'SATURN',
        'TITAN',
    ]


def test_the_feature_source_rows_come_back_in_key_order() -> None:
    """The inventory is aggregated into a mapping, which holds the recorded order.

    A consumer comparing two images' inventories, or a run comparing the two
    storages against each other, compares lists; the recorded order is the
    inventory's rather than the image's, so the rows are ordered by the key that
    identifies them instead.
    """
    document = metadata_document()
    document['navigation_result']['feature_inventory'] = [
        {
            'feature_id': 'star:UCAC4:10230452',
            'feature_type': 'STAR',
            'source_model': 'stars',
            'gated': False,
        },
        {
            'feature_id': 'body_disc:TITAN',
            'feature_type': 'BODY_DISC',
            'source_model': 'body:TITAN',
            'gated': False,
        },
        {
            'feature_id': 'ring_edge:SATURN:feature_135_ieg:IEG',
            'feature_type': 'RING_EDGE',
            'source_model': 'rings',
            'gated': False,
        },
    ]
    rows = facts_from_document(document, SOURCE)
    assert [
        (row['feature_type'], row['source_model'], row['source_name'])
        for row in rows.feature_sources
    ] == [
        ('BODY_DISC', 'body:TITAN', 'TITAN'),
        ('RING_EDGE', 'rings', 'SATURN'),
        ('STAR', 'stars', 'UCAC4'),
    ]


def test_the_corrected_pointing_columns_are_read_when_present() -> None:
    """No document in the tree carries these yet, so a fixture exercises them."""
    document = metadata_document()
    document['navigation_result']['times'] = {
        'start_et': 170000000.5,
        'stop_et': 170000002.5,
        'exposure_s': 2.0,
        'sclk_start': '1/1294561202.100',
        'sclk_midtime': '1/1294561203.100',
        'sclk_stop': '1/1294561204.100',
    }
    document['navigation_result']['pointing'] = {
        'camera_frame_id': -82360,
        'ck_frame_id': -82000,
        'cmatrix': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        'cmatrix_original': [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    }
    rows = facts_from_document(document, SOURCE)
    assert rows.image['sclk_midtime'] == '1/1294561203.100'


def test_a_corrected_pointing_matrix_is_stored_as_nine_floats() -> None:
    """The producer writes a row-major matrix, and the column holds one."""
    document = metadata_document()
    document['navigation_result']['pointing'] = {
        'cmatrix': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    }
    rows = facts_from_document(document, SOURCE)
    assert rows.image['cmatrix'] == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def test_the_offset_column_holds_what_a_reader_would_apply() -> None:
    """Exactly the pair a consumer applies, in the order it applies it."""
    document = metadata_document()
    document['offset'] = [1.5, -2.5]
    rows = facts_from_document(document, SOURCE)
    assert (rows.image['offset_dv'], rows.image['offset_du']) == (1.5, -2.5)


def test_a_document_with_no_pointing_stores_nulls() -> None:
    """An image that never navigated has no corrected attitude to record."""
    rows = facts_from_document(metadata_document(), SOURCE)
    assert rows.image['cmatrix'] is None


def test_a_camera_frame_id_is_read_as_an_integer() -> None:
    """The frame identifiers are integers, and a boolean is not one."""
    document = metadata_document()
    document['navigation_result']['pointing'] = {'camera_frame_id': True}
    rows = facts_from_document(document, SOURCE)
    assert rows.image['camera_frame_id'] is None


@pytest.mark.parametrize('field', ['camera_frame_id', 'ck_frame_id'])
def test_a_frame_id_written_as_a_float_is_not_an_identifier(field: str) -> None:
    """A frame identity is an integer, so ``-82360.0`` is not one.

    JSON has one number type, so a document is free to write any number as a
    float; that does not make a float an identifier.  Converting one to the
    integer it denotes would store an identifier the document did not record,
    and this column exists to say what a frame kernel calls the frame.

    Parameters:
        field: Which of the two identifier fields is under test.
    """
    document = metadata_document()
    document['navigation_result']['pointing'] = {field: -82360.0}
    rows = facts_from_document(document, SOURCE)
    assert rows.image[field] is None


def test_the_shutter_mode_is_stored_as_recorded() -> None:
    """Two cameras exposed together share one bus attitude, and this says so.

    A kernel writer pairs the two and keeps one correction; a column that did
    not carry the mode would leave it pairing on the camera name alone, which
    every image of that camera would match.
    """
    document = metadata_document()
    document['observation']['shutter_mode'] = 'BOTSIM'
    rows = facts_from_document(document, SOURCE)
    assert rows.image['shutter_mode'] == 'BOTSIM'


def test_a_document_recording_no_shutter_mode_stores_none() -> None:
    """Most datasets record none, and a missing mode pairs nothing."""
    rows = facts_from_document(metadata_document(), SOURCE)
    assert rows.image['shutter_mode'] is None


def test_the_recorded_kernels_are_stored_in_the_order_recorded() -> None:
    """A correction overlays one original kernel, named among these."""
    document = metadata_document()
    document['navigation_result']['provenance']['spice_kernels'] = [
        'cas00172.tsc',
        'naif0012.tls',
    ]
    rows = facts_from_document(document, SOURCE)
    assert rows.image['spice_kernels'] == ['cas00172.tsc', 'naif0012.tls']


def test_a_run_that_recorded_no_kernels_stores_the_empty_list() -> None:
    """An empty list is a statement about the run, and its reader refuses it."""
    document = metadata_document()
    document['navigation_result']['provenance']['spice_kernels'] = []
    rows = facts_from_document(document, SOURCE)
    assert rows.image['spice_kernels'] == []


def test_the_camera_frame_name_is_stored_as_recorded() -> None:
    """A kernel writer looks the name up among the frame kernels it furnishes."""
    document = metadata_document()
    document['navigation_result']['pointing'] = {'camera_frame': 'CASSINI_ISS_NAC'}
    rows = facts_from_document(document, SOURCE)
    assert rows.image['camera_frame'] == 'CASSINI_ISS_NAC'


@pytest.mark.parametrize(
    ('recorded', 'stored'),
    [
        (0, 0),
        (1, 1),
        (2**63 - 1, 2**63 - 1),
        (2**63, None),
        (2**70, None),
        (-1, None),
        (1.5, None),
        (True, None),
        ('4096', None),
        (None, None),
    ],
)
def test_a_peak_memory_the_column_cannot_hold_is_stored_as_absent(
    recorded: Any, stored: int | None
) -> None:
    """The column is a signed 64-bit integer, so the reader has to be too.

    No kernel publishes a peak above that, so a document carrying one says the
    document was edited or damaged.  Passing it through would fail the insert
    and end the ingest of every image behind it, in place of the one absent
    measurement it actually is.

    Parameters:
        recorded: The value as the document carries it.
        stored: What the column should hold for it.
    """
    document = metadata_document()
    document['timing']['peak_memory_bytes'] = recorded
    rows = facts_from_document(document, SOURCE)
    assert rows.image['peak_memory_bytes'] == stored


def test_a_document_without_an_image_name_is_refused() -> None:
    """This is what a file that is not a navigation document looks like."""
    with pytest.raises(MetadataDocumentError, match=r'no observation\.image_name'):
        facts_from_document({'observation': {}}, SOURCE)


def test_a_document_without_an_instrument_is_refused() -> None:
    """Half a document is no more ingestible than none."""
    with pytest.raises(MetadataDocumentError, match=r'no observation\.instrument'):
        facts_from_document(metadata_document(instrument=None), SOURCE)


def test_a_refusal_names_the_file() -> None:
    """A run that meets hundreds of these has to be able to name each one."""
    with pytest.raises(MetadataDocumentError, match=SOURCE.source_file):
        facts_from_document({'observation': {}}, SOURCE)


def test_a_refusal_carries_the_reason_without_the_file() -> None:
    """The reason is tallied across files, so it may not carry a file name."""
    with pytest.raises(MetadataDocumentError) as caught:
        facts_from_document({'observation': {}}, SOURCE)
    assert caught.value.reason == (
        'not a current-schema navigation document (no observation.image_name)'
    )


def test_the_excluded_set_is_stored_as_the_document_wrote_it() -> None:
    """A recorded list is stored as recorded, in the order it was recorded in.

    Sorting it here would be a second reading rule over a value the document
    already fixed, so a consumer comparing the column against the document it
    came from would find two different lists.  A consumer wanting some other
    order sorts a list of three names itself.
    """
    document = metadata_document(excluded=['StarRefineNav', 'BodyBlobNav', 'RingEdgeNav'])
    rows = facts_from_document(document, SOURCE)
    assert rows.image['excluded_from_consensus'] == [
        'StarRefineNav',
        'BodyBlobNav',
        'RingEdgeNav',
    ]


def test_an_excluded_set_holding_anything_but_names_is_refused() -> None:
    """A list of something other than technique names is another kind of file."""
    document = metadata_document()
    document['navigation_result']['excluded_from_consensus'] = ['BodyBlobNav', 7]
    with pytest.raises(MetadataDocumentError, match='holds a name that is not a string'):
        facts_from_document(document, SOURCE)


def test_an_empty_excluded_set_is_stored_as_the_empty_list() -> None:
    """An empty list is a statement: the ensemble excluded nothing."""
    rows = facts_from_document(metadata_document(excluded=[]), SOURCE)
    assert rows.image['excluded_from_consensus'] == []


# ---------------------------------------------------------------------------
# A fault in this code is not a fault of the document
# ---------------------------------------------------------------------------


class _NobodyEnumeratedThisError(Exception):
    """An exception type this reader has no way to name.

    Defined here and imported nowhere, so nothing can be catching it by name.
    """


def test_a_fault_in_the_reader_is_not_turned_into_a_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refusal is recorded against the file with its own modification time and size.

    So a refusal outlives the pass that wrote it, and a fault in this code
    recorded as one would outlive its own fix while every later pass reported a
    clean run.  It escapes instead.

    Parameters:
        monkeypatch: Fixture the reader's own helper is replaced through.
    """

    def exploding(_value: Any) -> Any:
        raise _NobodyEnumeratedThisError('a fault in this code')

    monkeypatch.setattr(facts_module, '_str_or_none', exploding)
    with pytest.raises(_NobodyEnumeratedThisError, match='a fault in this code'):
        facts_from_document(metadata_document(), SOURCE)
