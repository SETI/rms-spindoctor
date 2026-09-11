import json
import shutil
from enum import Enum
from pathlib import Path
from typing import Any, cast

import pdstemplate
from filecache import FCPath
from pdslogger import PdsLogger

from spindoctor.cli.pds4.data_objects import configured_methods, describe_backplane_fits
from spindoctor.cli.pds4.epochs import unrecorded_epoch
from spindoctor.cli.pds4.labels import write_label
from spindoctor.cli.pds4.statistic_checks import unindexable_statistic
from spindoctor.dataset.dataset import DataSet, ImageFiles
from spindoctor.support.file import json_as_string


class BundleDataOutcome(Enum):
    """What generating one image's bundle data products came to.

    Attributes:
        WRITTEN: Every label the image calls for is on disk.
        SKIPPED: The image has no products for the bundle to describe, because
            it was not navigated or because its backplanes were never
            generated.  A skip is an image the bundle has nothing to say about,
            not a failure of the run.
        FAILED: At least one of the image's products is not in the bundle: a
            label that could not be rendered, a browse product whose summary
            PNG the navigation results do not hold, or, with nothing written
            for the image at all, backplane metadata recording a statistic no
            global index column can hold -- one in a unit other than the one
            the configuration gives its plane, or with a minimum or maximum
            that is not a finite number within the range of a float -- or a
            navigation document that does not record the exposure's start,
            stop and midtime as finite numbers, the stop no earlier than the
            start.
    """

    WRITTEN = 'written'
    SKIPPED = 'skipped'
    FAILED = 'failed'


def generate_bundle_data_files(
    dataset: DataSet,
    image_files: ImageFiles,
    *,
    nav_results_root: FCPath,
    backplane_results_root: FCPath,
    bundle_results_root: FCPath,
    logger: PdsLogger,
) -> BundleDataOutcome:
    """Generate PDS4 bundle data files for a single image batch.

    Both the data label and the browse label are attempted even when the first
    of them fails, so one run reports every label it could not render rather
    than one per run.  Whichever label did render stays on disk.

    An image the bundle has nothing to describe is skipped rather than failed:
    a navigation document that is not there, a navigation that did not succeed,
    and backplane metadata that is not there are all cases of a selection
    naming more images than the bundle covers, which is the ordinary state of a
    selection made by volume.  A document that is there but cannot be read is
    not one of them, and still raises.

    A navigated image whose summary PNG is not in the navigation results is not
    one of them either.  The navigation stage writes that PNG before, and under
    the same condition as, the document that records the success, so a success
    document with no PNG beside it is a broken input rather than an image
    without a browse product; the image is failed, and its data label stays on
    disk.

    A navigated image whose backplane metadata records a statistic no global
    index column can hold is failed as well, before anything is written for it.
    A statistic in a unit other than the one the configuration gives its plane,
    or in none, was written before the statistics recorded their unit, or under
    another configuration, and indexing it would put one column in two units
    with nothing saying so.  A minimum or maximum that is not a finite number
    within the range of a float has no decimal form a column can hold, and a
    blank in its place would say the plane measured nothing.  A plane the
    document holds that the configuration does not declare is not checked.

    So is a navigated image whose navigation document does not record its
    exposure's epochs -- a ``start_et``, a ``stop_et`` and a ``midtime_et`` under
    ``navigation_result.times``, each a finite number, the stop no earlier than
    the start -- again before anything is written for it.  Its data label states
    when the exposure began and ended, and the navigation stamps a success
    document with both, so a success document that records neither is a broken
    input rather than an image whose time is unknown, and a label stating an
    empty time is not one PDS4 accepts.  The log names the image and what the
    document lacks.

    The backplane FITS is copied into the bundle, beside its data label, which names
    it with no directory part, and the label's size, checksum and time are the
    copy's.

    The data label describes every HDU of the FITS, through
    :func:`~spindoctor.cli.pds4.data_objects.describe_backplane_fits`, which reads the
    source before the copy is made; the copy is byte-identical, so the source's
    description is the copy's.

    Parameters:
        dataset: The dataset instance to get bundle-specific methods from.
        image_files: List of images; must have exactly one image in the batch.
        nav_results_root: Root containing navigation metadata JSONs and summary PNGs.
        backplane_results_root: Root containing backplane FITS files and metadata JSONs.
        bundle_results_root: Destination root for bundle files.
        logger: Logger for diagnostic messages.

    Returns:
        WRITTEN when the image's labels are on disk, SKIPPED when the image has
        nothing for the bundle to describe, and FAILED when a label could not be
        rendered, the summary PNG is not there, a backplane statistic is in a
        unit other than the one the configuration gives its plane or has a
        minimum or maximum that is not a finite number within the range of a
        float, or the navigation document does not record the exposure's epochs.

    Raises:
        ValueError: If the batch does not hold exactly one image, or if the
            configuration entry of a plane the backplane metadata holds declares
            a blank unit.
        TypeError: If that entry declares no unit, or one that is not a string.
        OSError: If the backplane FITS cannot be read or copied.
    """

    if len(image_files.image_files) != 1:
        raise ValueError(
            f'Expected exactly one image per batch; got {len(image_files.image_files)}'
        )

    image_file = image_files.image_files[0]
    image_path = image_file.image_file_path.absolute()
    results_path_stub = image_file.results_path_stub

    metadata_file = nav_results_root / (results_path_stub + '_metadata.json')
    backplane_metadata_file = backplane_results_root / (
        results_path_stub + '_backplane_metadata.json'
    )

    with logger.open(f'Generating PDS4 bundle data files for {image_path!s}'):
        # The record the enumeration already read, when there is one.  A run
        # whose selection names an error filter has retrieved and parsed the
        # document of every image it kept in order to decide, and the
        # enumeration hands that record on with the image; the two readers hold
        # caches of their own, so reading it again is a second download of the
        # same file on a cloud results root rather than a second look at one
        # already local.
        nav_metadata = image_file.nav_record
        if nav_metadata is None:
            try:
                metadata_text = metadata_file.read_text()
            except FileNotFoundError:
                # An image with no navigation document was never navigated,
                # which is what the status branch below reports in the other
                # spelling: no record of a navigation, rather than a record of a
                # navigation that did not succeed.
                logger.warning(
                    'Skipping bundle generation for "%s": no navigation metadata at %s',
                    image_path,
                    metadata_file,
                )
                return BundleDataOutcome.SKIPPED
            nav_metadata = cast(dict[str, Any], json.loads(metadata_text))

        status = nav_metadata.get('status', None)
        if status != 'success':
            # TODO Figure out what to do with non-navigated images
            logger.warning(
                'Skipping bundle generation for "%s": status=%s error=%s',
                image_path,
                status,
                nav_metadata.get('status_error', 'unknown'),
            )
            return BundleDataOutcome.SKIPPED

        # Read backplane metadata
        try:
            backplane_metadata_text = backplane_metadata_file.read_text()
        except FileNotFoundError:
            # A navigated image whose backplanes were never generated has
            # nothing a backplanes bundle can describe.
            logger.warning(
                'Skipping bundle generation for "%s": no backplane metadata at %s',
                image_path,
                backplane_metadata_file,
            )
            return BundleDataOutcome.SKIPPED
        bp_stats = cast(dict[str, Any], json.loads(backplane_metadata_text))

        # A backplane root can hold documents written before the statistics
        # recorded their unit, or under a configuration declaring another,
        # beside regenerated ones, and a statistic can be a value that has no
        # decimal form.  Indexing either would put something in a column of the
        # global index that the column cannot say, so the image is failed
        # before anything is written for it.
        unindexable = unindexable_statistic(bp_stats, dataset.config)
        if unindexable is not None:
            logger.error(
                'Failing bundle generation for "%s": the backplane metadata %s; %s. '
                'Nothing is written for the image until its backplanes are regenerated '
                'with statistics an index column can hold',
                image_path,
                unindexable.description,
                unindexable.reason,
            )
            return BundleDataOutcome.FAILED

        # A data label states when its exposure began and ended, and a success
        # document records both, so one that does not is a broken input.  The
        # image is failed before anything is written for it rather than labeled
        # with an empty time or left to raise from the template variables.
        unrecorded = unrecorded_epoch(nav_metadata)
        if unrecorded is not None:
            logger.error(
                'Failing bundle generation for "%s": the navigation metadata %s. Nothing is '
                'written for the image, whose data label states when its exposure began '
                'and ended',
                image_path,
                unrecorded,
            )
            return BundleDataOutcome.FAILED

        fits_source_path = backplane_results_root / (results_path_stub + '_backplanes.fits')
        fits_source_local = cast(Path, fits_source_path.retrieve())

        # The copy made below is byte-identical, so the source's description is the copy's.
        fits_objects = describe_backplane_fits(
            fits_source_local,
            masked_value=float(dataset.config.backplanes.masked_value),
            methods=configured_methods(dataset.config),
        )

        pds4_path_stub = dataset.pds4_path_stub(image_file)
        bundle_name = dataset.pds4_bundle_name()
        template_dir = dataset.pds4_bundle_template_dir()

        # TODO Clean up the nav metadata to only include the necessary fields

        # Combine metadata for supplemental file
        combined_metadata: dict[str, Any] = {
            'navigation': nav_metadata,
            'backplanes': bp_stats,
        }

        # Get template variables from dataset
        template_vars = dataset.pds4_template_variables(
            image_file=image_file,
            nav_metadata=nav_metadata,
            backplane_metadata=bp_stats,
        )

        # Determine output paths
        bundle_root = bundle_results_root / bundle_name
        data_dir = bundle_root / 'data'
        browse_dir = bundle_root / 'browse'
        label_file_path = data_dir / (pds4_path_stub + '_backplanes.lblx')
        fits_file_path = data_dir / (pds4_path_stub + '_backplanes.fits')
        suppl_file_path = data_dir / (pds4_path_stub + '_supplemental.txt')
        browse_label_path = browse_dir / (pds4_path_stub + '_summary.lblx')
        browse_image_path = browse_dir / (pds4_path_stub + '_summary.png')

        # The FITS goes into the bundle beside its label, which names it with no
        # directory part, and BACKPLANE_PATH names the copy, so that the size,
        # checksum and time the label states are the archived file's.  As with the
        # summary PNG below, the copy is written to a local path and uploaded, and
        # the label's FILE_* functions read BACKPLANE_PATH as a local file, so a
        # bundle root in the cloud is not handled here (#67).
        fits_file_local = cast(Path, fits_file_path.get_local_path())
        shutil.copy2(fits_source_local, fits_file_local)
        fits_file_path.upload()
        logger.info('Copied backplane FITS: %s', fits_file_path)

        # Add file path variables to template_vars
        summary_png_source = nav_results_root / (results_path_stub + '_summary.png')
        template_vars['BACKPLANE_FILENAME'] = fits_file_path.name
        template_vars['BACKPLANE_PATH'] = str(fits_file_path)
        template_vars['BACKPLANE_FITS'] = fits_objects
        template_vars['BACKPLANE_SUPPL_FILENAME'] = suppl_file_path.name
        template_vars['BACKPLANE_SUPPL_PATH'] = str(suppl_file_path)
        template_vars['BROWSE_FULL_FILENAME'] = browse_image_path.name
        template_vars['BROWSE_FULL_PATH'] = str(browse_image_path)

        # Generate supplemental file (JSON format) - must be written before template
        # The label declares the file 7-Bit ASCII Text with Line-Feed records.
        # json.dumps escapes every character outside ASCII and ends lines in a
        # line feed, and writing its bytes keeps them line feeds on a platform
        # whose text files end lines otherwise.
        suppl_file_path.write_bytes(json_as_string(combined_metadata).encode('ascii'))
        logger.info('Generated supplemental file: %s', suppl_file_path)

        # Generate PDS4 label file
        template_path = Path(template_dir) / 'data.lblx'
        template = pdstemplate.PdsTemplate(str(template_path))
        data_written = write_label(template, template_vars, label_file_path, logger=logger)
        if data_written:
            logger.info('Generated PDS4 label: %s', label_file_path)

        # Copy summary PNG to browse directory and generate browse label.
        # navigate_image_files writes the summary PNG before the navigation
        # document and under the same condition, so a success document always
        # has a PNG beside it.  One that does not is a broken input, not an
        # image with no browse product, and is failed rather than passed over.
        browse_written = False
        if summary_png_source.exists():
            # Copy the summary PNG file
            summary_png_local = cast(Path, summary_png_source.get_local_path())
            browse_image_local = cast(Path, browse_image_path.get_local_path())
            # TODO This needs to be updated for cloud storage
            browse_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(summary_png_local, browse_image_local)
            browse_image_path.upload()
            logger.info('Copied summary image: %s', browse_image_path)

            # Generate browse label
            browse_template_path = Path(template_dir) / 'browse.lblx'
            browse_template = pdstemplate.PdsTemplate(str(browse_template_path))
            browse_written = write_label(
                browse_template, template_vars, browse_label_path, logger=logger
            )
            if browse_written:
                logger.info('Generated browse label: %s', browse_label_path)
        else:
            logger.error(
                'No summary PNG at %s for "%s", whose navigation succeeded; the browse '
                'products for this image were not written',
                summary_png_source,
                image_path,
            )

        if data_written and browse_written:
            return BundleDataOutcome.WRITTEN
        return BundleDataOutcome.FAILED
