import csv
import json
from pathlib import Path
from typing import Any, cast

import pdstemplate
from filecache import FCPath
from pdslogger import PdsLogger

from spindoctor.cli.pds4.labels import write_label
from spindoctor.dataset.dataset import DataSet


def generate_collection_files(
    bundle_results_root: FCPath,
    dataset: DataSet,
    logger: PdsLogger,
) -> int:
    """Generate collection CSV and label files for the bundle.

    Every collection label is attempted, whichever of them fail: a broken data
    collection template must not hide a broken browse collection one.  The
    inventory tables are written whether or not the labels that describe them
    render.

    Every collection template the dataset declares is required.  The caller is
    expected to have checked them before processing anything, so one that is
    missing here raises rather than being passed over.

    Parameters:
        bundle_results_root: Root directory of the bundle. The bundle data directory
            will be scanned for all backplane label files.
        dataset: The dataset instance for bundle-specific methods.
        logger: Logger for diagnostic messages.

    Returns:
        The number of collection labels that could not be rendered.

    Raises:
        FileNotFoundError: If the bundle has no data directory to scan, or a
            collection template is not in the dataset's template directory.
    """

    bundle_name = dataset.pds4_bundle_name()
    template_dir = dataset.pds4_bundle_template_dir()
    bundle_root = bundle_results_root / bundle_name
    failed_labels = 0

    # Scan for all label files in data directory
    data_dir = bundle_root / 'data'
    label_files: list[FCPath] = []

    if not data_dir.exists():
        raise FileNotFoundError(f'Data directory does not exist: {data_dir}')

    # Recursively scan for .lblx files
    for label_file in data_dir.rglob('*_backplanes.lblx'):
        label_files.append(label_file)

    # Sort by image name (extracted from filename)
    def get_image_name_from_label(path: FCPath) -> str:
        # Extract image name from filename
        # (e.g., "1234567890w_backplanes.lblx" -> "1234567890w")
        name = path.stem
        if '_backplanes' in name:
            return name.split('_backplanes')[0]
        return name

    label_files.sort(key=get_image_name_from_label)
    logger.info('Found %d label files in bundle', len(label_files))

    # Generate collection_data.tab
    collection_data_csv = bundle_root / 'data' / 'collection_data.tab'
    collection_data_local = cast(Path, collection_data_csv.get_local_path())
    collection_data_local.parent.mkdir(parents=True, exist_ok=True)
    with collection_data_local.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Member Status', 'LIDVID_LID'])
        for label_file in label_files:
            lid_part = label_file.stem.replace('_backplanes', '')
            image_name = dataset.pds4_lid_part_to_image_name(lid_part)
            lidvid = dataset.pds4_image_name_to_data_lidvid(image_name)
            writer.writerow(['P', lidvid])
    collection_data_csv.upload()
    logger.info('Generated "collection_data.tab": %s', collection_data_csv)

    # Generate collection label files using template
    template_base = Path(template_dir)

    # Collection data label
    collection_data_template = template_base / 'collection_data.lblx'
    template = pdstemplate.PdsTemplate(str(collection_data_template))
    collection_data_label = bundle_root / 'data' / 'collection_data.lblx'
    template_vars = {
        'COLLECTION_DATA_CSV_PATH': str(collection_data_csv),
        'EARLIEST_START_DATE_TIME': '',  # TODO: Calculate from all images
        'LATEST_STOP_DATE_TIME': '',  # TODO: Calculate from all images
    }
    if write_label(template, template_vars, collection_data_label, logger=logger):
        logger.info('Generated "collection_data.lblx"')
    else:
        failed_labels += 1

    # Generate collection_browse.tab (must be written before collection_browse.lblx)
    collection_browse_csv = bundle_root / 'browse' / 'collection_browse.tab'
    collection_browse_local = cast(Path, collection_browse_csv.get_local_path())
    collection_browse_local.parent.mkdir(parents=True, exist_ok=True)
    with collection_browse_local.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Member Status', 'LIDVID_LID'])
        for label_file in label_files:
            lid_part = label_file.stem.replace('_backplanes', '')
            image_name = dataset.pds4_lid_part_to_image_name(lid_part)
            lidvid = dataset.pds4_image_name_to_browse_lidvid(image_name)
            writer.writerow(['P', lidvid])
    collection_browse_csv.upload()
    logger.info('Generated "collection_browse.tab": %s', collection_browse_csv)

    # Collection browse label
    collection_browse_template = template_base / 'collection_browse.lblx'
    template = pdstemplate.PdsTemplate(str(collection_browse_template))
    collection_browse_label = bundle_root / 'browse' / 'collection_browse.lblx'
    template_vars = {
        'COLLECTION_BROWSE_CSV_PATH': str(collection_browse_csv),
    }
    if write_label(template, template_vars, collection_browse_label, logger=logger):
        logger.info('Generated "collection_browse.lblx"')
    else:
        failed_labels += 1

    logger.info('Generated collection files: %d products', len(label_files))
    return failed_labels


def generate_global_index_files(
    bundle_results_root: FCPath,
    dataset: DataSet,
    logger: PdsLogger,
) -> int:
    """Generate global index files for bodies and rings.

    Both index labels are attempted, whichever of them fail, and the index
    tables are written whether or not the labels that describe them render.

    Both index templates the dataset declares are required.  The caller is
    expected to have checked them before processing anything, so one that is
    missing here raises rather than being passed over.

    Parameters:
        bundle_results_root: Root directory of the bundle. The bundle data directory
            will be scanned for all supplemental text files.
        dataset: The dataset instance for bundle-specific methods.
        logger: Logger for diagnostic messages.

    Returns:
        The number of index labels that could not be rendered.

    Raises:
        FileNotFoundError: If an index template is not in the dataset's template
            directory.
    """

    bundle_name = dataset.pds4_bundle_name()
    template_dir = dataset.pds4_bundle_template_dir()
    bundle_root = bundle_results_root / bundle_name
    config = dataset.config
    failed_labels = 0

    # Get configured backplane types from config
    bodies_cfg = getattr(config.backplanes, 'bodies', [])
    body_backplane_types = [bp['name'] for bp in bodies_cfg]
    rings_cfg = getattr(config.backplanes, 'rings', [])
    ring_backplane_types = [bp['name'] for bp in rings_cfg]

    # Scan for all supplemental files
    supplemental_files: list[FCPath] = []
    data_dir = bundle_root / 'data'
    for suppl_file in data_dir.rglob('*_supplemental.txt'):
        supplemental_files.append(suppl_file)

    # Sort by image name (extracted from filename)
    def get_image_name_from_supplemental(path: FCPath) -> str:
        # Extract image name from filename
        # (e.g., "1234567890w_supplemental.txt" -> "1234567890w")
        return path.name.replace('_supplemental.txt', '')

    supplemental_files.sort(key=get_image_name_from_supplemental)
    logger.info('Found %d supplemental files', len(supplemental_files))

    # Collect body and ring statistics
    body_index_rows: list[dict[str, Any]] = []
    ring_index_rows: list[dict[str, Any]] = []

    for suppl_file in supplemental_files:
        try:
            suppl_text = suppl_file.read_text()
            metadata = json.loads(suppl_text)
        except Exception:
            logger.exception('Error reading supplemental file %s', suppl_file)
            # TODO Should we continue here?
            continue

        backplanes = metadata.get('backplanes', {})
        bodies = backplanes.get('bodies', {})
        rings = backplanes.get('rings', {})

        # Derive pds4_path_stub from supplemental file path
        # Supplemental file is at: bundle_root/data/<pds4_path_stub>_supplemental.txt
        suppl_relative = suppl_file.relative_to(data_dir)
        pds4_path_stub = str(suppl_relative).replace('_supplemental.txt', '')
        pds4_path_stub = pds4_path_stub.replace('\\', '/')

        lid_part = suppl_file.stem.replace('_supplemental', '')
        image_name = dataset.pds4_lid_part_to_image_name(lid_part)
        lid = dataset.pds4_image_name_to_data_lid(image_name)
        # pds4_path_stub includes path and filename prefix
        # Path relative to data directory
        path_to_image = f'data/{pds4_path_stub}_backplanes.lblx'

        # Body index: one line per image per body
        for body_name, body_data in bodies.items():
            body_row: dict[str, Any] = {
                'LID': lid,
                'body_name': body_name,
                'path_to_image_file': path_to_image,
            }
            body_backplanes = body_data.get('backplanes', {})
            # Add min/max columns for each configured backplane type
            for bp_type in body_backplane_types:
                bp_values = body_backplanes.get(bp_type, {})
                body_row[f'{bp_type}_min'] = bp_values.get('min')
                body_row[f'{bp_type}_max'] = bp_values.get('max')
            body_index_rows.append(body_row)

        # Ring index: one line per image
        ring_backplanes = rings.get('backplanes', {})
        if ring_backplanes:
            ring_row: dict[str, Any] = {
                'LID': lid,
                'path_to_image_file': path_to_image,
            }
            # Add min/max columns for each configured ring backplane type
            for ring_type in ring_backplane_types:
                ring_values = ring_backplanes.get(ring_type, {})
                ring_row[f'{ring_type}_min'] = ring_values.get('min', '')
                ring_row[f'{ring_type}_max'] = ring_values.get('max', '')
            ring_index_rows.append(ring_row)

    # Generate global_index_bodies.tab
    supplemental_dir = bundle_root / 'document' / 'supplemental'
    bodies_tab = supplemental_dir / 'global_index_bodies.tab'
    bodies_tab_local = cast(Path, bodies_tab.get_local_path())
    with bodies_tab_local.open('w', newline='') as f:
        writer = csv.writer(f)
        # Build header: LID, body_name, path_to_image_file, then min/max for each backplane type
        header = ['LID', 'body_name', 'path_to_image_file']
        for bp_type in body_backplane_types:
            header.append(f'{bp_type}_min')
            header.append(f'{bp_type}_max')
        writer.writerow(header)

        for row in body_index_rows:
            row_data = [row['LID'], row['body_name'], row['path_to_image_file']]
            for bp_type in body_backplane_types:
                # TODO Need an appropriate sentinel value for missing data
                min_val = row.get(f'{bp_type}_min', '')
                max_val = row.get(f'{bp_type}_max', '')
                if isinstance(min_val, (int, float)):
                    min_val = f'{min_val:.5f}'
                if isinstance(max_val, (int, float)):
                    max_val = f'{max_val:.5f}'
                row_data.append(min_val)
                row_data.append(max_val)
            writer.writerow(row_data)
    bodies_tab.upload()
    logger.info('Generated global_index_bodies.tab with %d rows', len(body_index_rows))

    # Generate global_index_rings.tab
    rings_tab = supplemental_dir / 'global_index_rings.tab'
    rings_tab_local = cast(Path, rings_tab.get_local_path())
    # No explicit parent mkdir: get_local_path() creates parents (matching the
    # bodies index above), so an extra mkdir here was redundant and asymmetric.
    with rings_tab_local.open('w', newline='') as f:
        writer = csv.writer(f)
        # Build header: LID, path_to_image_file, then min/max for each ring type
        header = ['LID', 'path_to_image_file']
        # TODO Add planet name to rings table
        for ring_type in ring_backplane_types:
            header.append(f'{ring_type}_min')
            header.append(f'{ring_type}_max')
        writer.writerow(header)

        for row in ring_index_rows:
            row_data = [row['LID'], row['path_to_image_file']]
            for ring_type in ring_backplane_types:
                min_val = row.get(f'{ring_type}_min', '')
                max_val = row.get(f'{ring_type}_max', '')
                if isinstance(min_val, (int, float)):
                    min_val = f'{min_val:.5f}'
                if isinstance(max_val, (int, float)):
                    max_val = f'{max_val:.5f}'
                row_data.append(min_val)
                row_data.append(max_val)
            writer.writerow(row_data)
    rings_tab.upload()
    logger.info('Generated global_index_rings.tab with %d rows', len(ring_index_rows))

    # Generate label files using templates
    template_base = Path(template_dir)

    # Global index bodies label
    bodies_template = template_base / 'global_index_bodies.lblx'
    template = pdstemplate.PdsTemplate(str(bodies_template))
    bodies_label = supplemental_dir / 'global_index_bodies.lblx'
    template_vars = {
        'FILE_RECORDS': len(body_index_rows),
    }
    if write_label(template, template_vars, bodies_label, logger=logger):
        logger.info('Generated global_index_bodies.lblx')
    else:
        failed_labels += 1

    # Global index rings label
    rings_template = template_base / 'global_index_rings.lblx'
    template = pdstemplate.PdsTemplate(str(rings_template))
    rings_label = supplemental_dir / 'global_index_rings.lblx'
    template_vars = {
        'FILE_RECORDS': len(ring_index_rows),
    }
    if write_label(template, template_vars, rings_label, logger=logger):
        logger.info('Generated global_index_rings.lblx')
    else:
        failed_labels += 1

    logger.info(
        'Generated global index files: %d body rows, %d ring rows',
        len(body_index_rows),
        len(ring_index_rows),
    )
    return failed_labels
