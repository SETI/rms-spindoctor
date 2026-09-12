"""What the Cassini ISS host publishes for the metadata chapter's success example.

The chapter's success example is a real navigation of ``N1635282917_1_CALIB.IMG``, and
its ``observation`` block shows what
:meth:`~spindoctor.obs.obs_inst_cassini_iss.ObsCassiniISS.get_public_metadata` returned
for that image, loaded through the host's own reader.  The same facts are held here, in
the host's own key order and value types, so that a document the writer builds from them
can be held against the example.
"""

from typing import Any

CASSINI_ISS_PUBLIC_METADATA: dict[str, Any] = {
    'image_path': (
        '/holdings/calibrated/COISS_2xxx/COISS_2058/data/1635278317_1635374244/'
        'N1635282917_1_CALIB.IMG'
    ),
    'image_name': 'N1635282917_1_CALIB.IMG',
    'instrument_host_lid': 'urn:nasa:pds:context:instrument_host:spacecraft.co',
    'instrument_lid': 'urn:nasa:pds:context:instrument:issna.co',
    'start_time_utc': '2009-10-26T20:32:22.024',
    'midtime_utc': '2009-10-26T20:32:22.134',
    'end_time_utc': '2009-10-26T20:32:22.244',
    'start_time_et': 309861208.2064568,
    'midtime_et': 309861208.3164568,
    'end_time_et': 309861208.4264568,
    'start_time_sclk': 1635282917.24609375,
    'midtime_sclk': 1635282917.353515625,
    'end_time_sclk': 1635282917.4609375,
    'image_shape_xy': (1024, 1024),
    'camera': 'NAC',
    'exposure_time': 0.22,
    'filters': ['CL1', 'CL2'],
    'sampling': 'FULL',
    'gain_mode': 2,
    'description': 'N/A',
    'observation_id': 'ISS_120RH_MUTUALEVE001_PRIME',
}
"""The facts as the host returned them, the path's site prefix abbreviated."""
