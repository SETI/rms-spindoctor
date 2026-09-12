"""What the Galileo SSI host publishes for the metadata chapter's failed example.

The chapter's failed example is a real navigation of ``C0059750900R.IMG``, and its
``observation`` block shows what
:meth:`~spindoctor.obs.obs_inst_galileo_ssi.ObsGalileoSSI.get_public_metadata` returned
for that image, loaded through the host's own reader.  The same facts are held here, in
the host's own key order and value types, so that a document the writer builds from them
can be held against the example.
"""

from typing import Any

GALILEO_SSI_PUBLIC_METADATA: dict[str, Any] = {
    'image_path': '/holdings/volumes/GO_0xxx/GO_0002/RAW_CAL/C0059750900R.IMG',
    'image_name': 'C0059750900R.IMG',
    'instrument_host_lid': 'urn:nasa:pds:context:instrument_host:spacecraft.go',
    'instrument_lid': 'urn:nasa:pds:context:instrument:go.ssi',
    'start_time_utc': '1990-11-29T22:27:07.068',
    'midtime_utc': '1990-11-29T22:27:07.071',
    'end_time_utc': '1990-11-29T22:27:07.074',
    'start_time_et': -286810315.74894506,
    'midtime_et': -286810315.74582005,
    'end_time_et': -286810315.74269503,
    'image_shape_xy': (800, 800),
    'camera': 'SSI',
    'exposure_time': 0.00625,
    'filters': ['GREEN'],
}
"""The facts, as the host returned them; the path's site-specific prefix is abbreviated."""
