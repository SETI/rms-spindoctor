"""Shared test constants: external-image URLs and environment gates."""

import os

import pytest

# Tests importing these markers exercise real image reading against external
# data trees. Two tiers:
#
# REQUIRES_EXTERNAL_DATA needs the oops/SPICE resource tree (OOPS_RESOURCES)
# in addition to the PDS3 holdings. The resource tree is not publicly
# downloadable (bulk egress from its cloud bucket is prohibitively
# expensive, so public access is off), which is why CI does not set
# OOPS_RESOURCES: these tests run only locally, against a local copy of
# the oops resources directory.
#
# REQUIRES_PDS3_HOLDINGS needs only the PDS3 holdings tree, which is served
# by the PDS Ring-Moon Systems Node; CI sets PDS3_HOLDINGS_DIR, so these
# tests run in automated environments too.
REQUIRES_EXTERNAL_DATA = pytest.mark.skipif(
    'OOPS_RESOURCES' not in os.environ or 'PDS3_HOLDINGS_DIR' not in os.environ,
    reason='requires OOPS_RESOURCES and PDS3_HOLDINGS_DIR (external holdings/resources)',
)

REQUIRES_PDS3_HOLDINGS = pytest.mark.skipif(
    'PDS3_HOLDINGS_DIR' not in os.environ,
    reason='requires PDS3_HOLDINGS_DIR (PDS3 holdings)',
)

# TODO: Update to use PDS3_HOLDINGS_DIR

# Image of Rhea
# Camera: Cassini ISS WAC
# Size: 1024x1024
# Filter: VIO
# Exposure: 1.5 sec
# https://opus.pds-rings.seti.org/opus/#/view=detail&detail=co-iss-w1521598221
URL_CASSINI_ISS_RHEA_01 = 'https://pds-rings.seti.org/holdings/calibrated/COISS_2xxx/COISS_2021/data/1521584844_1521609901/W1521598221_1_CALIB.IMG'

# Image of Titan
# Camera: Cassini ISS WAC
# Size: 1024x1024
# Filter: CB3
# Exposure: 12 sec
# Note: RA wraps around
# https://opus.pds-rings.seti.org/#/view=detail&detail=co-iss-w1624353774
URL_CASSINI_ISS_TITAN_01 = 'https://pds-rings.seti.org/holdings/calibrated/COISS_2xxx/COISS_2055/data/1624240547_1624420949/W1624353774_1_CALIB.IMG'

# Image of a star field
# Camera: Cassini ISS WAC
# Size: 1024x1024
# Filter: CLEAR
# Exposure: 26 sec
# https://opus.pds-rings.seti.org/opus/#/view=detail&detail=co-iss-w1580760393
URL_CASSINI_ISS_STARS_01 = 'https://pds-rings.seti.org/holdings/calibrated/COISS_2xxx/COISS_2041/data/1580756433_1580830157/W1580760393_1_CALIB.IMG'

# Image of a star field
# Camera: Cassini ISS NAC
# Size: 1024x1024
# Filter: CLEAR
# Exposure: 1.0 sec
# https://opus.pds-rings.seti.org/#/view=detail&detail=co-iss-n1521881358
URL_CASSINI_ISS_STARS_02 = 'https://pds-rings.seti.org/holdings/calibrated/COISS_2xxx/COISS_2021/data/1521798868_1521893025/N1521881358_2_CALIB.IMG'

# Image from the earliest cruise volume, COISS_1001
# Camera: Cassini ISS NAC
# Exposure: 0.03 sec
# Its VICAR label marks its property section with a plain PROPERTY keyword rather
# than the numbered form, states fewer of the archive's observation keywords than a
# tour label does, and carries items of its own under different names.
URL_CASSINI_ISS_CRUISE_01 = 'https://pds-rings.seti.org/holdings/calibrated/COISS_1xxx/COISS_1001/data/1294561143_1295221348/N1294562651_1_CALIB.IMG'

# Image of Io
# Camera: Galileo SSI
# Size: 800x800 (cutout window 431x411)
# Filter: RED
# Exposure: 0.0625 sec
# https://opus.pds-rings.seti.org/opus/#/view=detail&detail=go-ssi-c0349673965
URL_GALILEO_SSI_IO_01 = (
    'https://pds-rings.seti.org/holdings/volumes/GO_0xxx/GO_0017/G1/IO/C0349673965R.IMG'
)

# Image of a star field
# Camera: Galileo SSI
# Size: 800x800
# Filter: Green
# Exposure: 0.0063 sec
# https://opus.pds-rings.seti.org/#/view=detail&detail=go-ssi-c0059750900
URL_GALILEO_SSI_STARS_01 = (
    'https://pds-rings.seti.org/holdings/volumes/GO_0xxx/GO_0002/RAW_CAL/C0059750900R.IMG'
)

# Image of a star field
# Camera: Galileo SSI
# Size: 800x800
# Filter: Clear
# Exposure: 0.1 sec
# https://opus.pds-rings.seti.org/#/view=detail&detail=go-ssi-c0059881700
URL_GALILEO_SSI_STARS_02 = (
    'https://pds-rings.seti.org/holdings/volumes/GO_0xxx/GO_0002/RAW_CAL/C0059881700R.IMG'
)

# Image of Charon
# Camera: New Horizons LORRI
# Size: 1024x1024
# Filter: N/A
# Exposure: 0.15 sec
# https://opus.pds-rings.seti.org/opus/#/view=detail&detail=nh-lorri-lor_0299147641
URL_NEWHORIZONS_LORRI_CHARON_01 = 'https://pds-rings.seti.org/holdings/volumes/NHxxLO_xxxx/NHPELO_2001/data/20150714_029914/lor_0299147641_0x630_sci.fit'

# Image of Io
# Camera: Voyager ISS
# Size: 800x800
# Filter: CLEAR
# Exposure: 0.12 sec
# https://opus.pds-rings.seti.org/opus/#/view=detail&detail=vg-iss-2-j-c2062133
URL_VOYAGER_ISS_IO_01 = 'https://pds-rings.seti.org/holdings/volumes/VGISS_5xxx/VGISS_5213/DATA/C20621XX/C2062133_GEOMED.IMG'

# Image of Uranus (crescent)
# Camera: Voyager ISS
# Size: 800x800
# Filter: CLEAR
# Exposure: 3.84 sec
# https://opus.pds-rings.seti.org/opus/#/view=detail&detail=vg-iss-2-u-c2712527
URL_VOYAGER_ISS_URANUS_01 = 'https://pds-rings.seti.org/holdings/volumes/VGISS_7xxx/VGISS_7207/DATA/C27125XX/C2712527_GEOMED.IMG'

# Image of a star field
# Camera: Voyager ISS
# Size: 800x766
# Filter: CLEAR
# Exposure: 0.96 sec
# https://opus.pds-rings.seti.org/#/view=detail&detail=vg-iss-2-s-c4288814
URL_VOYAGER_ISS_STARS_01 = 'https://pds-rings.seti.org/holdings/volumes/VGISS_6xxx/VGISS_6205/DATA/C42888XX/C4288814_GEOMED.IMG'

# Image of a star field - single star CMa in small image rectangle
# Camera: Voyager ISS
# Size: 89x210
# Filter: CLEAR
# Exposure: 0.01 sec
# https://opus.pds-rings.seti.org/#/view=detail&detail=vg-iss-2-n-c1036542
URL_VOYAGER_ISS_STARS_02 = 'https://pds-rings.seti.org/holdings/volumes/VGISS_8xxx/VGISS_8203/DATA/C10365XX/C1036542_GEOMED.IMG'
