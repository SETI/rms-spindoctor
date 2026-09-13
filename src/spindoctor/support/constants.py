"""Constants module for navigation calculations.

This module defines mathematical constants used throughout the navigation system.
"""

import math

PI = math.pi
HALFPI = math.pi / 2


PIXEL_CENTER_TO_CORNER_PX: float = 0.5
"""Half a pixel: what separates the two ways a pixel can be named.

In **pixel corner coordinates** an integer is a boundary between two pixels, so
the centre of pixel ``i`` lies at ``i + 0.5``.  In **pixel index coordinates**
an integer names a pixel, so index ``i`` *is* that centre.  The two differ by
this constant and by nothing else::

    index  = corner - PIXEL_CENTER_TO_CORNER_PX
    corner = index  + PIXEL_CENTER_TO_CORNER_PX

Nothing about it belongs to any one kind of thing in a frame.  A catalog star, a
body centre, a ring edge, a scene position and a PSF phase all cross the same
boundary, and they must all cross it the same way: a navigated offset is the
difference between a prediction and a measurement, so a datum error shared by
both sides cancels everywhere it can be checked and survives only in the
absolute answer.

Which convention a value is in is a property of the value, not of whatever
produced it.  A conversion written against this name is one a reader can check
against the rule above; a bare ``0.5`` is not.
"""
