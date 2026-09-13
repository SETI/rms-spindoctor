"""Constants module for navigation calculations.

This module defines mathematical constants used throughout the navigation system.
"""

import math

PI = math.pi
HALFPI = math.pi / 2


PIXEL_CENTER_TO_CORNER_PX: float = 0.5
"""Half a pixel, which is what separates the two coordinate systems in use.

In pixel-corner coordinates a whole number falls on the boundary between two
pixels, so the center of the first pixel is at 0.5.  In pixel-centric
coordinates a whole number falls on a pixel's center, so the center of the
first pixel is at 0.0.  Both are continuous; the systems say only where the
whole numbers fall::

    pixel_centric = pixel_corner  - PIXEL_CENTER_TO_CORNER_PX
    pixel_corner  = pixel_centric + PIXEL_CENTER_TO_CORNER_PX

The geometry layer works in pixel-corner coordinates and anything that draws
into or measures out of an array works in pixel-centric ones, so a conversion
appears wherever a value crosses between those stages.  Which system applies
follows from the stage, so a conversion written against this name says which
stage it is crossing out of; a bare ``0.5`` does not.

A half-pixel error here is hard to see from inside the pipeline: a navigated
offset is a difference between a predicted position and a measured one, so an
error common to both cancels in everything computed from the same pair and
survives only in the absolute answer.
"""
