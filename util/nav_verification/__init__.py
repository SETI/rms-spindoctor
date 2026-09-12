"""Checking a navigation pass against something that was not part of it.

A navigation run reports its own confidence, and that number is a statement
about the fit rather than about the answer: a technique can fit two stars
tightly and still be tens of pixels away from where the camera was pointing.
Deciding whether a run is right therefore needs evidence the run did not
produce, and this package holds the two kinds that are available.

- ``compare_pointing`` checks each frame against an independently navigated
  answer for the same frame, when one exists.  A published bundle that records
  the boresight its own project navigated to is such an answer, and
  ``bundle_cassini_fring`` is the one currently in reach.
- ``measure_core_radius`` needs no external answer at all.  A mosaic built in a
  co-rotating frame puts a known ring feature at a known radius, so where that
  feature actually lands measures the navigation of every frame that
  contributed, and a radius that steps between adjacent columns says two
  neighboring frames were navigated differently from each other.

``boresight`` is the arithmetic both of them share, and carries no mission
constants of its own.
"""

__all__: list[str] = []
