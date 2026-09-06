"""Returning a strip's transient memory to the operating system.

A backplane evaluated a strip of rows at a time keeps only one strip's
intermediates alive at once, and that is what bounds the live heap.  It does
not on its own bound the process's resident size.  Two things stand between
the two, and neither is enough by itself:

* The intermediates are held in reference cycles, so dropping the last name
  bound to a strip does not free it.  Nothing is reclaimed until a collection
  runs, and a collection of the oldest generation is not otherwise due within
  the handful of allocations a strip costs.
* Once freed, the C allocator keeps the arenas rather than returning them, so
  the address space a strip occupied is still charged to the process.

Left alone, the two together make a run's resident size grow by the sum of its
strips rather than by the largest of them, which is the number striping exists
to reduce.  Resident size is also the number that matters: it is what the
kernel's out-of-memory killer reads, and what a recorded peak reports.  A
striped pass that does not release is a pass whose striping cannot be observed.

Releasing costs nothing worth counting.  A strip is expensive enough that a
collection between strips does not register against it, and measurement on a
whole-frame ring pass put the released and unreleased runs within noise of each
other.  This is deliberately not wired into a general allocation path: it is
worth doing between units of work already large enough to pay for it, and
nowhere else.
"""

import ctypes
import ctypes.util
import gc


def _malloc_trim() -> ctypes.CDLL | None:
    """Find the C library's arena-releasing entry point, if it has one.

    Returns:
        The loaded library exposing ``malloc_trim``, or None where no such
        function exists, which is every C library other than glibc.
    """
    name = ctypes.util.find_library('c')
    if name is None:
        return None
    try:
        libc = ctypes.CDLL(name)
    except OSError:
        return None
    if not hasattr(libc, 'malloc_trim'):
        return None
    libc.malloc_trim.argtypes = [ctypes.c_size_t]
    libc.malloc_trim.restype = ctypes.c_int
    return libc


_LIBC: ctypes.CDLL | None = _malloc_trim()


def release_transient_memory() -> None:
    """Give back the memory a just-finished unit of work was holding.

    Collects the cycles the unit left behind and returns the freed arenas to
    the operating system, so the next unit starts from the resident size this
    one started from rather than from the sum of the two.

    Where the C library cannot return arenas the collection still runs, which
    is the larger of the two effects; the resident size then falls when the
    allocator next reuses the space rather than immediately.
    """
    gc.collect()
    if _LIBC is not None:
        _LIBC.malloc_trim(0)
