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

Reading the peak
----------------

The kernel keeps a process's high-water resident size, which is the figure an
out-of-memory kill is decided against.  It is a high-water mark over the life of
the process, so a process navigating several images in turn would report the
largest of them against every image after the first.  Resetting it at the start
of each navigation is what stops that.

The reset sets the mark to the resident size at that moment, not to zero, so
what a unit of work reports is what the process reached while it ran rather than
what the work itself allocated.  For a process handling one image the two are
the same.  For a process handling several they are not: memory an earlier image
left resident and unreturnable is inside the floor the next one is measured from.
That is the honest figure for sizing a worker, which has to hold the whole
process, and the wrong one for asking how much memory a single image uses on
its own.

Both operations read and write ``/proc``, which exists on Linux and not
elsewhere.  Where it is absent the peak is reported as None rather than as a
number that would mean something different, and a run records no peak instead
of recording a wrong one.
"""

import ctypes
import gc


def _library_with_malloc_trim() -> ctypes.CDLL | None:
    """Open the running program's C library, if it exposes ``malloc_trim``.

    The running program is opened rather than a library found by name: its
    symbol table already carries the C library's, so this costs one ``dlopen``
    and spawns nothing, where a lookup by name runs the dynamic linker's cache
    tool as a subprocess and finds nothing wherever that tool is absent.

    Returns:
        The loaded library exposing ``malloc_trim``, or None where the running
        program cannot be opened this way, as on Windows, or where its C
        library exposes no such function.
    """
    try:
        lib = ctypes.CDLL(None)
    except (OSError, TypeError):
        # Windows cannot open the running program by a missing name and
        # rejects it before opening anything; its C runtime has no
        # malloc_trim to find in any case.
        return None
    if not hasattr(lib, 'malloc_trim'):
        return None
    lib.malloc_trim.argtypes = [ctypes.c_size_t]
    lib.malloc_trim.restype = ctypes.c_int
    return lib


_LIBC: ctypes.CDLL | None = _library_with_malloc_trim()


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


def peak_resident_bytes() -> int | None:
    """The largest resident size this process has reached.

    Returns:
        The peak in bytes, or None where the kernel does not publish one.
    """
    try:
        with open('/proc/self/status') as handle:
            for line in handle:
                if line.startswith('VmHWM:'):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def reset_peak_resident() -> bool:
    """Set the recorded peak back to the resident size right now.

    Called at the start of a unit of work whose own peak is wanted, so that
    what a longer-lived process reached earlier is not attributed to it.

    Returns:
        True when the peak was reset, False where the kernel does not allow
        it, in which case the peak remains the whole process's.
    """
    try:
        with open('/proc/self/clear_refs', 'w') as handle:
            handle.write('5\n')
    except OSError:
        return False
    return True
