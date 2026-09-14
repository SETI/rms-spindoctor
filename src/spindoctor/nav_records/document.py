"""The document a navigation record is written to, and how one is read back.

:mod:`spindoctor.support.nav_record` decides what the values of a record mean.
This module is about the file those values are written to: what it is named,
where the document of one image lives under a results root, and how one is read.
It knows nothing about a database, so every reader of a document shares it
whether or not the program it belongs to can read an index.

The three rules here are each written once because each of them was written more
than once before, and a rule about which files may be read is not a rule while
two readers hold different versions of it:

* **What a document is named.**  A results path stub is a document's path under
  its root with the suffix taken off, so the walk that lists documents, the
  ingest that records their stubs and every reader that rebuilds a path from one
  have to agree about which suffix that is.
* **What makes a stub a key.**  A stub names one document under one root, so it
  may not be an absolute path, may not walk upwards out of the root, and may not
  carry a byte the filesystem will not accept.  It is checked where a key is
  written rather than where one is read: a results root is canonical from the
  moment it is spelled, so a validated key joined onto it is the same answer for
  every caller, and no reader needs a rule of its own about what that join may
  have produced.
* **What makes a file a document.**  Valid JSON holding an object, which is what
  every reader needs before it can read a field off one.

The reasons a file is not a document are named here once, and
:func:`document_or_refusal` is the one place that decides which of them a file
earns.  A vocabulary spelled out twice is not a vocabulary: two readers of one
tree would give one file two different reasons for the same fault, and a
consumer comparing what the two storages hold would read that as a disagreement
about the file.

The same function fixes the policy on an escape nobody enumerated: it ends the
pass rather than becoming a reason.  A parse that recursed too deeply or ran out
of memory is a property of the file -- nothing came out of it -- and is named.
Anything else out of a reader is a property of this code, and reporting it
against the file would say that a tree of perfectly good documents was
malformed, which is a worse answer than stopping.
"""

import json
from pathlib import Path
from typing import Any, cast

from filecache import FCPath

__all__ = [
    'ABSOLUTE_PATH_FRAGMENT',
    'COULD_NOT_RETRIEVE',
    'METADATA_SUFFIX',
    'NOT_A_JSON_OBJECT',
    'NOT_A_SINGLE_COMPONENT',
    'NOT_VALID_JSON',
    'NULL_BYTE_IN_PATH',
    'PARENT_SEGMENT_IN_PATH',
    'UNREADABLE',
    'document_or_refusal',
    'document_path',
    'read_document',
    'stub_for_document',
    'stub_refusal',
    'subtree_refusal',
]

METADATA_SUFFIX = '_metadata.json'
"""What a per-image navigation document is named."""

NULL_BYTE_IN_PATH = 'metadata path contains null byte'
"""Why a stub naming a path with a null byte may not be read.

It renders perfectly well and then fails at the first call that reaches the
filesystem, which is a failure charged to a directory listing rather than to the
key that caused it.
"""

ABSOLUTE_PATH_FRAGMENT = 'metadata path fragment is absolute'
"""Why a stub that is an absolute path may not be read under a root.

Joining an absolute fragment onto a root discards the root, so such a key names a
file under no root at all -- and one read under it would be recorded against a
root that never held it.
"""

PARENT_SEGMENT_IN_PATH = 'metadata path fragment names a parent directory'
"""Why a stub carrying a ``..`` segment may not be read under a root.

A key that walks upwards names a file outside the root it was asked under, or the
same file under a second name; both are answers a consumer keyed on
``(root, stub)`` cannot match.  It is refused as a key rather than resolved and
compared, because there is no reading of it this root answers for.
"""

NOT_A_SINGLE_COMPONENT = 'subtree is not one directory under the root'
"""Why a subtree that is not one path component may not be descended.

A subtree names one directory immediately under a root: it is the first component
of every stub beneath it, which is what an index stores it as and what a walk
builds those stubs from.  Anything else means one thing to a walk that joins it
and another to a query that compares it, so the two storages would answer one
selection two ways.
"""

COULD_NOT_RETRIEVE = 'could not be retrieved'
"""Why a file that never arrived is not a record.

The one reason that says nothing about the file itself: the storage layer did not
deliver it, so nothing is known about it beyond what the listing said.  It is
told apart from the reasons below because a retrieval that failed once is worth
trying again, where a file that was read and refused will be refused again for as
long as it does not change.
"""

UNREADABLE = 'unreadable'
"""Why a file whose bytes could not be read as text is not a record."""

NOT_VALID_JSON = 'not valid JSON'
"""Why a file no JSON value came out of is not a record.

Carried by every way the decoder can fail to produce a value, not by malformed
syntax alone: a document nested deeply enough to exhaust the recursion limit and
one large enough to exhaust memory both end with nothing parsed, which is the
fact the reason states.
"""

NOT_A_JSON_OBJECT = 'not a JSON object'
"""Why a file parsing to a JSON value of another kind is not a record."""


def stub_refusal(results_path_stub: str) -> str | None:
    """Return why a results path stub is not a key, or None when it is one.

    A stub is the identity of one image under one results root, so it names a
    location relative to that root and nothing else.  Three spellings are not
    that, and each of them would otherwise reach a file the root does not hold:
    a null byte, an absolute path, and a segment naming a parent directory.

    Nothing is logged and nothing is raised: which of those a refusal becomes is
    the caller's to decide, and the callers decide differently.  A selection
    refuses where it is written; a per-image reader reports it against the image
    and carries on with no pointing; a reader asked for the record itself has
    nothing to hand back and raises.

    Parameters:
        results_path_stub: The stub to check.

    Returns:
        One of :data:`NULL_BYTE_IN_PATH`, :data:`ABSOLUTE_PATH_FRAGMENT` and
        :data:`PARENT_SEGMENT_IN_PATH`, or None when the stub breaks no rule.
    """
    if '\x00' in results_path_stub:
        return NULL_BYTE_IN_PATH
    # Absolute in the storage layer's own terms rather than the filesystem's: a
    # cloud URL is a fragment no local path test calls absolute, and joining one
    # onto a root discards the root exactly as a leading separator does.
    if FCPath(results_path_stub).is_absolute():
        return ABSOLUTE_PATH_FRAGMENT
    if '..' in results_path_stub.split('/'):
        return PARENT_SEGMENT_IN_PATH
    return None


def subtree_refusal(subtree: str) -> str | None:
    """Return why a subtree is not one directory under a root, or None when it is.

    A subtree is the first component of the stubs beneath it: a walk descends
    the directory of that name and builds those stubs from it, and an index
    stores that component and compares it.  So it is one path component, and one
    that names a directory: empty names the root itself, ``.`` and ``..`` name a
    directory the root does not hold under that name, and anything carrying a
    separator is a fragment rather than a component.

    Parameters:
        subtree: The subtree to check.

    Returns:
        One of :data:`NULL_BYTE_IN_PATH` and :data:`NOT_A_SINGLE_COMPONENT`, or
        None when the subtree breaks no rule.
    """
    if '\x00' in subtree:
        return NULL_BYTE_IN_PATH
    if subtree in ('', '.', '..') or '/' in subtree:
        return NOT_A_SINGLE_COMPONENT
    return None


def document_path(nav_results_root: str | Path | FCPath, results_path_stub: str) -> FCPath:
    """Return where one stub's document lives under a results root.

    The one join in the seam, made the same way by every caller: the writer about
    to create a document, the reader about to open one, and the message naming
    where a record came from.  A root is normalized to an absolute, resolved
    location where it is spelled and a stub is refused unless it is a key where
    it is written, so this join has one answer and nothing left to check.

    Parameters:
        nav_results_root: Root the navigator wrote its documents under.
        results_path_stub: The image's results path stub.

    Returns:
        The document's location, which is the stub under the root with the
        document suffix restored.
    """
    return FCPath(nav_results_root) / f'{results_path_stub}{METADATA_SUFFIX}'


def stub_for_document(root: FCPath, path: FCPath) -> str:
    """Return the results path stub of one document under a root.

    The root comes off a whole component at a time rather than a character at a
    time.  ``/data/res`` is a character prefix of ``/data/results`` and is the
    parent of nothing under it, so taking it off by characters would turn a
    neighboring directory's document into something that looks like a key,
    reads like a key, and names a file this root does not hold -- silently,
    since nothing downstream can tell such a stub from a real one.

    Parameters:
        root: The navigation results root.
        path: The document.

    Returns:
        The file's path relative to the root, without the document suffix.  The
        full path is used when it does not lie under the root, which cannot
        happen for a document the root's own listing produced.
    """
    prefix = root.as_posix().rstrip('/')
    spelled = path.as_posix()
    relative = spelled[len(prefix) + 1 :] if spelled.startswith(f'{prefix}/') else spelled
    return relative.removesuffix(METADATA_SUFFIX)


def read_document(path: FCPath | Path) -> dict[str, Any]:
    """Read one navigation document.

    The bytes are decoded as UTF-8 because that is what JSON is, so a document
    says the same thing to every machine that reads it.  A reader left to the
    encoding its machine prefers reads some other document out of the same
    bytes, or none: which files are documents at all would then be a property of
    the reader rather than of the tree, and two readers of one tree would not
    agree on it.

    Parameters:
        path: The file to read.  A local path is read where a retrieval has
            already produced one, so that reading it costs nothing further.

    Returns:
        The document.

    Raises:
        UnicodeDecodeError: If the bytes are not UTF-8, which is a file that is
            not a JSON document whatever else it may be.
        ValueError: If the file does not hold a JSON object.  A file holding
            valid JSON that is not an object is not a document, and reading
            fields off one would fail later and further away.
        OSError: If it cannot be read.  A file that is not there raises
            :exc:`FileNotFoundError`, which is what a caller distinguishing an
            unnavigated image from an unreadable one catches.
    """
    document = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(document, dict):
        raise ValueError(f'holds a {type(document).__name__}, not a JSON object')
    return cast(dict[str, Any], document)


def document_or_refusal(path: FCPath | Path) -> dict[str, Any] | str:
    """Read one file as a navigation document, or say why it is not one.

    The one classification of a file that yielded no document, shared by every
    reader of a results tree, so one file earns one reason whichever reader met
    it.

    Parameters:
        path: The file to read.  A local path is read where a retrieval has
            already produced one, so that reading it costs nothing further.

    Returns:
        The document, or one of :data:`UNREADABLE`, :data:`NOT_VALID_JSON` and
        :data:`NOT_A_JSON_OBJECT`.

    Raises:
        Exception: Whatever the reader raises that is not one of the enumerated
            faults of a file.  A fault in this code is a property of the code
            rather than of the file, and reporting it as a refusal would say the
            file was malformed and record that answer for as long as the file
            does not change.
    """
    try:
        return read_document(path)
    except (OSError, UnicodeDecodeError):
        return UNREADABLE
    except json.JSONDecodeError:
        return NOT_VALID_JSON
    except ValueError:
        # What is left of ValueError once the decoder's own failure is taken
        # out is the document rule itself: valid JSON that is not an object.
        return NOT_A_JSON_OBJECT
    except (RecursionError, MemoryError):
        # The decoder reports more than malformed syntax.  A decoder that
        # recurses once per level of nesting exhausts the recursion limit on a
        # deeply nested document rather than failing to parse it, and one that
        # runs out of memory says so its own way.  How deep is deep enough is
        # the decoder's business and not this code's.  Neither is a reason to
        # end a pass, and what happened in both is that no value came out of the
        # file, which is the fact the reason states.
        return NOT_VALID_JSON
