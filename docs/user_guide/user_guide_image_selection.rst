===========================
Selecting Images to Process
===========================

For PDS3 datasets (``coiss``, ``coiss_pds3``, ``coiss_cruise``, ``coiss_cruise_pds3``, ``coiss_saturn``, ``coiss_saturn_pds3``, ``gossi``, ``gossi_pds3``, ``nhlorri``, ``nhlorri_pds3``, ``vgiss``, ``vgiss_pds3``), the following options control which images are selected. All filters combine with logical AND, and explicit lists restrict the search domain before range filters to improve performance.

* ``--pds3-holdings-root PATH``: root directory or URL of the PDS3 holdings tree
  the images are read from, overriding both the ``environment.pds3_holdings_root``
  configuration setting and the ``PDS3_HOLDINGS_DIR`` environment variable, in
  that order. Offered by the dataset, so it is present exactly when the dataset
  named on the command line reads PDS3 holdings, and absent otherwise.
* ``img_name`` (positional, repeatable): specific image name(s) to process.
* ``--first-image-num N``: minimum image number (inclusive).
* ``--last-image-num N``: maximum image number (inclusive). Voyager Flight
  Data Subsystem (FDS) counts restart per spacecraft/encounter, so for Voyager
  a number range can match frames from more than one encounter; combine it
  with the volume options to bound the selection.
* ``--volumes NAME[,NAME...]`` (repeatable): one or more complete PDS3 volume names; you may pass comma-separated values or specify the option multiple times.
* ``--first-volume NAME``: starting PDS3 volume; only that volume and chronologically later ones are processed.
* ``--last-volume NAME``: ending PDS3 volume; only that volume and chronologically earlier ones are processed.
* ``--image-filespec-csv FILE`` (repeatable): CSV file(s) containing PDS3 file specifications; files must include a header column named ``Primary File Spec`` or ``primaryfilespec``.
* ``--image-file-list FILE`` (repeatable): file(s) containing file specifications or names, one per line; lines beginning with ``#`` are ignored.
* ``--choose-random-images N``: choose a random subset of N images, uniformly
  distributed across all images (in every selected volume) that meet the other
  criteria; the selected images are yielded in random order.
* ``--has-offset-file`` / ``--has-no-offset-file``: only images whose offset
  metadata file (``*_metadata.json`` under the navigation results root) already
  exists / does not exist.
* ``--has-offset-error`` / ``--has-no-offset-error``: only images whose offset
  metadata file exists and records / does not record a fatal error (``status``
  of ``error``). Every error filter asks what a document records, so each one
  requires the document to exist: an image nothing has been written for records
  no error, and ``--has-no-offset-file`` is what selects it. So
  ``--has-offset-file --has-no-offset-error`` is how a run asks for the images
  whose navigation reached a result rather than dying before it could read the
  image. What it selects is stated exactly below, along with the answers a
  results index is known to give differently from the results tree.
* ``--has-offset-spice-error`` / ``--has-offset-nonspice-error``: like
  ``--has-offset-error``, but restricted to fatal errors caused by / not caused
  by missing SPICE data.

An error filter asks what a document records about its image, and reads it from
the per-image facts the document yields: the values a results index holds in its
columns, so a document is narrowed on exactly what a row is narrowed on. An
offset metadata file that cannot be read, does not parse as JSON, does not parse
to a JSON object, or parses to a JSON object that is not a navigation document of
the current metadata schema yields no such facts, and satisfies no error filter,
positive or negative: what it records is unknown rather than known to be an
outcome. A document written to an earlier metadata schema is one of those, so a
results root holding nothing else answers every error filter, including
``--has-offset-spice-error``, with no image at all; re-navigating those images
rewrites their documents to the current schema. Such a file is still a file that
exists, so the presence filters count it, ``--has-offset-file`` selects its image
and ``--has-no-offset-file`` passes it over. When a run ends, its log says how
many of the candidates it read yielded no facts and names one of them with the
reason, so a selection short for this reason says so rather than only coming back
smaller than expected.

Two questions cover all six flags, and they are asked at different moments
because they need different things.

Which images the results root holds a document for opens no document at all, and
a run selecting images that *have* one --- ``--has-offset-file``, and every error
filter, since each of those reads what a document records --- asks it once, when
the enumeration starts, by listing the volumes it selected. That costs one
listing per directory rather than one read per image, and testing a candidate
image against the answer afterwards is a set lookup that costs nothing.

A run selecting images that have *none* --- ``--has-no-offset-file`` --- asks the
same question about the candidate images themselves, in batches, as the
enumeration offers them. It keeps the images the results root holds nothing for,
so every entry a listing of a whole volume produced would be an entry to reject
from: a run whose other constraints name ten images would pay for fifty thousand
of them to answer about ten. Asking about the candidates instead costs a check
per candidate on a local results root, where a check is a system call, and one
listing per directory on a cloud one, where it is a paid round trip --- and on a
cloud root one listing serves every batch of the run rather than being taken
again for each.

What each document records is read from the per-image facts, in batches, as the
enumeration offers its candidates. An error filter has to read a document, and
which documents to read is the set of candidate images, which the other
selection constraints decide and which is not known when the enumeration starts.
So a batch of candidates names its images and asks about those. A run whose
other constraints keep one image in a hundred therefore reads a hundredth of the
documents, rather than every document under the volumes it selected --- on a
cloud results root, one paid download apiece. Only images that have a document
are ever asked about, since the listing has already excluded the rest, which is
also what makes every error filter keep only images that have one.

A document read to answer an error filter is not read a second time. Every
program that processes an image reads one navigation record for it, and a run of
``sd_backplanes`` or ``sd_mosaic`` that names an error filter has already read
that record while it was deciding what to select: the record travels with the
image, and the per-image stage uses it rather than opening the document again.
On a cloud results root that is a download saved for every image the run goes on
to process, not merely a second look at a local copy --- the enumeration and the
per-image stage keep separate download caches, so neither ever sees the other's
files.

Two consequences follow, and both are properties of the run rather than of any
one image. What such a run applies is what the document said when the selection
was made, so a document rewritten or deleted while the run is going is not
noticed for an image already selected; the same is true of any run for the
window between its listing and its per-image stage, and this narrows that window
rather than opening a new one. And the reasons that describe failing to read a
document --- reported as ``no_metadata`` for one that is not there, and as
``unreadable_metadata``, ``invalid_json`` or ``metadata_not_an_object`` for one
that will not read --- are not reported for an image whose record was carried,
because a record is carried only for a document that was read whole. Such an
image is reported under what its record records, or under nothing at all.

Given ``--results-index-db`` nothing is carried, and this saves nothing: the
filter narrows on columns and the per-image stage reads the columns it needs,
which are a different set, so the two stay separate reads of the index.

The listing taken when the enumeration starts covers the volumes the enumeration
selected and no others, and they are asked about one at a time. Reading the
results tree, a selected volume the results root has no directory for
contributes nothing: a volume nobody has navigated yet has no directory under
the results root, which is an ordinary state of a results tree rather than a
reason to stop. A directory that is there and will not be listed -- this user may
not read it, the share it lives on has gone away -- ends the run instead, because
a filter answering from what it could see would silently select images it has no
evidence about. Asking one volume at a time is what tells those apart: a single
request covering all of them would end at the first volume it could not read, and
every volume after it would go unasked.

A run that asks about its candidates needs no such restriction and applies none.
The images it is offered come from the volumes the enumeration walked, so naming
those volumes as well would narrow nothing, and a volume the results root has no
directory for still costs nothing and ends nothing: the images under it are
images the root holds no document for.

Given ``--results-index-db``, whichever of the two questions the flags call for
is answered from the index's rows, and the results tree is not read at all. A
results root for which the index has no completed ingest is refused rather than
answered, because absence of a row would otherwise read as "this image was never
navigated". The paragraphs below are the answers an index is known to give
differently from the tree; each of them is a property of what an ingest pass
could read and record rather than of the storage.

**A file the index has no row at all for reads as absent**, and
``--has-no-offset-file`` reads absence as "this image was never navigated", so
it selects that image again. One pass ends that way: one that could not retrieve
the file. Nothing is recorded for it deliberately -- a recorded row would be
skipped for as long as the file did not change, and a download that failed once
says nothing that will still be true next pass. Two other ways a file could go
unrecorded are not this, because neither leaves a completed pass behind it: an
ingest that cannot list a directory stops there, and one whose document the
index would not store stops there. A root the index holds a completed pass over
is a root every directory of which was listed and every document of which was
stored.

**A document rewritten in place, keeping the length and the modification time
it had before,** is one the ingest skips, because those two metrics are
everything a listing supplies about a file. Its row goes on recording what the
document before it said, so an error filter answers from that one however
recently the last pass finished. A tree restored by a copy that preserves times,
a document patched and stamped back from a sibling, and a backend reporting one
modification time for two writes all produce it; an ordinary re-navigation
writes a different length at a later time and does not. Running
``sd_results_index ingest --force`` over the root re-reads every document and is
what puts such a row right.

An index is also a snapshot of the tree as of the last ingest over that root,
with no staleness detection: an image navigated since is one the index does not
hold, so ``--has-no-offset-file`` selects it again, and a result file deleted
since is one the index still holds, so ``--has-offset-file`` selects an image
whose metadata file is gone. The run log says when the pass that filled the
index finished and how long ago that was, which is what says whether either
applies to this run. Run ``sd_results_index ingest`` to bring the index up to
date, or
pass ``--results-index-db none``, which names no index, for a run that must read
the tree. That age is
what decides the answer outside the paragraphs above; inside them it decides
nothing, since each of them survives a pass that finished a second ago.

An ingest that meets a directory it cannot list stops there, reports it as an
error, and completes no root from that point on, so no index a consumer reads
holds a root that was only partly walked. A results root that cannot be listed
at all is the other case and does not stop the pass: that root alone is left
unfinished, which every consumer already refuses, and the ingest goes on to the
next root. Fix what stopped the walk -- a directory permission, a share that was
not answering -- and run ``sd_results_index ingest`` again.

