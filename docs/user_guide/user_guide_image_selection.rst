===========================
Selecting Images to Process
===========================

Most SpinDoctor programs work on many images in one run. Selecting images is how you say
which ones. The same options do this job in every program that processes images.

What Selection Is
=================

A program that processes images starts by building the list of images to work on. The
list begins as every image the dataset named on the command line offers, and each
selection option narrows it. All the options you pass must be satisfied at once: they
combine with logical AND, so ``--volumes COISS_2001 --camera nac`` selects the
narrow-angle images of that one volume.

The whole list is settled before any image is processed, so a selection can be inspected
on its own. ``sd_offset --dry-run`` prints the images a selection resolves to and does
nothing else.

You will want to narrow a selection for several reasons:

* An archive holds far more images than one run needs. A selection bounded by volume, by
  image number, or by camera keeps the run to the images that interest you.
* Work often arrives as a list of images someone else chose, in a file. A selection can
  read that list.
* A long run may be interrupted. A selection that keeps only the images that have no
  result yet resumes it.
* A run may have failed on some images for a reason you have since fixed, such as a SPICE
  kernel you have now downloaded. A selection can pick out exactly those images and
  nothing else.
* A spot check over a whole mission wants a sample rather than the first few thousand
  images. A selection can draw one at random.

Which Programs Accept These Options
===================================

These options come from the dataset rather than from any one program, so every program
that enumerates images out of an archive offers the same set:

* ``sd_offset`` -- navigation (:doc:`user_guide_navigation_running`).
* ``sd_backplanes`` -- backplane generation (:doc:`user_guide_backplanes`).
* ``sd_create_bundle`` -- PDS4 bundle generation (:doc:`user_guide_pds4_bundle`).
* ``sd_consolidate_metadata`` -- gathering navigation results into one directory
  (:doc:`user_guide_consolidate_metadata`).
* ``sd_mosaic`` and ``sd_backplane_viewer`` -- reprojection and viewing
  (:doc:`user_guide_reprojection`).

Which options a run accepts depends on the dataset it names, because each dataset offers
the options that suit its archive. Everything in this chapter applies to the PDS3
datasets, which is all four supported instruments: ``coiss``, ``coiss_cruise``,
``coiss_saturn``, ``gossi``, ``nhlorri``, and ``vgiss``, together with the ``_pds3``
alias of each. The ``sim`` dataset takes the paths of its scene files instead
(:doc:`user_guide_simulated_images`). Running a program with ``--help`` and the dataset
you intend to use lists exactly what that combination accepts.

Where the Answers Come From
===========================

Selection reads two kinds of data, and never opens an image file.

The candidate images, and everything known about them before processing, come from the
**PDS3 index table** of each selected volume. That is the table shipped with the volume
itself, holding one row per image, which a holdings tree keeps under ``metadata``. The
options that select by volume, by image name, by image number, or by camera are all
answered from those rows. Index tables are cached locally after they are first read, so
repeating a run over the same volumes does not fetch them again.

The options that ask what a previous navigation recorded are answered from the
**navigation results**: either the tree of metadata documents under the navigation
results root, or, for a program that was given one, the **results index**. The results
index is a database holding one row per navigated image, built by ``sd_results_index``
(:doc:`user_guide_results_index`). Where this chapter says "results index" it always
means that database, and where it says "PDS3 index table" it always means the table that
came with the volume.

Where the Images Are Read From
==============================

``--pds3-holdings-root ROOT``
  The root directory or URL of the PDS3 holdings tree the images and their PDS3 index
  tables are read from. Overrides the ``environment.pds3_holdings_root`` configuration
  setting and the ``PDS3_HOLDINGS_DIR`` environment variable, in that order. See
  :doc:`user_guide_installation` for the layout the tree must have.

Selecting by Volume
===================

``--volumes NAME[,NAME...]``
  One or more complete PDS3 volume names. Only images in those volumes are processed.
  Pass several names separated by commas, or repeat the option, or both. A name that is
  not a volume of this dataset is refused and the run stops.

``--first-volume NAME``
  Process only this volume and the chronologically later ones.

``--last-volume NAME``
  Process only this volume and the chronologically earlier ones.

Volume options are the cheapest way to make a run smaller, because a volume nobody
selected has its PDS3 index table left unread.

Selecting by Image Name
=======================

``img_name`` (positional, repeatable)
  One or more image names. A name is matched case-insensitively against the start of each
  image name, so a complete name selects one image and a partial name selects every image
  whose name begins with it.

``--image-file-list FILE`` (repeatable)
  A file holding one image name or file specification per line. Blank lines and lines
  beginning with ``#`` are ignored, and anything after the first space on a line is
  ignored. The names are matched exactly as the positional names are. A line that is not
  a valid name for this dataset is refused and the run stops.

``--image-filespec-csv FILE`` (repeatable)
  A CSV file of PDS3 file specifications, of the kind PDS publishes for a search result.
  The file must have a header row with a column named ``Primary File Spec`` or
  ``primaryfilespec``. Each row's image name must match an image exactly. A row too short
  to have that column is reported and skipped, and the run goes on.

Naming images does not by itself restrict the volumes a run looks in. Combining a name
list with volume options keeps the run from reading PDS3 index tables that it does not
need.

Selecting by Image Number
=========================

``--first-image-num N``
  The lowest image number to process, inclusive.

``--last-image-num N``
  The highest image number to process, inclusive.

An explicit list of names or file specifications also tightens the number range on its
own, to the span of the numbers in the list, so a name list does not cost a scan of rows
that are outside it.

For most datasets image numbers rise from one volume to the next, so a run bounded above
stops scanning once it is past the range. Voyager Flight Data Subsystem (FDS) counts
restart per spacecraft and encounter, so a Voyager run scans every selected volume, and a
Voyager number range can match frames from more than one encounter. Combine it with the
volume options to bound the selection.

Selecting a Random Sample
=========================

``--choose-random-images N``
  Process a random sample of N images, drawn uniformly from every image that satisfies
  the other options across all the selected volumes, and yielded in random order.

Drawing a uniform sample means every selected volume's PDS3 index table is read, because
the pool has to be complete before it can be sampled. The tables are cached locally, so
the cost is paid once per machine rather than once per run.

Selecting by Camera
===================

``--camera {nac,wac}``
  Process only images from the named camera. Offered by the Cassini ISS datasets, which
  have two cameras. The value may be given in upper or lower case. The camera is read
  from the PDS3 index table row, so an image is attributed to its camera without being
  opened.

Selecting by What a Previous Run Recorded
=========================================

Navigating an image writes a metadata document for it under the navigation results root,
recording the outcome. Six options select images by what those metadata documents say.
These are the options that let a run resume where an earlier one stopped, or go back over
the images an earlier one could not navigate. All six look at the navigation results --
the tree of metadata documents, or the results index when the program was given one -- and
none of them opens an image file.

``--has-offset-file``
  Keep only images that already have a metadata document.

``--has-no-offset-file``
  Keep only images that have no metadata document. These are the images that were never
  navigated.

``--has-offset-error``
  Keep only images whose metadata document exists and records a fatal error.

``--has-no-offset-error``
  Keep only images whose metadata document exists and records something other than a
  fatal error. These are the images whose navigation ran to a result, whether or not it
  found an offset.

``--has-offset-spice-error``
  Keep only images whose metadata document exists and records a fatal error caused by
  missing SPICE data.

``--has-offset-nonspice-error``
  Keep only images whose metadata document exists and records a fatal error from some
  other cause.

Each of the four error options asks what a metadata document records, so each of them
requires the document to exist. An image that has no metadata document records no error,
and ``--has-no-offset-file`` is what selects it.

Combinations that nothing could satisfy are refused before the run starts, and the
message names every flag involved:

* ``--has-offset-file`` with ``--has-no-offset-file``.
* ``--has-offset-spice-error`` with ``--has-offset-nonspice-error``.
* ``--has-no-offset-file`` with any of the four error options, since those need a
  document to read.
* ``--has-no-offset-error`` with any of the three options that name an error.

What These Options Cost
-----------------------

Whether a metadata document exists is a question that opens no document, and it is asked
in one of two ways.

A run selecting images that *have* one -- ``--has-offset-file``, and every error option,
each of which needs a document to read -- lists the results directories of the selected
volumes once, when the run starts. That is one listing per directory rather than one read
per image, and testing a candidate image against the result afterwards costs nothing.

A run selecting images that have *none* -- ``--has-no-offset-file`` -- asks about the
candidate images themselves, in batches, as they are enumerated. It has to ask that way
to be worth asking. A listing of whole volumes would be a list of images to reject, so a
run whose other options name ten images would pay for fifty thousand entries to answer
about ten. Asking about the candidates costs one check per candidate on a local results
root, where a check is a system call, and one directory listing on a cloud results root,
where a check is a paid round trip; there, one listing serves every batch of the run.

An error option has to read metadata documents, and which ones to read is the set of
candidate images, which the other options decide and which is not known when the run
starts. So the documents are read in batches, as the candidates are enumerated. A run
whose other options keep one image in a hundred reads a hundredth of the documents,
rather than every document under the volumes it selected. On a cloud results root that is
a hundredth of the downloads. Only images that already passed the listing are ever read,
which is also what makes every error option keep only images that have a document.

A filter answers from what the metadata document said at the moment the selection was
made. A document rewritten or deleted while the run is under way is not noticed for an
image the run has already selected.

Metadata Documents That Cannot Be Read
--------------------------------------

An error option needs to know what a document records. A metadata document that cannot be
read, that does not parse as JSON, that parses to something other than a JSON object, or
that was written to an earlier version of the metadata schema tells it nothing. What such
a document records is unknown rather than known, so its image satisfies no error option,
including ``--has-no-offset-error``. A results root holding nothing but documents from an
earlier schema therefore selects no image at all for any error option. Re-navigating
those images rewrites their documents to the current schema.

Such a document is still a file that exists, so ``--has-offset-file`` selects its image
and ``--has-no-offset-file`` passes over it.

When a run ends, its log reports how many candidate documents it could read nothing out
of and names one of them with the reason. A selection that is short for this reason says
so, rather than only coming back smaller than expected.

Results Directories That Cannot Be Listed
-----------------------------------------

The listing taken when a run starts covers the selected volumes and no others, and each
one is asked about separately.

A selected volume that has no directory under the results root contributes nothing. A
volume nobody has navigated yet has no directory there, and that is an ordinary state of
a results tree.

A directory that is there and cannot be listed ends the run instead. This user may not
have permission to read it, or the storage it lives on may have gone away. A filter
answering from a partial listing would silently select images for which it has no
evidence,
so the run stops and says which directory it could not read. Asking about one volume at a
time is what tells the two cases apart: a single request covering all of them would end
at the first unreadable volume, and every volume after it would go unasked.

Answering These Options From the Results Index
==============================================

Reading a results tree costs a listing per directory and, for an error option, a read per
candidate document. On a cloud results root each of those is a network round trip. The
results index is a database holding one row per navigated image, and a program given one
answers these six options from its rows without reading the results tree at all.

``sd_offset`` and ``sd_backplanes`` accept ``--results-index-db URL``; see
:doc:`user_guide_results_index` for the index itself and for how to build and refresh it.
A program that does not accept the option always reads the results tree.

A run given an index refuses to answer when the index holds no completed ingest of the
results root. An image that has no row otherwise reads as an image that was never
navigated, and for a root the index knows nothing about, that answer would be wrong for
every image under it.

Where the Results Index Answers Differently
-------------------------------------------

The results index holds what an ingest pass could read and record. Three things follow,
all of them worth knowing before a selection is trusted.

**The index answers as of its last ingest.** An image navigated since that pass has no
row, so ``--has-no-offset-file`` selects it again. A metadata document deleted since that
pass still has a row, so ``--has-offset-file`` selects an image whose document is gone.
The run log reports when the pass finished and how long ago that was, which is what says
whether either applies to your run. Run ``sd_results_index ingest`` to bring the index up
to date, or pass ``--results-index-db none`` for a run that must read the tree.

**An image that has no row in the results index reads as never navigated.** One kind of
ingest failure leaves a metadata document unrecorded, and that is a pass that could not
retrieve the file. We deliberately record nothing for such a file. A row for it would be
skipped for as long as the file did not change, and a download that failed once says
nothing that will still be true on the next pass. The other two ways a document could go
unrecorded leave no completed pass behind, so a completed ingest cannot contain them: a
pass that cannot list a directory stops there, and a pass whose document the database
refuses stops there. After a completed pass, every directory under the root was listed
and every metadata document under it was stored.

**A metadata document rewritten in place, keeping the length and modification time it had
before, keeps the row the earlier document produced.** Those two are everything a
directory listing says about a file, and they are how an ingest decides whether a
document needs re-reading, so an error option answers from the earlier document however
recently the last pass finished. A tree restored by a copy that preserves timestamps, a
document patched and stamped back from a sibling, and a storage backend reporting one
modification time for two writes all produce this. An ordinary re-navigation writes a
different length at a later time and does not. Run ``sd_results_index ingest --force``
over the root to re-read every document and put such a row right.
