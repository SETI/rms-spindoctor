# SpinDoctor PDS4 Draft Bundle Plan

*Implementation plan for the first PDS4 bundle this project can hand to the
RMS Node for review: one Cassini ISS Saturn backplanes bundle over a chosen
cohort, complete enough to pass the NASA PDS `validate` tool. Written to be
executed by an implementing model with no briefing beyond
`/seti/newnav/CLAUDE.md` and the repository itself. Conventions from
`CLAUDE.md` and `.cursor/rules/` apply throughout: line length 100, mypy
strict, pdslogger-only logging, Google-style docstrings with `Parameters:`,
Conventional Commits, one logical change per commit, modules under 1000
lines, no issue numbers in docstrings or `.rst` files (only in `#`
comments).*

Integration branch: `rf_pds4_draft_bundle`, cut from `main`. Each phase
below lands as its own pull request targeting that branch, so each gets an
independent review pass before the branch merges to `main`.

---

## 0. Status

Written 2026-09-08 from a verified end-to-end run of `main` at `0a5fe670`
rather than from reading alone: section 2 records what that run produced and
what it did not. The `rf_pds4_draft_bundle` branch was cut 2026-09-09 from
`main` at `bc103ffb`. `main` was merged into the branch on 2026-09-10 as
`7d12a974`, bringing #613.

Phases 1-3 have run; Phases 4-10 have not. Two changes landed ahead of
the phases, both because they must precede anything generated against them: the
rings dictionary bump recorded in section 3.9, and the masked-value change
recorded in section 3.13, which alters what the backplane arrays contain and
so has to be settled before a label describes one. A third, the statistics
units fix recorded in section 3.8, landed with Phase 2 rather than ahead of it,
and for the same reason: it alters what the metadata documents contain and
therefore what Phase 7's tables and labels are sized and written against.

This plan is the "finish and validate the Cassini path" half of #53, which
`plans/ENGINEERING_PLAN.md` (Track D, "PDS4 output bundles") lists as the
prerequisite for the generalization half. It does not generalize to the
other three instruments; that work begins when this plan's acceptance
criteria hold.

### 0.1 Status board

The authoritative record of what has run. A session resuming this work reads
this table first and trusts it over any recollection.

| Phase | State | Notes |
|---|---|---|
| Landed ahead: rings dictionary to `1F00` | **done** | `ed0b9e15`, section 3.9 |
| Landed ahead: backplane masked value `-999` | **done** | `04b84a62`, section 3.13 |
| 1 — Surface label-write failures | **done** | `937e6cf4`, the squash on `rf_pds4_draft_bundle`, section 4 |
| 2 — The synthetic cohort | **done** | `rf_pds4_phase2`, sections 3.12 and 4 |
| Landed with Phase 2: statistics compared by measure, each carrying its unit | **done** | `rf_pds4_phase2`, section 3.8 |
| 3 — Epochs | **done** | `rf_pds4_phase3`, section 3.4; #519 is closed by hand when its PR merges (section 8) |
| 4 — The FITS in the bundle, with its data objects | not started | |
| 5 — Inventories that conform | not started | |
| 6 — Bundle-level and static products | not started | |
| 7 — The miscellaneous collection and its global index labels | not started | |
| 8 — Targets, mission area, ring geometry | not started | |
| 9 — Parameterize the bundle name and version | not started | |
| 10 — Validation, the integrity pass, and the draft run | not started | |

Issues opened by this work, all open: #595 (LaTeX template for the user
guides), #596-#599 (the four instrument guides), #600 (what a bundle says
about images that did not navigate), #601 (the `Special_Constants`
declaration, which is what remains of the masked-value work), #602 (a
skipped or failed product leaves the bundle inconsistent, which Phases 5 and
6 own), #611 (the backplane viewer carries the same unit equality the
statistics carried, on the same plane), #614 (a dataset without PDS4 support
ends both passes in a traceback rather than a refusal). #603, the two passes
disagreeing about a missing template, was closed by hand on 2026-09-11, after
#605, Phase 1's PR, merged. #607, the index tables written to one precision
whatever the column's unit, closes in Phase 2 with a format per unit (section
3.8); the missing-value sentinel it raised beside that is Phase 7's. #519,
which predates this plan and found every data label's start and stop empty,
closes with Phase 3 (section 3.4).

Open questions, none blocking Phases 1-9: #600; whether this information
model build's dictionaries are registered, with the Engineering Node
(section 3.9); the cohort choice and the user-guide PDF for Phase 10.

---

## 1. Purpose and scope

A PDS4 bundle is not a directory of labels. It is a bundle product, four or
seven collection products, an inventory per collection, and a data file
beside every label that names one, all of which must resolve against each
other and against the PDS4 schemas. What the pipeline produces today is the
per-image half of that and nothing else, and nothing in the repository can
tell you so, because nothing validates anything.

The purpose of this plan is one reviewable artifact: a bundle rooted at
`cassini_iss_saturn_backplanes_rsfrench2027`, built from a real COISS
cohort, that the NASA PDS `validate` tool accepts with zero errors. That
artifact is what turns every remaining PDS4 question from a guess into a
review comment.

**In scope:**

- Every structural product the bundle label declares and the generator does
  not write: the bundle product itself, the readme, and the context,
  document and schema collections — plus the `miscellaneous` and
  `spice_kernels` collections the label does not yet declare, which are where
  the global index tables and the metakernel belong.
- The backplane FITS as an archived file with a described data object, in
  the bundle, beside its label.
- The label content that is empty, self-referential, or absent today:
  epochs, targets, mission-area attributes, ring geometry, the data object
  the display settings point at.
- Collection inventories that conform: correct name, no header, declared
  record delimiter, correct record count.
- Schema validation, wired into a repeatable command and into CI.
- One draft run over a real cohort, and the reconciliation of both guides to
  what it produces.

**Out of scope, deliberately:**

- **The other three instruments.** Voyager, Galileo and New Horizons each
  implement `pds4_bundle_template_dir` and `pds4_bundle_name`, over template
  directories that do not ship, and raise `NotImplementedError` from every
  other `pds4_*` hook, `pds4_required_templates` among them, so both passes
  stop on them, in a traceback, before looking for a template (#614 is to
  make that a refusal); `DataSetPDS4` raises throughout. Their template
  trees and hooks are the second half of #53 and are mechanical once a
  validated reference tree exists. This plan produces that reference. The
  Cassini cruise dataset has every hook but names `cassini_iss_cruise_1.0`,
  which does not ship either, so both passes refuse it at the
  missing-template check: as shipped, only `coiss_saturn` bundles.
- **PDS4 input** (#34). Unrelated to output bundles despite the shared
  acronym; no such archive exists to read.
- **The backplane set and HDU content decisions** (#55, #57, #54, #77).
  This plan describes whatever the generator writes; it does not decide what
  the generator should write. Section 3.8 is the closest it comes: it records
  why the arrays and the tables carry different angular units, because the
  labels have to state both and a later reader will otherwise take one of
  them for a mistake, and it carries the one change that made every angular
  column follow that rule.
- **Cloud-only operation** (#67). The `shutil.copy2` at
  `bundle_data.py:131` stays, and this plan adds a second local-path copy
  for the FITS. Both are recorded as #67's work, and the draft run is local.
- **Removing `sd_create_bundle_cloud_tasks`** (#424).
- **The backplanes user-guide PDF itself.** Section 3.6 wires the document
  collection so that dropping the PDF into the template directory completes
  it; writing the document is an operator deliverable, not a coding task.

---

## 2. Current state (verified 2026-09-08)

### 2.1 What runs

Both passes run clean on `main` and exit 0:

```bash
sd_create_bundle labels coiss_saturn N1702240231 \
  --nav-results-root /data/nav-offset-results/nav \
  --backplane-results-root /data/nav-offset-results/backplanes \
  --bundle-results-root <out>
sd_create_bundle summary coiss_saturn --bundle-results-root <out>
```

producing, for one image:

```text
cassini_iss_saturn_backplanes_rsfrench2027/
  data/1702xxxxxx/170224xxxx/1702240231n_backplanes.lblx    10537
                             1702240231n_supplemental.txt   14794
  data/collection_data.{tab,lblx}
  browse/1702xxxxxx/170224xxxx/1702240231n_summary.{png,lblx}
  browse/collection_browse.{tab,lblx}
  document/supplemental/global_index_{bodies,rings}.tab
  document/supplemental/global_index_{bodies,rings}.lblx         0 bytes
```

`coiss_saturn` is the only dataset whose `pds4_*` hooks are implemented.

### 2.2 The defects, and where they live

Each row is verified, not inferred. "New" means no open issue covers it and
one must be filed (`.cursor/rules` and the standing practice: future work,
decisions and known limits get an issue, referenced from the PR and the
plan).

| # | Defect | Location | Tracked as |
|---|---|---|---|
| 1 | The backplane FITS is never copied into the bundle. The label names a bare `1702240231n_backplanes.fits`, which must therefore sit beside it in `data/`; the file stays in `backplane_results_root`. | `bundle_data.py:108`, xfail-pinned at `test_bundle_data.py:319` | #265 (layout item), #69 |
| 2 | `bundle.lblx` is never written. The template exists and nothing references it: `grep -rn "bundle.lblx" src/ --include=*.py` is empty. | — | #265 area |
| 3 | Three of the five collections `bundle.lblx:204-227` declares — context, document, xml_schema — are never generated, though their `.lblx` and `.csv` templates ship in the template directory. Two more, `miscellaneous` and `spice_kernels`, are neither declared nor generated; section 3.1 adds both, moving the global index tables into the first and the metakernel into the second. | — | #72, #74 |
| 4 | `readme.txt` is never copied to the bundle root, and `bundle.lblx:196` declares a `File_Area_Text` over it. | — | #265 area |
| 5 | `<start_date_time>` and `<stop_date_time>` are empty in every data label and in `collection_data.lblx`. The code reads `observation.start_time`; the document holds `navigation_result.times.{start_et,stop_et,midtime_et}`. `collections.py:80-81` separately hardcodes the collection range to `''`. | `dataset_pds3_cassini_iss.py:589-593`, `collections.py:80-81` | #519 |
| 6 | `File_Area_Observational` has a `<File>` and no data object. The FITS carries a `PrimaryHDU` plus one `ImageHDU` per surviving backplane (9 HDUs on the verified frame); `writer.py:58-60` drops any plane with no valid pixels, so the set is per-image dynamic. | `data.lblx:174-183` | #69, #30 |
| 7 | `data.lblx:104` references `<local_identifier_reference>image</local_identifier_reference>`; no object in the label defines that identifier. Falls out of 6. |  `data.lblx:104` | **new** |
| 8 | `SOURCE_IMAGE_LIDVID` is set to the product's own data LIDVID, so every product cites itself as its source. | `dataset_pds3_cassini_iss.py:608` | **new** |
| 9 | Inventory filename mismatch: templates declare `collection_data.csv` / `collection_browse.csv`; the code writes `.tab`. The labels point at files that do not exist. | `collections.py:55,88` vs `collection_data.lblx:193`, `collection_browse.lblx:131` | #265 |
| 10 | Inventories carry a `Member Status,LIDVID_LID` header row, and `<records>` counts it. PDS4 collection inventories are headerless. | `collections.py:60,93` | **new** |
| 11 | Inventories are written CRLF (`csv.writer`'s default dialect) while the labels declare `<record_delimiter>Line-Feed</record_delimiter>`. Verified with `od -c`. | `collections.py:58,91` | **new** |
| 12 | `global_index_bodies.lblx` and `global_index_rings.lblx` templates are 0 bytes, so 0-byte labels are emitted. Their columns are config-driven and cannot be static. | template dir | #76 |
| 13 | The document product's LID is `…:document:backplanes-user-guide`, but `data.lblx:138`, `bundle.lblx:176` and `collection_document.csv` all drop the `:document:` segment. | three files | **new** |
| 14 | `cassini:ISS_Specific_Attributes` is an empty element. Meanwhile `pds4_template_variables` computes about thirty `cassini:*` variables that `data.lblx` never references — `grep -c "cassini:" data.lblx` is 4, all structural. | `data.lblx:94-100` | #53 list |
| 15 | No `Target_Identification` anywhere; no rings discipline area; no ring incidence angle in the label. `config_900_backplanes.yaml` already reserves `target_lids: {}` for the mapping. | `data.lblx:93,133` | #73, #79, #75, #47 |
| 16 | `geom:SPICE_Kernel_Files` names a metakernel `kernels.ker` that no bundle contains. | `data.lblx:115-131` | #53 list |
| 17 | Bundle name and `version_id` `1.0` are hardcoded throughout the templates, though config carries `bundle_name`. | templates | #71 |
| 18 | Nothing validates. No `validate` invocation, no schema check in CI, no `xmlschema` or `lxml` dependency in `pyproject.toml`. | — | #53 list |

None of these gets its own tracking issue. Every row is fixed by a named
phase of this plan, which carries the evidence and the disposition together;
an issue whose content is "see Phase 5" has no reader, and five more entries
in Track D's index means five more closes to reconcile on a branch where
every PR already re-conflicts `plans/PROGRAM_PLAN.md`. Defect 1 additionally
has an `xfail` and belongs to #265 and #69. The rows that *would* have
outlived this plan -- the ones true of shipped products whether or not a
bundle is ever built -- were the units pair. Section 3.8 records the
difference between the arrays and the tables as settled design rather than a
defect; the conversion implementing it tested the configured unit for equality
against `rad` and so left `rad/pixel` unconverted, and that half was a defect,
fixed ahead of the phases.

### 2.3 What this implies about order

Label-write failures hid the rest, which is why surfacing them is Phase 1 and
why nothing else could honestly precede it. `pdstemplate` reports an
unresolved variable through the `(errors, warnings)` pair `write` returns
rather than by raising, so a template edit that mistyped a variable produced
a product with no label -- or, for the error classes `pdstemplate` counts as
recoverable, a label carrying an embedded `[[[ ]]]` marker -- and a run that
said it succeeded either way. With that surfaced, every phase after it can
tell a label it rendered from one it did not.

---

## 3. Target design

### 3.1 The bundle tree

This is the authoritative layout. `docs/dev_guide/dev_guide_pds4.rst`
currently describes a different one and
`docs/user_guide/user_guide_pds4_bundle.rst` a third; Phase 10 reconciles
both to this.

```text
<bundle_name>/
  bundle.lblx
  readme.txt
  browse/
    collection_browse.csv
    collection_browse.lblx
    <path stub>/<image>_summary.png
    <path stub>/<image>_summary.lblx
  context/
    collection_context.csv
    collection_context.lblx
  data/
    collection_data.csv
    collection_data.lblx
    <path stub>/<image>_backplanes.fits
    <path stub>/<image>_backplanes.lblx
    <path stub>/<image>_supplemental.txt
  document/
    collection_document.csv
    collection_document.lblx
    user_guide/
      cassini-iss-saturn-backplanes-user-guide.pdf
      cassini-iss-saturn-backplanes-user-guide.lblx
  miscellaneous/
    collection_miscellaneous.csv
    collection_miscellaneous.lblx
    global_bodies_index.tab
    global_bodies_index.lblx
    global_rings_index.tab
    global_rings_index.lblx
  spice_kernels/
    collection_spice_kernels.csv
    collection_spice_kernels.lblx
    kernels.ker
    kernels.lblx
  xml_schema/
    collection_xml_schema.csv
    collection_xml_schema.lblx
```

That layout is not invented. It follows the F ring mosaics bundle -- the
closest existing product from this group, generated by
`/seti/research/f-ring/f-ring/pds4_bundle_gen/` and built at
`/data/fring-bundles/pds4/` -- which is a working, delivered bundle of the
same shape. Section 3.13 records what was taken from it and where this
bundle still differs.

Four decisions are recorded in that tree. **Collection inventories are
`.csv`** and **the global index tables are `.tab`** -- the two are different
kinds of file and PDS4 spells them differently, which the F ring bundle
already does. An earlier draft of this plan made them uniformly `.csv` for
one rule instead of two; that was wrong, and the reference settles it. The
FITS lives beside its label, because the label names it with no directory
part. And the document collection puts its guide in a `user_guide/`
subdirectory rather than loose in the collection root.

The global index tables live in a **`miscellaneous` collection of their
own**, not under `document/supplemental/` where the code writes them today.
They are not documents: they are derived tables a pipeline reads to select
images without opening a FITS, and PDS4 has a collection type for exactly
that. Checked against `PDS4_PDS_1O00`: `collection_type` must be one of ten
values, `Miscellaneous` is among them, and the bundle-level reference type
`bundle_has_miscellaneous_collection` is in the Schematron's controlled
list. The F ring bundle has the same collection, spelled the same way, with
its three `global_*_index.tab` products in it.

The tables become **products with LIDs** rather than loose files beside a
label, because a collection inventory lists its members:
`urn:nasa:pds:<bundle>:miscellaneous:global_bodies_index` and
`...:global_rings_index`, following the reference's `global_mosaic_index`
naming -- underscores, and `index` last. And `readme.txt` currently tells a
reader that the document collection holds the user guide "along with index
files that summarize information about all backplanes"; that sentence stops
being true and is edited with the move.

A **`spice_kernels` collection** is the other addition, and it closes a TODO
rather than adding scope: `data.lblx:115-131` already names a metakernel
`kernels.ker` in `geom:SPICE_Kernel_Files` with a comment saying it has not
been figured out. The reference figured it out -- a `Product_SPICE_Kernel`
with `kernel_type` `MK` over the metakernel, in its own collection, listed
by `collection_spice_kernels.csv`. Which kernels the metakernel names is a
question for the navigation side, not for this plan; that it has a home in
the bundle is settled here.

So the bundle has **seven collections**, and `bundle.lblx` grows two more
`Bundle_Member_Entry` blocks, with `bundle_has_miscellaneous_collection` and
`bundle_has_spice_kernel_collection`.

### 3.2 Rendered products versus copied products

Two kinds of file end up in a bundle, and the generator currently
understands only the first.

**Rendered** — a template plus variables, one per image or one per run:
`data.lblx`, `browse.lblx`, the seven collection labels, the two global-index
labels, `bundle.lblx`.

**Copied** — a file that ships in the template directory and belongs in the
bundle verbatim: `readme.txt`, `collection_context.csv`,
`collection_document.csv`, `collection_xml_schema.csv`, and the user-guide
PDF when it exists.

A new module `src/spindoctor/cli/pds4/bundle_products.py` owns both for the
run-level products, so `collections.py` keeps to collection inventories and
does not grow past its purpose. The summary pass calls it after
`generate_collection_files` and `generate_global_index_files`.

### 3.3 The FITS and its data objects

The labels pass copies `<stub>_backplanes.fits` from `backplane_results_root`
into the bundle `data/` directory beside the label, and `BACKPLANE_PATH`
names the copy, so the `FILE_BYTES`/`FILE_MD5`/`FILE_ZULU` calls describe the
archived file rather than the source.

The data object block is generated from the copied file. `astropy.io.fits`
gives everything the label needs without a second convention:
`hdu.fileinfo()` returns `hdrLoc` and `datLoc`, and the header carries
`NAXIS1`, `NAXIS2`, `BITPIX` and `BUNIT`. For each HDU the label gets a
`Header` (offset `hdrLoc`, size `datLoc - hdrLoc`) and, for every HDU past
the primary, an `Array_2D_Image` with `offset` `datLoc`, an `Element_Array`
whose `data_type` comes from `BITPIX` (`IEEE754MSBSingle` for -32,
`SignedMSB4` for 32 -- MSB because FITS is big-endian) and whose `unit`
comes from `BUNIT`, and two `Axis_Array` blocks named `Line` and `Sample`
with `elements` from `NAXIS2` and `NAXIS1`.

`Array_2D_Image` rather than the generic `Array_2D`, and `Line`/`Sample`
axis names, are what the F ring bundle's `data_reproj_img.lblx` uses for an
image-shaped array; there is no reason to differ.

`pdstemplate` supports `$FOR` / `$END_FOR` and `$IF` / `$ELSE` (verified in
the installed 2.4.0), so the XML stays in `data.lblx` and Python supplies a
list of per-HDU dictionaries as one template variable. The first
non-primary array carries `local_identifier` `image`, which is what
`data.lblx:104`'s display settings reference; if the operator would rather
the display settings point at a specific plane, that is a template edit
against a named HDU, not a code change.

A frame with no ring backplanes has no ring HDUs. The `$FOR` handles that
without a special case, which is the point of generating from the file
rather than from the config.

### 3.4 Epochs

`navigation_result.times` holds `start_et`, `stop_et` and `midtime_et` as
TDB seconds past J2000. `spindoctor/support/time.py` is the one rule that
turns one into UTC: `et_to_utc` writes the plain ISO spelling the observation
metadata and the statistics report use, and `et_to_pds4_utc` the spelling a
PDS4 label takes, `ASCII_Date_Time_YMD_UTC` with its trailing `Z`, to a given
number of decimals, rounded to the nearer or down or up. The module this plan
first named, `spindoctor/cli/stats/classify.py`, no longer exists; the
statistics report's two derived values, `date_from_image_et` and
`datetime_from_image_et` in `spindoctor/nav_records/derived.py`, carried the
second and third copies of `julian.iso_from_tai(julian.tai_from_tdb(...))`,
and now go through `et_to_utc` with the same digits, which leaves the report
byte-identical: regenerating the statistics fixture tree and its golden output
changes nothing. `julian` agrees with SPICE's `et2utc` to the millisecond over
the Cassini mission, both leap seconds included, and an integration test holds
the cohort's epochs to the leapseconds kernel. #519 asked for exactly this and
said so. The C-kernel report converts through `cspyce.et2utc` against the
kernel its generator furnishes, and is the one conversion outside the rule.

A data label states its exposure's start and stop to the millisecond, the
start rounded down and the stop up, so the interval stated contains the
exposure; `IMAGE_MID_TIME`, an instant, goes to the nearer millisecond. This
departs from the reference, which writes a product's times at whole seconds.
The reference's own delivery changelog (`CHANGELOG-review-to-final.md`, item 13)
says why it floors the start and ceils the stop: its review copy rounded to
the nearer second, and the intervals it stated missed their exposures. The
same rule is applied here at a millisecond, because a Cassini exposure is
often shorter than a second, and whole seconds would state a 5 ms exposure as
a window of one or two; a millisecond is also the precision the PDS3 label and
index record an image's start and stop to.

An image whose navigation never reached a solution has no `times` block;
section 3.11 says what happens to it, and the answer is that it never reaches
a label. A success document always has one, so the labels pass holds every
document it labels to recording it -- `spindoctor.cli.pds4.epochs.unrecorded_epoch`:
all three epochs, each a finite number, the stop no earlier than the start --
and fails an image whose document does not before anything is written for it,
the way it fails a statistic no index column can hold. The empty string is not
reachable.

The data collection label states the cohort's earliest start and latest stop,
at whole seconds as the reference's collection and bundle labels do, the start
rounded down and the stop up, so the range contains every product's own
interval at the millisecond it is written to. The summary pass already read
every supplemental file to build the global index, so the range is taken
there, in that same read, by an `EpochRangeScan`. The index therefore runs
before the collection files in `main_summary`, and returns the range in a
`GlobalIndexOutcome`, which the driver hands to `generate_collection_files`
and holds for `bundle.lblx`, which Phase 6 renders, without a second
computation. With no range to state -- no supplemental file, or one that
cannot be read or whose document the epoch check refuses -- the data
collection label is counted as not written, with an error saying why, and the
inventory table is still written. Running first, the index generator refuses a
bundle with no data directory itself, as the collection generator does, rather
than write its tables into a root the labels pass would then refuse.

### 3.5 Inventories

`.csv`, no header row, `\n` line terminator (`csv.writer(f,
lineterminator='\n')`, file opened with `newline=''`), one `P,<lidvid>` line
per member. `<records>` counts members. The `FILE_RECORDS` template function
counts lines, so it agrees once the header is gone.

Three inventories are **generated**, because their membership depends on
what the run produced: `collection_data.csv`, `collection_browse.csv`, and
`collection_miscellaneous.csv`, whose two members are the global-index
products of section 3.1.

Four are **copied verbatim** from the template directory (section 3.2),
because their membership is fixed: context, document, spice_kernels and
schema. Their `collection_*.csv` templates get the same treatment as the
generated ones -- no header, LF -- and `collection_context.csv`'s trailing
`# TODO` comment line is removed, since it is currently counted as a record.

Members carry an explicit version: the reference writes
`P,urn:...:miscellaneous:global_mosaic_index::1.0` and
`S,urn:nasa:pds:context:instrument:issna.co::1.2`, naming the actual
published version of each secondary product rather than leaving it open. Our
`collection_context.csv` currently has no `::` at all on any line, which has
to be filled in from the context products as they are registered. The one
exception the reference itself makes is its own `collection_context.csv`,
which is LID-only; both forms appear to pass, so prefer the versioned one
and let validation say otherwise.

A collection inventory also carries **`S` members**, not only `P`. The
reference's document and miscellaneous inventories both list the context
products and the external ISS data user guide as secondaries alongside their
own primaries, so an inventory is a statement about everything the
collection references, not just what it owns.

### 3.6 The document collection

`collection_document.csv` lists the backplanes user guide and the PDS3 ISS
Data User's Guide. The first is a product this bundle owns and must
therefore contain; the second is an external reference and stays `S`.

The user-guide PDF is an operator deliverable, tracked as #595 (a shared
LaTeX template for all four instruments' guides) and #596 (the Cassini guide
written from it; #597, #598 and #599 are the other three, which wait on
their instrument's half of #53). The code path is written so that the PDF's
presence in the template directory is what decides:

- PDF present: it is copied, `cassini-iss-saturn-backplanes-user-guide.lblx`
  is rendered, `collection_document.csv` lists it `P`, and
  `bundle.lblx` keeps its document `Bundle_Member_Entry`.
- PDF absent: the document collection still exists, because the global index
  tables live under it, but the user-guide product is omitted from the
  inventory and the run logs one warning naming the missing file.

The draft is acceptable either way; a bundle delivered to the Node is not.
Acceptance criterion 8 records that distinction.

Whichever way it goes, the LID gets its `:document:` segment back in all
three places that drop it (defect 13).

### 3.7 Targets and the mission area

`config_900_backplanes.yaml` already reserves `target_lids: {}` with the
comment "used for label target identification". That is the home.

For the draft, populate it by hand for the Saturn system bodies the cohort
actually contains — the backplane metadata's `bodies` keys name them, so the
required set is discoverable from the products, not guessed — plus Saturn's
rings. Each entry carries the PDS4 context LID, the target name and the
target type. `data.lblx` grows a `$FOR` over the targets present in this
image's backplane metadata, emitting one `Target_Identification` each.
Scraping the context products to maintain that table is #79 and stays a
follow-up; a hand table for one planetary system is a dozen lines and does
not block a draft.

`cassini:ISS_Specific_Attributes` is filled from the `cassini:*` variables
`pds4_template_variables` already computes and the template already ignores.
Which of the thirty belong in the label is a schema question, not a code
question: implement the ones the `PDS4_CASSINI_1O00_1800` schema requires,
plus those with a non-empty value from the index row, and let validation
settle the rest.

Ring geometry (#75) and the ring incidence angle in the label rather than as
a backplane (#47) go in the same phase, from `rings.backplanes` in the
backplane metadata.

### 3.8 Units: radians in the arrays, degrees in the tables

The FITS arrays are radians and say so in `BUNIT`. The statistics -- and
therefore the global index tables -- are degrees. Both per-source stages reduce
their planes through `spindoctor/cli/backplanes/statistics.py`, which converts
an angular plane and records the unit the resulting minimum and maximum are in
beside them, so the document states which of the two conventions each statistic
follows rather than leaving a reader to infer it from the plane's name.

What is compared there is the unit's measure, everything up to the first
solidus, and not the whole string: `rad` becomes `deg` and `rad/pixel` becomes
`deg/pixel`, while `km` and `km/pixel` are left alone. That generality is not
decoration. The rule was first written as an equality against the whole string,
and `ring_longitudinal_resolution` is declared `rad/pixel`, so it was the one
angular column of the tables published in radians per pixel while every other
angular column was degrees -- a table mixing units without saying so, which for
a human-readable product is worse than the precision loss #607 was opened for.

The formatting half of #607 closes here too. The tables are written with a
format per unit, from one public mapping in `collections.py`: three decimals
for `deg`, one for `km`, eight for `deg/pixel`, and five significant figures
for `km/pixel`, written positionally, never in exponent form, whose values
run from 6e-4 a hundred kilometres off Enceladus to 7e4 at the grazing limb
of a wide-angle frame (`70853` on W1629148548_1). The ceiling on all of them
is seven significant digits, because the arrays are float32 and a statistic
cannot carry more than the plane it was taken from; the formats are chosen
within that from what one pixel resolves. No format fixes a column's width:
values under one format differ in length, so Phase 7 sizes each field from
the widest value its column holds. A plane declared in a unit the mapping
cannot size fails either pass before it reads anything.

Radians is spelled `rad` and degrees `deg`, which is both what the
configuration writes and what the PDS4 units-of-angle vocabulary names, so the
unit Phase 7 puts in a label is a token that vocabulary has. An angle spelled
another way -- `mrad`, `arcsec` -- needs scaling as well as renaming and is not
converted; a test over the shipping configuration fails if one is ever
declared, which is the only place such an entry can appear. The operator's
ruling of
2026-09-09 is that the index tables are human-readable and everything in them
is degrees; the fix landed on `rf_pds4_phase2` ahead of the phases that consume
the statistics.

A backplane root can hold documents written before the conversion beside
regenerated ones -- a ring longitudinal resolution in `rad/pixel`, or a
statistic with no `units` key at all -- and one index column would then be
in two units with nothing saying so. The bundle stage reads the unit every
statistic now records: a document whose statistic for a configured plane is
not in the unit the configuration gives that plane fails its image before
anything is written for it, rather than being indexed. That is the
operator's decision of 2026-09-10, and the remedy is to regenerate the
backplanes. The summary pass makes the same comparison, from one module
both passes share, on each supplemental file it reads, which holds that
document, and refuses the run before writing either index table when one
disagrees. Such a file was written from a backplane document recorded before
the statistics carried their unit, or under another configuration -- the
likeliest case being a whole tree from one labels run that predates the
unit -- so the remedy is the labels pass's: regenerate the backplanes, then
the bundle into an empty directory. A plane the document holds that the
configuration does not declare is not compared. Beside that guard, both
passes refuse before reading anything a configuration that declares a
backplane in a unit the bundle cannot use -- a spelling the format mapping
lacks, or no `units` at all -- naming every such entry once, as the
missing-template check does. `sd_create_bundle_cloud_tasks` makes the same
check per task, through the same `unusable_units`, and returns the task as an
`unusable_unit` error having generated nothing: the per-document guard sees
only the planes a document holds, so without it every task would write labels
the summary pass then refuses.

A minimum or maximum that is not a finite number within the range of a
float -- NaN, an infinity, an integer too large for a float, or no number at
all -- is refused the same way, by the same module: it fails its image in the
labels pass and the run in the summary pass, since no column can hold it and
a blank in its place would say the plane measured nothing. That was decided
on 2026-09-10, after a NaN kilometers-per-pixel statistic in one supplemental
file was found to stop the index writer halfway through the bodies table
with a message naming neither the file nor the plane. The summary pass
renders every cell of both tables before it opens either, so no failure of
any kind leaves a table half-written.

`sd_backplane_viewer` still decides the same question the same way, and so
displays that one plane in radians per pixel. Nothing this plan generates goes
through it, so it is #611 rather than a phase; what it needs from here is to
call `statistics_units` instead of testing for a literal, and a ruling on the
name heuristic it carries beside that test for an HDU with no `BUNIT`.

**This is the design and it stays.** The two products have different
readers. A backplane array is consumed by software, which wants the unit its
trigonometry is already in and no conversion step it can get wrong. An index
table is read by a person deciding whether an image is worth opening, and a
latitude range of -88 to 81 tells them that where -1.54 to 1.42 does not.
Making them agree would cost one of the two readers the form it wants, to
satisfy a consistency no reader is asking for.

What follows for the labels, and what a later reader must not "fix":

- The `Array_2D` blocks Phase 4 generates state `unit` from the HDU's
  `BUNIT`, so an angular plane is labelled `rad`. The label describes the
  array, and the array is radians.
- The `Field_Character` blocks Phase 7 generates for the global index take
  their `unit` from the same config entry the column was built from, mapped
  through `statistics_units`, which is the function that produced the
  column's values. An angular column is labelled `deg`, a resolution in
  radians per pixel `deg/pixel`. Phase 7 calls it rather than reproducing
  it, so a label and the column it describes cannot disagree.
- So one bundle carries `unit="rad"` on an array and `unit="deg"` on the
  table summarizing it, deliberately. Both labels are correct about the file
  they describe, which is the only thing a label is required to be correct
  about.

Two places say so in prose rather than leaving it to be rediscovered: the
backplanes user guide and the backplanes developer guide, both of which carry
the difference and the recorded unit. The conversion site says it too --
`statistics.py` exists to hold the rule and its module docstring is where the
reasoning lives -- rather than reading as an incidental unit fix. What is still
section 3.6's operator deliverable is the user-guide PDF, which repeats it for
the bundle's own readers.

It is worth expecting the RMS Node to ask about it during review. The answer
above is the answer; the point of writing it down here is that it should be
given once, from the plan, rather than reconstructed under review.

### 3.9 Dictionary versions

Checked against `pds.nasa.gov` on 2026-09-09. Every declaration in
`data.lblx` resolves; one is superseded within the same information-model
build:

| Namespace | Template declares | LDD version | Current for `1O00` |
|---|---|---|---|
| `pds` | `PDS4_PDS_1O00` | IM 1.24.0.0 | yes (`1P00` is IM 1.25.0.0) |
| `rings` | `PDS4_RINGS_1O00_1F00` | 1.15.0.0, 2025-09-15 | yes, since the bump below (was `1E00` = 1.14.0.0) |
| `disp` | `PDS4_DISP_1O00_1510` | -- | yes, nothing newer |
| `geom` | `PDS4_GEOM_1O00_19B0` | -- | yes, nothing newer |
| `cassini` | `PDS4_CASSINI_1O00_1800` | -- | yes, nothing newer |

**The rings dictionary moved to `1F00`, applied 2026-09-09.** Both builds
are generated from IM 1.24.0.0, System Build 15.1, so it disturbed no other
declaration: an in-build dictionary bump, not an information-model change.
Two files, which are the whole of it -- a repository-wide search for `1E00`
and `::1.14` finds nothing else:

- `data.lblx`, the `xml-model` href and the `xsi:schemaLocation` entry
  (`.sch` and `.xsd`; both `1F00` resources were confirmed to resolve).
- `collection_xml_schema.csv`, whose `rings-xml_schema` LIDVID moved from
  `::1.14` to `::1.15` in step. That line names the dictionary the label
  points at, so leaving it behind would have had the bundle claim a schema
  it does not use.

A label regenerated afterwards carries the `1F00` declarations and renders
with no template errors.

One thing not to be confused by: the rings LDD changelog switched to
three-element semantic versioning between 1.12.0.0 and v1.13.0, but the
generated schema still stamps four elements internally (`version="1.15.0.0"`)
and the filename encoding is unchanged (`1E00` is 1.14, `1F00` is 1.15). The
switch is in the LDD source repository's own versioning, and nothing in a
label needs to spell three elements.

Its provenance, since it would otherwise be asked again: Matt Tiscareno
made the change believing three-element semantic versioning was what PDS
expected everywhere, and recalls -- vaguely, and as his own account -- a
later conversation with Jordan suggesting he may have been mistaken about
that. So the three-element spelling may or may not survive a future rings
release. Either way this bundle is insulated from it: what the labels and
the inventory carry is the filename encoding (`1F00`) and the registry
LIDVID (`::1.15`), and neither takes its spelling from the changelog.

**Open, and externally owned:** the registry returns 404 for
`urn:nasa:pds:system_bundle:xml_schema:pds-xml_schema_1.24.0.0` while the
1.23.0.0 product resolves, which is surprising with the information model
now at 1.26. Every LIDVID in `collection_xml_schema.csv` names a
`system_bundle:xml_schema` product, so acceptance criterion 6 -- every
reference resolves -- cannot be met for that collection until this is
understood. The operator is following up with the Engineering Node. Two
outcomes to be ready for: the 1.24 products appear and nothing changes, or
the bundle is built against a build whose dictionaries are registered, which
would move every declaration in the table above together.

### 3.10 Bundle name and version

`pds4_bundle_name()` already reads config. The templates hardcode the same
string in about a dozen LIDs, and `version_id` `1.0` alongside it. Both
become template variables supplied from config — `bundle_name` exists;
`bundle_version` is added beside it in `config_950_pds4.yaml`. This is #71,
and it lands late because doing it early means re-editing every template
touched by the earlier phases.

### 3.11 Images that were never navigated

`bundle_data.py:61-70` already skips an image whose `status` is not
`success`, with a warning. That stays. The consequence for a cohort is that
the bundle contains a subset of the images the selection named, and nothing
today says which or how many.

**Whether that is right is now #600**, filed as a decision rather than
settled here: exclude entirely, carry a global-index row with a status and
no product, or record the attempt as a product of its own. The F ring
reference excludes entirely (recorded on that issue), which is a precedent
and not an answer, because its exclusions are operational failures while
ours are scientific outcomes. Whatever #600 decides replaces this section.

Until then, and regardless: the integrity pass (#66) is the right answer for
reporting the gap outside the bundle, and this plan includes a minimal form
of it: a `--check-only` flag on the labels pass that reports,
per selected image, whether the navigation document, the summary PNG, the
backplane FITS and the backplane metadata all exist and whether the
navigation succeeded, and exits non-zero if any selected image is
incomplete. Choosing a cohort for the draft run is exactly this question,
so the flag pays for itself before the plan ends.

### 3.12 The synthetic cohort the tests run on

Every phase from here on asserts something about a rendered label, and none
of it can depend on holdings, on SPICE, or on a navigation run. A test that
needs a navigated Cassini frame to check that a `Target_Identification` block
appeared is a test that will not run in CI and will rot.

`tests/mini_nav_results/` is the package that builds navigation documents
**through the production writer** --
`build_metadata_from_result` and `build_timing_section` from
`navigate_image_files` -- over hand-constructed `NavResult` objects. It
touches no SPICE and no holdings, it covers three instruments, three
outcomes, BOTSIM pairs and gated features, and it derives every spacecraft
clock reading from the epoch beside it rather than inventing one.

That package is the nav half of the cohort, and it holds the cohort in a set
of its own rather than by extending the set it already held. The two sets are
selected for different things. The eight documents in
`results_tree_documents()` are chosen for what they make the *statistics
report* exercise -- three outcomes, four feature sources, a BOTSIM pair, a
suspect offset -- and the package docstring states that every document earns
its place there. An image added for bundle sharding earns
nothing in that report; it adds rows to a fixture whose whole value is that
each row is deliberate.

The stored golden output is cheap to regenerate and no one has signed it
off, so the cost of touching it is not the argument. The argument is that a
fixture selected for two unrelated criteria stops being legible for either.

So the package holds a **second document set**, `cohort_documents()`, built
from the same `shared.py` primitives and written to a cohort root rather
than into `RESULTS_TREE`. `results_tree_documents()` and the stats fixture
tree are untouched by it.

**The cohort is never checked in.** It is built once a session into a
directory `tmp_path_factory` makes, and torn down with it; no cohort bytes live under `tests/`, and the
`python -m tests.mini_nav_results cohort <outdir>` form writes wherever the
operator points it. The builders are the artifact, not their output. That is
what keeps a fifth instrument's cohort from costing the repository anything
-- adding one is a module beside `cohort_cassini.py`, and the FITS, the PNGs
and the tables it implies exist only while a test is running.

The two sets differ on this deliberately, and it is worth saying why rather
than leaving it to look like an inconsistency. The stats tree stays stored
because `test_results_tree_documents.py` holds the stored tree against what
the builders emit today, and that comparison is what reports a change in the
production writer instead of absorbing it. Generate that tree at test time
and the test compares the builders to themselves and can never fail. The
cohort has no such frozen counterpart -- what it feeds is a schema
validator, which is an external judge -- so storing it would buy nothing and
cost the repository a growing pile of binary fixtures.

Build it once per session rather than once per test: a session-scoped
fixture that writes the cohort into a temporary directory, since a dozen
label tests should not each rewrite a FITS.

What the cohort set holds beyond the documents:

- **A summary PNG per successful image.** A real PNG, small; the browse
  label states its byte size and checksum.
- **A backplane FITS per successful image.** A *real* FITS written by
  `astropy.io.fits`, 16x16 per plane, because Phase 4 reads `hdrLoc` and
  `datLoc` out of it through `fileinfo()` and the label states its size and
  MD5. The byte blob the current `xfail` test writes cannot serve that.
- **Backplane metadata beside each FITS**, whose `bodies` and `rings`
  statistics name the same backplanes the FITS carries, since the global
  index columns come from one and the arrays from the other.
- **`ImageFile`s carrying `index_file_row`**, because sixty-six of the
  seventy `cassini:*` template variables are read from the PDS3 index row and
  Phase 8 puts them in the label. No index file is parsed; the row is a dict,
  keyed by the index file's own column names.
- **Coverage the bundle stage cares about**, which the stats corpus has no
  reason to carry: two images whose numbers shard into *different*
  `1234xxxxxx/123456xxxx` directories, one image with ring backplanes and one
  without, and one image whose navigation did not succeed.

Three things about the cohort's products are known not to hold, and are
recorded here rather than only in a docstring, because each is a
property of the product a later phase describes rather than of the code
that writes it.

**A 16 by 16 frame records `INSTRUMENT_MODE_ID = FULL`.** `FULL` is a
claim about size and no Cassini mode value names a frame this small, so
there is no truthful value to record instead and the simplification
stands. It stops being purely internal at Phase 8, which puts
`cassini:instrument_mode_id` into the same label as Phase 4's 16-element
`Array_2D_Image` blocks: re-check it there, and decide whether the
cohort grows a full-size frame for one image or the label carries the
mode the row holds.

**One body per image, where a real frame often has several.** The merge
resolves overlapping bodies by nearest distance and aggregates a plane's
statistics over all of them, and a cohort with one body per frame cannot
tell a correct aggregation from one that reports the first body it
found. A second body is what Phase 7 and Phase 8 need, since that is
where a per-body inventory and the global index columns over it are
written; the backplane fixture takes a tuple of bodies already, so
adding one is a line in the image's declaration.

**The backplane products under `/data` are not ground truth for this.**
They were written before the masked value became `-999`, so every plane
in them reads 100% valid with a minimum of `0`, and a cohort adjusted to
match them would be adjusted to match a product the pipeline no longer
writes. What they are good for is what does not depend on the fill: the
HDU order, the shape of the metadata document, and which planes a frame
carries.

Two rules bind the additions.

**Epochs first, everything else derived.** #530 is the open record of what
happens otherwise: four Cassini documents in the statistics set carry clock
seconds taken from the image number rather than converted from the epoch
beside them. The response here is not a test that exempts those four. It is
a `shared.py` constructor that takes an epoch and returns the clock triple,
so a document built through it cannot carry an invented one, and a second
beside it derives the image number from the same epoch. The cohort set is
built entirely through both. Routing the existing four through it as well
is #530's own work -- a coordinated change to four documents, four
filenames, the `filtered` variant's image-number bounds and both goldens,
which belongs in a PR about the statistics fixtures rather than on a PDS4
branch.

The conversion those readings come out of is a line through two
correlation points read out of `cas00172.tsc`, which calibrates the rate
the clock runs at as well as where it started. One point does not: the
clock gains 6.5 ppm on ephemeris time, so a single anchor reads 0.6 s
off a day away and 9.1 s off at the far end of this cohort's own 16-day
span, and named one image for a second nine seconds from the one the
kernel gives it. Two points, measured against the kernel across that
span, are never more than half a tick out, and each cohort epoch
converts to exactly the tick the kernel returns for it. A cohort
reaching much further has to measure that again or take a third point;
the arithmetic is a line either way and no SPICE is called at build
time.

What holds it there is an `integration`-marked test that furnishes the
kernel and converts every cohort epoch again. It has to be that test and
cannot be one of the cohort's own: those compare a reading to a reading
and a name to the reading it came from, both derived here from one
function, so they report a hand-authored triple added later -- their
real job -- and an anchor moved by an hour leaves every one of them
green with every image renamed. Both the module and the guide say so, so
that nobody reads the green as more than it is.

**Production writers write the fixture.** `writer.py:write_fits` writes the
FITS and its metadata sidecar for real code, so it writes them for the
fixture too. So does `merge.py:merge_sources_into_master`, which is the
stage before it: what the fixture synthesizes is what a `Backplane` computes
-- one array and one mask per plane per source, and the range to each source
-- and the merge resolves those into the master arrays and the body identity
map that the writer writes. A fixture built by a second, parallel writer is a
fixture that stops describing the product the moment the real writer changes,
and the merge is where that first bit: it is the merge that decides the HDU
order, by inserting the body planes sorted and then the ring planes sorted,
and Phase 4 states every array's byte offset against that order and calls the
first one the image.

The package sits at `tests/mini_nav_results/`, beside `tests/shims/` and
`tests/cmatrix_helpers.py`, because a package two suites import should not
live inside one of them, and it takes the shorter name because it is a
miniature of what a navigation run leaves behind, which is what every
consumer of it wants it for. Getting it there was import-only -- the
documents it emits are unchanged, so the stats suite's stored golden output
did not move.

The backplane products it writes are not, strictly, navigation results: they
live under `backplane_results_root`, not `nav_results_root`. The name is
still the right one, because what the package models is the state of disk
after a navigation run and the stages that follow it, and no reader will
mistake a package under `tests/` for the naming of the roots themselves.
Its docstring says which roots it writes so the point does not have to be
re-derived.

The cohort builder lives in the same package rather than beside it, so there
is one name and one entry point. That entry point takes the set to write and
where to write it, in that order, for both sets alike -- an argument that
changes *which* files are written depending on whether a later argument is
present is exactly the surprise a fixture tool should not hold:

```bash
PYTHONPATH=src python -m tests.mini_nav_results results_tree \
    tests/spindoctor/cli/stats/data/results_tree
PYTHONPATH=src python -m tests.mini_nav_results cohort <outdir>
```

Both arguments are required in both forms. The stats path is spelled out
rather than defaulted so that regenerating a checked-in fixture tree is
something the operator asked for by name; it is written here and in the
package docstring so it can be copied rather than remembered.

What the `cohort` form feeds is the bundle stage's library entry points
and the tests over them, without waiting for a navigation run. It does
not feed `sd_create_bundle` itself, and saying that it did was wrong in
two ways, each independently sufficient. The first is fixed, and not on
this branch: `--pds3-holdings-root` belongs to `DataSetPDS3`, which
declares it among its selection arguments and reads it when it
enumerates (#613, closing #43), and it reached the integration branch
by merging `main` in. The labels subcommand declares nothing of its own
about holdings and reads nothing off the namespace; a dataset that is
not PDS3 never sees the option. The second is real work and is not done: PDS3
enumeration reads a volume's index table out of
`<holdings>/metadata/<set>/<vol>/`, and the cohort writes no index label
and no index table, so a selection by volume matches nothing. Growing
the cohort a minimal holdings tree -- a parseable index label and table,
and an image stub per row -- belongs to Phase 10, which is where the CLI
and `validate` workflow is actually needed, and is tracked as an issue
of its own.

### 3.13 The reference implementation, and where this bundle differs

`/seti/research/f-ring/f-ring/pds4_bundle_gen/` generates the F ring mosaics
bundle (`urn:nasa:pds:cassini_iss_fring_mosaics_rsfrench2025`), built at
`/data/fring-bundles/pds4/`. It is delivered, it has a DOI, and it is the
closest existing product from this group. Where this plan and that generator
disagree without a reason, the generator wins; the paragraphs above already
took its layout, its inventory conventions, its `Table_Character` index
tables and its `Array_2D_Image` arrays.

Four more things worth taking:

**Every schema URL is a template variable.** `BASIC_XML_METADATA` holds
`PDS4_RINGS_SCHEMA_XSD`, `PDS4_PDS_SCHEMA` and the rest in one dictionary,
and the templates substitute them. Our templates hardcode each URL in each
file, which is why the section 3.9 dictionary bump had to be a search and
replace across two files rather than a one-line edit. Fold this into Phase 9
with the bundle name and version, since it is the same parameterization
problem.

**A declared sentinel for absent data, and we are adopting it.** The
reference fills invalid pixels with `-999`, passes it to the label as a
variable, and carries a `SENTINEL_DESCRIPTION` explaining what an absent
pixel means. Our arrays fill with `0.0`, which is not ambiguous for the body
planes -- `BODY_ID_MAP` is nonzero exactly where a body claimed the pixel,
and a check over a real product finds no pixel that is `0.0` while the map
is nonzero -- but is unresolvable for the ring planes, which have no such
map because the ring merge never writes `body_id_map`.

Decided 2026-09-09: **every masked value becomes `-999`**, which is outside
the range of every plane written. That replaces two inference rules
(`BODY_ID_MAP != 0` for bodies, `RING_RADIUS != 0` for rings) with one
comparison that works on every plane, and lets the label declare it through
`Special_Constants` rather than explaining it in prose. `BODY_ID_MAP` keeps
`0` for unclaimed -- raised as the one place "every masked value" could be
read either way, and settled the same day: it is the mask rather than a
measurement, `0` is not a NAIF ID, and a sentinel there would make it the
one plane a reader has to special-case.

**Applied 2026-09-09, in the backplane generator.** `backplanes.masked_value`
in `config_900_backplanes.yaml` is the single source; the two
`master = np.zeros(...)` in `merge.py` became `np.full(..., masked_value)`;
`writer.py`'s plane-worth-writing test compares against it instead of `0.0`,
which also stops a plane whose only valid pixels are exactly `0.0` from
being dropped; and the per-source fills in `backplanes_bodies.py` and
`backplanes_rings.py` match, which incidentally ends a second divergence --
bodies filled with `0.0` while rings filled with `NaN`. Per-image statistics
are untouched, because they already run off the boolean masks
(`valid_values = full[full_mask]`) rather than off the fill.

`sd_backplane_viewer` moved with it, and is better for the change: it used
to take `BODY_ID_MAP != 0` as validity for body planes and `== 0` for ring
planes, the second of which marks empty sky valid. Both now ask the array
whether a pixel is measured, which is one rule and the right answer for
each.

What this plan still owes is the declaration: Phase 4 gives every
`Array_2D_Image` a `Special_Constants` block naming the configured value.

`collections.py`'s "TODO Need an appropriate sentinel value for missing
data" is the table-cell half of the same question. The same `-999` is the
obvious answer there and should be settled with it.

**DOIs are products of their own.** The reference carries `BUNDLE_DOI` and a
separate `USERGUIDE_DOI`, and its user-guide label fills a real `<doi>`
where ours has `TODO DOI`. Registering both is an operator step with the
node, not a coding step, and it should be started early rather than
discovered at delivery.

**Authors and editors are template variables**, not prose: the reference
carries an `AUTHORS` string and an `EDITORS` string naming the node staff
who reviewed the bundle. Ours has a single hardcoded `List_Author` block.

Three places where this plan deliberately does **not** follow the reference:

- `populate_template` there discards `template.write`'s `(errors, warnings)`
  return exactly as ours does. Phase 1 fixes that here; it is a defect the
  reference shares, not a convention to copy.
- The reference declares `PDS4_RINGS_1O00_1E00`. Section 3.9 moved us to
  `1F00`, so on this one point we are ahead of it, and the F ring bundle may
  want the same bump.
- The reference writes a product's start and stop at whole seconds, floored
  and ceiled. A data label here writes them to the millisecond, floored and
  ceiled there (section 3.4): the containment the reference chose for its
  delivery, at the precision a Cassini exposure needs. The collection range is
  written at whole seconds, as the reference writes its ranges.

---

## 4. Implementation phases

Each phase is one pull request onto `rf_pds4_draft_bundle`, with tests, and
with `docs/` and the plan files reconciled as the standing convention
requires.

### Phase 1 — Surface label-write failures

**Done, `937e6cf4` on `rf_pds4_draft_bundle`, the squash of the phase branch
with the review rulings that followed it applied.**

All six `template.write` call sites go through one helper,
`spindoctor.cli.pds4.labels.write_label`, which takes the label's `FCPath` in
the bundle -- never a local cache path standing in for it -- renders in
`pdstemplate`'s `mode='repair'` -- which saves a label that drew warnings but
never one that drew errors -- logs warnings at warning level and errors at
error level naming the label path, and returns whether the label is on disk.
That is the whole function, apart from clearing the label path first: repair
mode saves nothing when a render errors, so a summary pass run a second time
would otherwise leave the first run's label beside the inventory table this run
has already rewritten. `main_labels` establishes an empty directory for its own
pass, but it cannot establish one for a summary re-run.

A bundle is written into an empty directory. Before it processes any image,
`main_labels` requires `bundle_results_root / DATASET.pds4_bundle_name()` to
be empty or absent, and otherwise logs an error naming the directory and exits
1 having written nothing; a dry run is refused the same way. The program will
not clear the directory itself, so an operator re-running after a partial
failure clears it deliberately, and a bundle assembled out of two runs is
impossible rather than detected. The check is the local driver's alone:
`sd_create_bundle_cloud_tasks` runs many workers into one bundle root, so a
per-image check there would refuse every task after the first, and
`main_summary` reads the tree `main_labels` wrote, so it requires a populated
bundle. There is no `--force`: it would reintroduce exactly the state the
precondition removes.

`generate_bundle_data_files` returns a `BundleDataOutcome` of written,
skipped or failed rather than `None`. An image the bundle has nothing to
describe is skipped, which is not a failure and does not affect the exit
status: no navigation metadata document, a navigation status that is not
`success`, or no backplane metadata document. That is the ordinary state of a
selection made by volume; what a bundle should say about such images stays
open on #600. A document that is there and cannot be read still raises.

Treating absence as a skip is **provisional**, not ratified: it was reviewed
with the rest of Phase 1 and kept as it stands, and #600 is where it is
decided for good. A later session reading this should not take the skip rule
as settled.

A navigated image whose summary PNG is not in the navigation results is
failed rather than skipped. `navigate_image_files` writes that PNG before the
document that records the success and under the same condition, so a success
document with no PNG beside it is a broken input rather than an image without
a browse product; its data label is written and stays, its browse products are
not, and the run exits 1.

Both per-image labels are attempted before failed is returned, and so is
every collection and index label, so one run reports every label it could not
write rather than one per run. `generate_collection_files` and
`generate_global_index_files` each return the number of labels that failed;
the inventory and index `.tab` tables are written either way.

`main_labels` counts the images whose labels it did not write -- including an
image whose generation raised and a batch that did not hold exactly one image
-- carries on to the next image either way, and closes with a line giving
that count beside the images it labeled and the images it skipped, so a
selection that matched nothing reads as zero. A dry run reports what it would
have processed and counts nothing. `main_summary` sums the two returned
counts; each exits 1 when its count is not zero.
`sd_create_bundle_cloud_tasks` maps a failed product onto a `status: error`
result carrying `status_error: label_not_written`, with no retry.

A missing template is fatal, and both passes check for one up front. A
`pds4_required_templates()` hook on `DataSet`, beside the other `pds4_*`
hooks, names the templates that dataset's tree must carry for a given pass;
`main_labels` checks the per-image pass's, `main_summary` checks the summary
pass's, and either exits 1 naming each file that is not there before it has
written anything. Every product of a pass renders from the same directory, so
a missing `data.lblx` would otherwise fail identically for thousands of
images. The four `if <template>.exists():` guards in `collections.py` go with
it: they are what made a missing template invisible on that side, where the
other side already raised. `DataSetPDS3CassiniISS` is the reference
implementation, as it is for the rest of the `pds4_*` surface.

Two things this phase does not reach. `global_index_bodies.lblx` and
`global_index_rings.lblx` ship as zero-byte templates, so a healthy run
writes two zero-byte labels and the guarantee "the label is on disk" is
satisfied by a file that is not a label; Phase 7 is where those labels get
their content. And an image that got a data label and no browse label still
leaves the bundle internally inconsistent, because the browse inventory is
built from the data labels and so lists a product that is not on disk; the
labels pass counts that image, but the summary pass that follows inspects
nothing. That is #602, and Phases 5 and 6 own it. A skipped image is not that
case: it has no data label, so it is in none of the inventories, and what it
leaves is a bundle covering fewer images than the selection named.

Closes #603, and the swallowed-label-write part of #265.  #265's
inventory-filename part closes in Phase 5 and its dev-guide output-layout part
in Phase 10.

### Phase 2 — The synthetic cohort

**Done, `rf_pds4_phase2`.**

`tests/mini_nav_results/` is the fixture cohort section 3.12 describes: the
package moved out of the statistics suite that held it, with two document sets
built through the production writers. `results_tree_documents()` is the statistics fixture
tree, unchanged and still stored under `tests/spindoctor/cli/stats/data/`;
`cohort_documents()` is the three Cassini images bundle generation is asserted
against, and is never stored.

The cohort is two navigated images and one that is not. The two shard into
different `1234xxxxxx/123456xxxx` pairs at both levels, one carries ring
backplanes and one carries none, and each navigated image has a real
`astropy`-written FITS of 16x16 planes, the backplane metadata document beside
it, and a real summary PNG. Every image carries the index row an enumeration
hands on with it, keyed by the index file's own column names.

`backplanes.py` calls `writer.py:write_fits` with synthesized arrays rather
than writing a FITS of its own, so the fixture drifts when the product does. It
hands the writer an observation that reports itself simulated and carries an
inventory dict, which is all the writer reads before it stops asking about
SPICE.

Every clock reading and every image number is derived from one epoch, through
`shared.py`'s `cassini_sclk_triple` and `cassini_image_number`, both counted
from a line through two correlation points the mission clock kernel gives --
calibrating where the clock started and the rate it runs at -- and stamped onto
a result by `with_pointing_from_epoch`, which takes no clock argument at all.
Routing the four existing Cassini documents of the statistics set through the
same constructor is #530's own work: it is a coordinated change to four
documents, four filenames, the `filtered` variant's image-number bounds and
both goldens, and it belongs in a PR about the statistics fixtures.

`cohort.py` assembles the two roots and the `ImageFiles` list, and
`tests/conftest.py` serves the result as the session-scoped `mini_nav_cohort`
fixture. `python -m tests.mini_nav_results` takes the set to write and where to
write it, both required for both sets.
`tests/spindoctor/cli/pds4/conftest.py` keeps `FakePds4DataSet` for the
plumbing questions and gains `CohortBundleEnv`, which runs the registered
Cassini dataset over the shipped templates and is what the phases after this
one assert against.

Two things the phase does not reach. An enumeration reads only the index
columns it declares -- `FILE_SPECIFICATION_NAME` and `INSTRUMENT_ID` for
Cassini -- so the `cassini:*` template variables read from an index row are
empty on every real run today whatever the cohort holds; that is Phase 8's, and
is recorded there. And a dozen of those variables name columns the COISS index
has no such column for; the cohort uses the index's own names, so those render
empty in the fixture exactly as they do on a real image.

No issue closes here. This phase exists because the eight after it are
untestable without it.

### Phase 3 — Epochs

Done on `rf_pds4_phase3`. One ET-to-UTC rule in `spindoctor/support/time.py`
(`et_to_utc`, and `et_to_pds4_utc` for the PDS4 spelling), used by the
statistics report's `date_from_image_et` and `datetime_from_image_et` and by
`pds4_template_variables`. `START_DATE_TIME` and `STOP_DATE_TIME` read
`navigation_result.times` to the millisecond, floored and ceiled, and
`IMAGE_MID_TIME` to the nearer millisecond; an image whose success document
records no usable epochs is failed before anything is written. The data
collection range is taken in the global index's read of the supplemental files,
which now runs first, and written at whole seconds, rounded outward; with no
range the data collection label is counted as not written. The bundle label's
range is Phase 6's, from the same `GlobalIndexOutcome`.

Tests: known epochs to the strings `et2utc` writes for them, leap seconds
included, and an integration test converting every cohort epoch through the
kernel; the shipped data label over the cohort states each navigated image's
start and stop; a success document with no `times` block fails with nothing
written; the collection range over three supplemental files is the min and the
max; the shipped `collection_data.lblx` over the cohort states the range of its
two navigated images; a summary over no supplemental file writes no data
collection label.

Closes #519, by hand when its PR merges into `rf_pds4_draft_bundle` (section 8).

### Phase 4 — The FITS in the bundle, with its data objects

Copy the FITS into `data/`. Build the per-HDU descriptor list from the
copied file. `data.lblx` grows the `$FOR` block described in section 3.3,
with `local_identifier` `image` on the first array.

Each array also declares its masked value: a `Special_Constants` block whose
`missing_constant` is the sentinel from `config_900_backplanes.yaml`, per
section 3.13, so a reader masks on the label rather than on a convention.
This phase can be written before the generator emits `-999` -- it reads the
configured value either way -- but the draft bundle should be built from
backplanes that carry it.

Tests: a two-HDU fixture FITS produces two `Array_2D_Image` blocks with the
offsets `fileinfo()` reports; each declares the configured sentinel as its
`missing_constant`; a frame with no ring planes produces no ring arrays; the
display settings' referenced identifier is defined in the label.

Closes #69; contributes to #30.

### Phase 5 — Inventories that conform

`.csv`, headerless, LF, correct record counts, for the two generated
inventories and the three copied ones. `.tab` removed everywhere including
the global index tables.

Tests: the generated inventory has no header, ends every line with `\n`, and
`FILE_RECORDS` equals the member count.

Closes the inventory-filename part of #265 and the new format issue.

### Phase 6 — Bundle-level and static products

`bundle_products.py`: render `bundle.lblx`, copy `readme.txt`, copy the
three static inventories and render their three collection labels. Fix the
`:document:` LID segment in all three places. Fix `SOURCE_IMAGE_LIDVID` to
name the source Cassini ISS product rather than the backplane product — the
correct form comes from the PDS4 Cassini ISS bundle's data LID scheme, and
the value is derived in `pds4_template_variables` beside the other LID
builders.

`bundle.lblx` also grows two `Bundle_Member_Entry` blocks, taking it to
seven: `bundle_has_miscellaneous_collection` (Phase 7 supplies what it
points at) and `bundle_has_spice_kernel_collection`. The latter brings the
`spice_kernels` collection with it -- a `Product_SPICE_Kernel` label with
`kernel_type` `MK` over `kernels.ker`, and a static
`collection_spice_kernels.csv` naming it -- which closes the metakernel TODO
at `data.lblx:115-131`. Which kernels the metakernel lists is a navigation
question and is not settled here; the label and the collection are.

The document collection's guide moves into a `user_guide/` subdirectory, per
section 3.1. And `readme.txt` loses the clause placing the index files in
the document collection, which section 3.1 moved.

Tests: the summary pass over a one-image bundle produces every file section
3.1 lists; the seven `Bundle_Member_Entry` LIDs each resolve to a
`collection_*.lblx` that exists.

Closes #72, #74, and the new LID and source-product issues.

### Phase 7 — The miscellaneous collection and its global index labels

Move the global index tables out of `document/supplemental/` into a
`miscellaneous` collection of their own (section 3.1), and make them
products rather than loose files: each gets a LID
(`urn:nasa:pds:<bundle>:miscellaneous:global_bodies_index` and
`...:global_rings_index`) built from the bundle name beside the data and
browse LID builders.

The tables become **fixed-width `.tab`**, which is what the reference does
and what the label class requires. `global_mosaic_index.tab` there has a
comma-separated header line and data rows padded to a constant length (761
bytes, against a 1530-byte header), described by a `Table_Character` with
one `Field_Character` per column, a `Header` object of `$HEADER_LENGTH$`
bytes at offset 0, the table at that offset, and
`<records>$FILE_RECORDS(...)-1$</records>` so the header is not counted.
This is more work than emitting CSV -- the generator has to size every
column, pad to it, and carry byte offsets into the label -- and it is the
cost of matching the product the Node already has.

Replace the two 0-byte templates accordingly: `Field_Character` blocks from
a `$FOR` over the configured backplane list, the same list `collections.py`
builds the header from, so a config change moves table and label together.
Field `name`, `data_type`, `unit` and `description` come from the config
entry. Follow the reference's naming: namespace-prefixed where a field is a
dictionary attribute (`pds:logical_identifier`, `cassini:observation_id`,
`rings:minimum_corotating_ring_longitude`), bare where it is local to the
table (`percent_coverage`). `unit` is the degrees spelling per section 3.8,
which is not the unit the arrays carry. Data types come from the reference's
vocabulary: `ASCII_LID`, `ASCII_String`, `ASCII_Real`,
`ASCII_NonNegative_Integer`, `ASCII_Date_Time_YMD_UTC`.

A field's width comes from the widest value written in its column, not from
the format: the per-unit formats section 3.8 records fix a column's decimals
or significant figures, and values under one format differ in length
(`1.235` and `-89.999`; `0.00060000` and `70853`), so the generator sizes
each field by scanning the column it wrote. The column's unit and its format
-- a number of decimals, or a number of significant figures written
positionally -- come from the same public mapping in `collections.py` that
wrote it, so a field cannot describe the column in a form other than the one
it was written in. What a column says where an image has no statistic for a
plane -- the missing-value sentinel #607 raised beside the precision -- is
decided here as well, since it is the label that has to declare it; the
generator writes a blank there until then.

Then the collection itself: a `collection_miscellaneous.lblx` template with
`collection_type` `Miscellaneous`, and a generated
`collection_miscellaneous.csv` listing the two products, written after them
for the same reason the data inventory is written after the labels it lists.

Tests: adding a backplane to the config adds a column to the table and a
`Field_Character` to the label; `fields` matches the column count; every
`Field_Character` offset and length lands on the column it names in a
generated row; and the inventory's two LIDVIDs each resolve to a label in
the same collection.

Closes #76.

### Phase 8 — Targets, mission area, ring geometry

`target_lids` populated for the Saturn system. `Target_Identification` per
body present in the image's backplane metadata.
`cassini:ISS_Specific_Attributes` filled from the already-computed
variables. Ring geometry class fields and the ring incidence angle in the
label.

**The index row a run hands over holds two columns.** The enumeration reads
only the columns it declares -- `_INDEX_COLUMNS` plus `_INDEX_CAMERA_COLUMNS`,
which for Cassini are `FILE_SPECIFICATION_NAME` and `INSTRUMENT_ID`, plus the
four a BOTSIM grouping adds -- and `PdsTable` returns those alone, so every one
of the seventy `cassini:*` variables `pds4_template_variables` reads from
`index_file_row` takes its default on every real run. Declaring the columns the
label reads is this phase's, and so is the second half of it: a dozen of those
variables name columns the COISS index has no such column for
(`SPACECRAFT_CLOCK_COUNT_PARTITION` against the index's
`SPACECRAFT_CLOCK_CNT_PARTITION`, `FILTER1` and `FILTER2` against one
two-element `FILTER_NAME`, `GROUND_SOFTWARE_VERSION_ID` against
`SOFTWARE_VERSION_ID`, `START_TIME_DOY` and `STOP_TIME_DOY` against
`START_TIME` and `STOP_TIME`, and the `EXPECTED_MAXIMUM` / `VALID_MAXIMUM` /
`INST_CMPRS_RATE` pairs against one array column each). The cohort keys its
rows by the index's own names, so both halves are visible there rather than
papered over.

Tests: an image with two bodies emits two `Target_Identification` blocks; an
image with rings emits the ring geometry block and one without emits none.

Closes #73, #75, #47; contributes to #53's template list. #79 stays open.

### Phase 9 — Parameterize the bundle name and version

`bundle_version` in `config_950_pds4.yaml`; every hardcoded bundle name and
`version_id` in every template becomes a variable.

The five schema URLs go the same way, per section 3.13: one place holding
`PDS4_PDS_SCHEMA`, `PDS4_RINGS_SCHEMA_XSD` and the rest, substituted into
every template that declares them, so a future dictionary bump is one edit
rather than the search-and-replace section 3.9 needed.

Tests: a config naming a different bundle produces that name in every LID in
every rendered label — one test that walks the generated tree and asserts no
label contains the default name.

Closes #71.

### Phase 10 — Validation, the integrity pass, and the draft run

Schema validation as a repeatable command: `xmlschema` plus `lxml`'s ISO
Schematron against the five schemas the labels declare, over a generated
tree, added to `scripts/run-all-checks.sh` and to CI. The NASA PDS
`validate` tool is the authority for the draft acceptance and additionally
checks referential integrity, but it is Java and does not belong in this
repository's CI; the Python check is the gate that runs on every PR, and
`validate` is run once by hand for the draft.

`--check-only` on the labels pass, per section 3.11.

Then the draft run itself. Run it twice: first over the synthetic cohort,
which needs nothing but the repository and is where every schema error
should already have been found, and then over the real one. For the real
run: choose a cohort, navigate it, generate backplanes, generate the bundle,
run `validate`, and record the result. The
cohort should be one COISS volume — large enough that the collection
machinery is exercised over more than one directory, small enough to
regenerate in an afternoon when a review comment lands. The local tree at
`/data/nav-offset-results` is not it: 75 documents of which 17 are
`status == "success"`, and 2 backplane products.

Finally, reconcile `docs/dev_guide/dev_guide_pds4.rst` and
`docs/user_guide/user_guide_pds4_bundle.rst` to section 3.1, and the four
plan files as if this branch had merged.

Closes the dev-guide output-layout part of #265, the last of its three
parts, and #66; contributes to #53.

---

## 5. Acceptance criteria

1. `sd_create_bundle labels` followed by `sd_create_bundle summary` over a
   COISS volume produces a tree matching section 3.1 exactly — asserted by a
   test that walks the tree, not by inspection.
2. The NASA PDS `validate` tool reports zero errors over that tree. The
   command and its output are recorded in the branch's final PR.
3. The Python schema check in `scripts/run-all-checks.sh` reports zero
   errors over a bundle built from the synthetic cohort, and runs in CI.
4. The whole suite passes with no holdings mounted, no `SPICE_PATH`, and no
   network — the cohort supplies every input the PDS4 tests read.
5. Every `<file_name>` in every generated label names a file that exists in
   the same directory as the label.
6. Every `lid_reference` and `lidvid_reference` in the bundle resolves to a
   product in the bundle or to a PDS4 context product that exists.
7. No generated label contains an empty required element, and no generated
   label contains a `[[[` error marker.
8. A template referencing an undefined variable causes a non-zero exit and
   leaves no label on disk.
9. The document collection contains the user-guide PDF, or the run logs one
   warning naming it and the plan's delivery note records its absence.
10. `--check-only` over the chosen cohort reports every image's four inputs
   and exits non-zero if any selected image is incomplete.
11. Both guides describe the tree in section 3.1 and nothing else.

Criterion 2 is the one that matters. The other ten are how you get there
without discovering at the end that you cannot.

---

## 6. Risks and constraints

**The schema declarations are checked; the registry entries are not.** All
five dictionary URLs resolve and four of the five are current for `1O00`
(section 3.9, checked 2026-09-09), so a mismatched dictionary build -- the
whole-bundle failure that looks like a hundred element errors -- is ruled
out. What is not ruled out is the registry: `pds-xml_schema_1.24.0.0`
returns 404 where 1.23.0.0 resolves, and every LIDVID in the xml_schema
collection is that kind of product. If the Engineering Node's answer is that
this build's dictionaries are not registered, the bundle moves to one whose
are, and every declaration moves together. That is a Phase 1 edit repeated,
not a redesign, but it invalidates any label generated before it.

**Validation will find more than this plan lists.** Nineteen defects were
found by reading and running; a validator reads the schemas too. Phase 10
should be budgeted as discovery, not as a checkbox, and the phases before it
should not be tuned to a guess about what it will say.

**A synthetic cohort can agree with the code and not with the archive.**
Every test after Phase 2 runs on fixtures, so a label can be schema-valid,
self-consistent and wrong about a real Cassini frame. Two things bound that:
the cohort is written by the production writers, so it drifts when they do
rather than silently disagreeing, and criterion 2 is asserted on a real
cohort, not the synthetic one. Whether the values in the label are
*correct* is #232 and is not settled by anything in this plan.

**A zero-byte template legitimately produces a zero-byte label.**
`global_index_bodies.lblx` and `global_index_rings.lblx` ship as zero-byte
files and stay that way through Phase 6 by decision, so for that stretch
`sd_create_bundle summary` exiting 0 does not mean those two labels are real: a
healthy run writes both at zero bytes, `pdstemplate` reports no warnings and no
errors, and the run reports the labels as generated. That is honest reporting
of a template that renders to nothing, and it is deliberately not worked around
in the label writer -- having the writer second-guess its own template would
hide the defect rather than fix it. Phase 7 gives those two labels their
content and closes it.

**Two guides and four plan files change.** Every PR in this branch edits
`plans/PROGRAM_PLAN.md`, so each merge re-conflicts the rest. Take both
removals on a two-sided conflict.

---

## 7. Follow-ups

**No issues are filed for section 2.2.** Each row there is fixed by a named
phase of this plan, which holds the evidence, the location and the
disposition in one place; a tracking issue whose content is "see Phase 5"
adds a close to reconcile and no reader. Defect 1 additionally has an
`xfail` and belongs to #265 and #69.

The one row that would have outlived this plan was the angular-unit
difference between the arrays and the tables, and the difference itself
turned out not to be a defect: section 3.8 records it as the design, decided
2026-09-09, and both backplane guides now say so where a reader will meet it.
What was a defect was the conversion recognising only the literal `rad`,
leaving `ring_longitudinal_resolution` in radians per pixel in a table of
degrees; that is fixed, and the formatting half of #607 closes on the same
branch, with a format per unit that section 3.8 records; Phase 7 sizes its
columns from those formats. Nothing else there is left for someone else to
pick up.

If this branch is abandoned, section 2.2 is where the findings live. That is
a deliberate trade against five issues that would each close within the same
branch.

**Deferred, with the issue that carries them:**

- #595, #596, #597, #598, #599 — the backplanes user guides: one shared
  LaTeX template and one guide per instrument. #596 is what section 3.6's
  acceptance criterion 9 turns on for this bundle; the other three wait on
  their instrument's half of #53. #595 carries an open decision on where the
  LaTeX sources and the built PDFs live relative to the template directory.
- #600 — what the bundle says about images that did not navigate. Replaces
  section 3.11 when it is decided; nothing before Phase 10 depends on it.
- #601 — masked backplane values are `-999` as of 2026-09-09, applied on
  this branch (section 3.13), so one comparison masks every plane. What
  remains of it here is the `Special_Constants` declaration in the data
  label, which Phase 4 carries. The ring half of the original finding turned
  out to duplicate #251, which is the `xfail`-pinned record that ring-won
  pixels get no `BODY_ID_MAP` entry; the sentinel makes that gap harmless
  for consumers without closing it.
- #79 — scrape the PDS4 context products so `target_lids` is maintained
  rather than hand-written.
- #530 — the stats corpus's own Cassini clock seconds, which do not follow
  from their epochs. Phase 2 builds the epoch-first constructor that makes
  the defect unrepeatable and uses it for every cohort document. Routing the
  existing four through it renames four fixtures, moves the `filtered`
  variant's image-number bounds and regenerates both goldens; that is #530's
  own work and belongs in a PR about the statistics fixtures, not on a PDS4
  branch. It can land before, after or independently of this plan.
- #67 — cloud-aware bundle generation. This plan adds a second
  `get_local_path()`/`shutil` copy for the FITS; both copies are #67's work.
- #424 — remove `sd_create_bundle_cloud_tasks`.
- #53's generalization half — Voyager, Galileo and New Horizons template
  trees and `pds4_*` hooks, against this plan's validated Cassini tree as
  the reference.
- #232 — whether the label's geometry values are *correct*, as opposed to
  present and schema-valid. This plan makes a bundle that validates; it does
  not check a single number against independent truth.
- #30 — the backplane label design as a whole, of which Phase 4 implements
  the part the file forces.

---

## 8. Execution protocol

`rf_pds4_draft_bundle` is cut and pushed. One PR per phase onto it. For
each: implement, run `./scripts/run-all-checks.sh`, run the unit suite at
`-n 4` with the BLAS and OpenMP thread counts pinned to 1 (`export
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1`, set nowhere in the repository), open the PR, wait
for CodeRabbit to settle and reply on every comment with a disposition, then
merge. Update `docs/` and the four plan files in the same PR as the change
they describe, not afterwards. Update section 0.1 as each phase lands. The
issues a phase closes are closed by hand when its PR merges into
`rf_pds4_draft_bundle`, as the operator ruled on 2026-09-11, since GitHub's
closing keywords fire only on a merge into the default branch.

**How each phase is built.** A subagent implements it against the phase
text. Then **two** adversarial reviewers, neither of which wrote the code,
because they are looking for different things and one does not find the
other's defects:

- **A code reviewer**, reading the diff and the code it touches. Correctness
  and edge cases first: what happens on an empty selection, a single image,
  a plane with no valid pixels, a name that does not parse, a value that is
  `None` or `NaN` where a number was assumed. Probe the input domain
  deliberately rather than trusting coverage -- malformed-input defects are
  invisible to line and branch coverage and to mutation testing. Then the
  conventions: mypy strict with no new suppressions, no issue numbers in
  docstrings or `.rst`, positional-versus-keyword grouping, modules under
  1000 lines. And the tests themselves: a test that cannot fail is worse
  than no test, so check that each new one fails when the behaviour it
  claims to pin is broken.
- **A product reviewer**, reading what came out. Generate a bundle from the
  cohort, open the files, and hold them against the phase's acceptance
  wording and against the reference bundle at `/data/fring-bundles/pds4/`.
  Does the label name a file that exists, do the byte offsets land on the
  values they claim, does the inventory list what the collection contains.
  A green suite says nothing about whether the product is right.

The orchestrator reconciles all three, re-runs the gates on the final
revision, and does not accept "the tests pass" as the review. Where a review
finds something net-negative or unfixable, capture it on an issue and ask
rather than working around it.

Two lessons from this plan's own drafting, which cost time and should not be
repeated by an implementing session. Check the claim against the product
before writing it down: the masked-value finding was wrong for the body
planes and one line of numpy over an existing FITS would have shown it.
And read the whole path before concluding: that same finding missed
`merge.py`, where the ID map is written.

Phase 10's draft run needs the operator's cohort choice, and the user-guide
PDF if the draft is to be delivered rather than reviewed internally
(sections 3.6 and 10 of Phase 10). Neither blocks Phases 1-9. One question
is open and is not the operator's to answer alone: whether this information
model build's dictionaries are registered (section 3.9), which the
Engineering Node is being asked. It bears on acceptance criterion 6 and on
nothing before Phase 10. The angular-unit question that section 3.8 once
held open was settled 2026-09-09 in favour of degrees in the tables, which is
what the products did for every angular column but one.

The branch merges to `main` as a merge commit, not a squash, so the
individually reviewed phase PRs survive in the history.
