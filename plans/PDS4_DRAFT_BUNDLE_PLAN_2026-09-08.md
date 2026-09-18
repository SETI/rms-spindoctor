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

Phases 1-7 have run, Part A of Phase 8, Phase 9 and Part A of Phase 10. The
operator ruled on 2026-09-15 that this work is a prototype: it finishes on the
synthetic cohort, with the `TODO DOI` placeholders left in place, since the
DOIs and a fresh navigation will not come soon. Part B of Phase 10, the draft
run over a real COISS volume, moves to #708. What remains is Part B of Phase
8, the mission area, which waits on #698. Two changes landed ahead of
the phases, both because they must precede anything generated against them: the
rings dictionary bump recorded in section 3.9, and the masked-value change
recorded in section 3.13, which alters what the backplane arrays contain and
so has to be settled before a label describes one. A third, the statistics'
conversion to degrees, each statistic recording its unit (section 3.8), landed
with Phase 2 rather than ahead of it, and for the same reason: it alters what the metadata documents contain and
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
| 4 — The FITS in the bundle, with its data objects | **done** | `rf_pds4_phase4`, sections 3.3 and 3.13; #69 is closed by hand when its PR merges (section 8) |
| 5 — Inventories that conform | **done** | `rf_pds4_phase5`, section 3.5; #602 is closed by hand when its PR merges (section 8), #265 staying open for its Phase 10 part |
| 6 — Bundle-level and static products | **done** | `rf_pds4_phase6`, sections 3.1, 3.2, 3.5, 3.6, 3.9 and 3.13; #74 is closed by hand when its PR merges (section 8), #72 staying open for Phase 8's targets; the operator has since ruled on the source product, and Phase 7 applies the ruling (#678, section 3.13) |
| 7 — The miscellaneous collection and its global index labels | **done** | `rf_pds4_phase7`, sections 3.1, 3.4, 3.5, 3.8 and 3.13; #76, #601 and #678 are closed by hand when its PR merges (section 8), #601 and #678 by the operator's rulings of 2026-09-14 (section 3.13) |
| 8 — Targets, mission area, ring geometry | **Part A done**; Part B not started | Part A, the targets and the ring geometry, on `rf_pds4_phase8`, sections 3.5, 3.7 and 3.13; #73, #75, #47 and #72 are closed by hand when its PR merges (section 8). Part B, `cassini:ISS_Specific_Attributes`, is read from the navigation document's `observation` block, which #698 adds, for #684, on a branch against `main` by the operator's direction; it reaches this stack once #698 merges and `main` is merged into `rf_pds4_draft_bundle` (section 3.7), and it is what remains of this work |
| 9 — Parameterize the bundle name and version | **done** | `rf_pds4_phase9`, sections 3.9, 3.10 and 3.13; #71 is closed by hand when its PR merges (section 8) |
| 10 — Validation, the integrity pass, and the draft run | **Part A done**; Part B moved to #708 | Part A, the bundle check, the test that gates it and `--check-only`, on `rf_pds4_phase10`, its PR #707, sections 3.1, 3.6, 3.11 and Phase 10; #66 and #265 are closed by hand when its PR merges (section 8). Part B, the draft run over a real COISS volume, is deferred to #708 by the operator's ruling of 2026-09-15, since it needs the DOIs and a fresh navigation of the volume |

Issues opened by this work, all open: #595 (LaTeX template for the user
guides), #596-#599 (the four instrument guides), #600 (what a bundle says
about images that did not navigate), #601 (the masked value, whose
`Special_Constants` declaration Phase 4 made on the arrays and Phase 7 on the
index tables' statistic fields, which hold it where an image has no statistic;
the operator accepted that sentinel for the tables on 2026-09-14, and it is
closed by hand when Phase 7 merges), #602 (a
skipped or failed product leaves the bundle inconsistent), which is Phase 5's
by the operator's ruling of 2026-09-14 that every data product needs a browse
product, and is closed by hand when Phase 5 merges, #611 (the backplane viewer
decides degrees from `BUNIT` and the plane's name rather than through
`statistics_units`, so it shows the `rad/pixel` plane in radians per pixel),
#614 (a dataset without PDS4 support
ends both passes in a traceback rather than a refusal), #677
(which SPICE kernels the bundle's metakernel lists, a navigation question
Phase 6 left the metakernel empty for), #678 (what a data label names as
its source product, the calibrated image the navigation reads having no PDS4
counterpart; the operator ruled on 2026-09-14 that it is cited as an external
source product, which Phase 7 applies, and it is closed by hand when Phase 7
merges). #603, the two passes
disagreeing about a missing template, was closed by hand on 2026-09-11, after
#605, Phase 1's PR, merged. #607, the index tables written to one precision
whatever the column's unit, closes in Phase 2 with a format per unit (section
3.8); the missing-value sentinel it raised beside that is the masked value
Phase 7 writes (section 3.13), which the operator accepted on #601. #519,
which found every data label's start and stop empty, closes with Phase 3
(section 3.4). #69, which asked for the FITS to be described in its data label,
closes with Phase 4 (section 3.3).

Open questions, none blocking what remains: #600; whether this information
model build's dictionaries are registered, with the Engineering Node (section
3.9); and which kernels the metakernel lists (#677). The cohort choice, and
the user-guide PDF for a delivered bundle, go with #708.

---

## 1. Purpose and scope

A PDS4 bundle is not a directory of labels. It is a bundle product, four or
seven collection products, an inventory per collection, and a data file
beside every label that names one, all of which must resolve against each
other and against the PDS4 schemas. What the pipeline produces today is the
per-image half of that and nothing else, and nothing in the repository can
tell you so, because nothing validates anything.

The purpose of this plan is one reviewable artifact: a prototype bundle
rooted at `cassini_iss_saturn_backplanes_rsfrench2027`, built from the
synthetic cohort, whose only findings under the bundle check and the NASA
PDS `validate` tool are the ones section 5 names, the `TODO DOI`
placeholders chief among them. That artifact is what turns every remaining
PDS4 question from a guess into a review comment. The same bundle built from
a real COISS volume, its DOIs registered, which `validate` accepts with zero
errors, is #708's.

**In scope:**

- Every structural product the bundle label declares and the generator does
  not write: the bundle product itself, the readme, and the context,
  document and schema collections — plus the `miscellaneous` and
  `spice_kernels` collections, which are where the global index tables and the
  metakernel belong.
- The backplane FITS as an archived file with a described data object, in
  the bundle, beside its label.
- The label content that is empty, self-referential, or absent today:
  epochs, targets, mission-area attributes, ring geometry, the data object
  the display settings point at.
- Collection inventories that conform: correct name, no header, declared
  record delimiter, correct record count.
- Schema validation, wired into a repeatable command and into CI.

**Out of scope, deliberately:**

- **The run over a real volume** (#708). Navigating a COISS volume afresh,
  registering the DOIs, and building its bundle for `validate` to accept
  with zero errors wait on what this work does not have, so the work
  finishes as a prototype over the synthetic cohort (section 0).
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
  for the FITS. Both are recorded as #67's work.
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
| 1 | `SOURCE_IMAGE_LIDVID` is set to the product's own data LIDVID, so every product cites itself as its source. The calibrated image the navigation reads has no PDS4 counterpart to name instead (section 3.13). | `dataset_pds3_cassini_iss.py:688` | #678; fixed by Phase 7, which applies the operator's ruling and cites the calibrated image as a `Source_Product_External` (section 3.13) |
| 2 | `cassini:ISS_Specific_Attributes` is an empty element. Meanwhile `pds4_template_variables` computes about thirty `cassini:*` variables that `data.lblx` never references — `grep -c "cassini:" data.lblx` is 4, all structural. | `data.lblx:94-100` | #53 list; Part B of Phase 8 fills it from the navigation document's `observation` block (#684, section 3.7) |
| 3 | No `Target_Identification` anywhere, though the data label's schema requires one and the PDS4 Schematron one in the bundle label, in the data collection label (a Mission Science Data collection, whose references are `collection_to_target`) and in a `Product_SPICE_Kernel`; no rings discipline area; no ring incidence angle in the label. `config_900_backplanes.yaml` already reserves `target_lids: {}` for the mapping. Fixed by Part A of Phase 8: the table is filled, every label the schemas require a target of names its targets from it, a data label states its ring geometry and the incidence angle the backplane stage now records, and the context inventory lists the targets (section 3.7). | `data.lblx:93,130`, `bundle.lblx`, `collection_data.lblx`, `kernels.lblx` | #73, #75 and #47, closed by Phase 8; #79 stays open |
| 4 | Bundle name and `version_id` `1.0` are hardcoded throughout the templates, though config carries `bundle_name`. | templates | #71; fixed by Phase 9, which sets the name, the version and the schema locations in the configuration and has every template take them as variables (section 3.10) |
| 5 | Nothing validates. No `validate` invocation, no schema check in CI, no `xmlschema` or `lxml` dependency in `pyproject.toml`. | — | #53 list; fixed by Part A of Phase 10: `sd_create_bundle check` holds a written bundle to the shipped XML schemas and Schematron, reads each table through its label and checks the tree as a whole, with `lxml`, `elementpath` and `xmlschema` runtime dependencies, and a test in the default suite gates it over the synthetic cohort; `validate` 4.2.0, run by hand over the same cohort, agrees with it |
| 6 | A navigated image whose navigation recorded no pointing has no `navigation_result.times`: `build_metadata_dict` writes the times only beside a pointing, and the navigation records a success with no pointing when `compute_pointing` raises `NavPointingError` or the instrument has no SPICE camera frame mapped. Its data label has no start or stop to state, so the labels pass fails the image with nothing written. | `curator.py:353-355`, `orchestrator.py:512-538` | #619, closed by #624, which records the times in the `observation` block; fixed by Phase 7, whose labels pass reads them from there, and fails the image of a document an earlier version wrote, which records none there, until it is navigated again (section 3.4) |
| 7 | The supplemental file ends without a line feed after its last line (`json_as_string` writes none), and its label declares a `Stream_Text` with `Line-Feed` records. The Standards Reference requires a delimiter after a delimited table's last record (section 4C.1) and says nothing of the kind for `Stream_Text`; `validate` 4.2.0 accepts the last line as it is, with data-content validation on. | `bundle_data.py`, `data.lblx` | Phase 10; not a defect, by `validate` 4.2.0's run over the cohort (Part A of Phase 10), so nothing changes |

No row but 1 and 6 gets its own tracking issue. Each of the others is fixed by a
named phase of this plan, which carries the evidence and the disposition together;
an issue whose content is "see Phase 5" has no reader, and five more entries
in Track D's index means five more closes to reconcile on a branch where
every PR already re-conflicts `plans/PROGRAM_PLAN.md`. Row 6 was the
navigation's, and was tracked as #619, which #624 closed by recording the
host's exposure times in the `observation` block; Phase 7 takes every time the
bundle states from that block, which fixes the row. Row 1 was Phase 6's, which
stopped on it: the image the navigation reads has no PDS4 counterpart to name,
and what a data label names instead was a decision for the operator, tracked
as #678; the operator ruled on 2026-09-14, and Phase 7 applies the ruling
(section 3.13). The rows that *would* have
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

This is the authoritative layout. Both guides,
`docs/dev_guide/dev_guide_pds4.rst` and
`docs/user_guide/user_guide_pds4_bundle.rst`, describe the tree the two
passes write, which is this one, in this order. Part A of Phase 10 reconciled
both to it; the run over a real volume (#708) reconciles them again, should
it change the tree.

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
    user_guide/                  # only with the guide's PDF
      cassini-iss-saturn-backplanes-user-guide.pdf
      cassini-iss-saturn-backplanes-user-guide.lblx
  miscellaneous/
    collection_miscellaneous.csv
    collection_miscellaneous.lblx
    global_bodies_index.tab
    global_bodies_index.lblx
    global_rings_index.tab       # only with a ring row
    global_rings_index.lblx      # only with a ring row
  spice_kernels/
    collection_spice_kernels.csv
    collection_spice_kernels.lblx
    kernels.ker
    kernels.lblx
  xml_schema/
    collection_xml_schema.csv
    collection_xml_schema.lblx
```

Two parts of it are conditional: `document/user_guide/` is written only when
the template directory holds the guide's PDF (section 3.6, criterion 9), and
the rings index table and its label only when an image the data collection
holds gives that table a row (Phase 7). Both guides and the built trees say
so.

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
own**, not under `document/supplemental/`, where the code wrote them until
Phase 7.
They are not documents: they are derived tables a pipeline reads to select
images without opening a FITS, and the PDS4 standard has a collection type for
exactly that. Checked against `PDS4_PDS_1O00`: `collection_type` must be one of ten
values, `Miscellaneous` is among them, and the bundle-level reference type
`bundle_has_miscellaneous_collection` is in the Schematron's controlled
list. The F ring bundle has the same collection, spelled the same way, with
its three `global_*_index.tab` products in it.

The tables become **products with LIDs** rather than loose files beside a
label, because a collection inventory lists its members:
`urn:nasa:pds:<bundle>:miscellaneous:global_bodies_index` and
`...:global_rings_index`, following the reference's `global_mosaic_index`
naming -- underscores, and `index` last. And `readme.txt` told a
reader that the document collection holds the user guide "along with index
files that summarize information about all backplanes"; Phase 6 removed that
clause, and Phase 7 added a sentence saying what each index table has a row
for, that a table no image gives a row is left out, and that the tables are in
the miscellaneous collection, so that it holds whichever tables the bundle has.

A **`spice_kernels` collection** is the other addition, and it closed a TODO
rather than adding scope: the data label named a metakernel `kernels.ker` in
`geom:SPICE_Kernel_Files` with a comment saying it had not been figured out.
The reference figured it out -- a `Product_SPICE_Kernel` with `kernel_type`
`MK` over the metakernel, in its own collection, listed by
`collection_spice_kernels.csv` -- and Phase 6 followed it. Which kernels the
metakernel names is a question for the navigation side, not for this plan
(#677); that it has a home in the bundle is settled here. Until
the question is answered, the metakernel lists no kernels and every label
describing it says so.

So the bundle has **seven collections**, and `bundle.lblx` grew two more
`Bundle_Member_Entry` blocks, each with the collection it names:
`bundle_has_spice_kernel_collection` in Phase 6 and
`bundle_has_miscellaneous_collection` in Phase 7, so that every entry the
bundle label declares resolves at every phase; all seven do.

### 3.2 Rendered products versus copied products

Two kinds of file end up in a bundle, and the generator currently
understands only the first.

**Rendered** — a template plus variables, one per image or one per run:
`data.lblx`, `browse.lblx`, the seven collection labels, the two global-index
labels, `kernels.lblx`, the user-guide label, `bundle.lblx`; and `readme.txt` and
the four inventories the template directory ships, `collection_context.csv`,
`collection_document.csv`, `collection_spice_kernels.csv` and
`collection_xml_schema.csv`, which name the bundle's own products through the
variables every template is handed (section 3.10), the XML schema inventory
listing the configured schemas (section 3.9). `collection_document.csv` and
`collection_spice_kernels.csv` are written as they render when the label of their
primary member -- the user guide's, the metakernel's -- is written, and without
their `P` lines when it is not (section 3.6); the SPICE kernel inventory, then
empty, is not written at all (section 3.5).

**Copied** — a file that ships in the template directory and belongs in the
bundle verbatim: `kernels.ker`, and the user-guide PDF when it exists.

`src/spindoctor/cli/pds4/bundle_products.py` owns both for the run-level
products, so `collections.py` keeps to collection inventories and does not
grow past its purpose; `global_index.py` writes the global index tables, their
labels and the miscellaneous collection. The summary pass calls it last, after
`generate_global_index_files` and `generate_collection_files`, and the index
generator clears its products with its own once it has found the bundle's
data directory, before it reads any supplemental file.

### 3.3 The FITS and its data objects

The labels pass copies `<stub>_backplanes.fits` from `backplane_results_root`
into the bundle `data/` directory beside the label, and `BACKPLANE_PATH`
names the copy, so the `FILE_BYTES`/`FILE_MD5`/`FILE_ZULU` calls describe the
archived file rather than the source.

The data object block is generated from the FITS by
`spindoctor/cli/pds4/data_objects.py`, which reads the source in
`backplane_results_root` before the copy is made; the copy is byte-identical,
so the source's description is the copy's. `astropy.io.fits` gives everything the
label needs without a second convention: `hdu.fileinfo()` returns `hdrLoc` and
`datLoc`, and the header carries `NAXIS1`, `NAXIS2`, `BITPIX` and `BUNIT`. For
each HDU the label gets a `Header` (offset `hdrLoc`, size `datLoc - hdrLoc`,
parsing standard `FITS 3.0`) and, for every HDU past the primary, an
`Array_2D_Image` with `offset` `datLoc` and `axis_index_order` `Last Index
Fastest`, an `Element_Array` whose `data_type` comes from `BITPIX`
(`IEEE754MSBSingle` for -32, `SignedMSB4` for 32 -- MSB because FITS is
big-endian) and whose `unit` comes from `BUNIT`, and two `Axis_Array` blocks
named `Line` and `Sample` with `elements` from `NAXIS2` and `NAXIS1`. Each
array's `local_identifier` is its HDU name in lower case: `body_id_map`,
`body_latitude`, and so on. The allowed values of `parsing_standard_id`,
`data_type` and `axis_index_order` are the Schematron's, not the XSD's, which
types all three as plain strings; `FITS 3.0` and `FITS 4.0` are both allowed,
and 3.0 is stated because every construct the writer uses is in it. Under the
full rule set of all five dictionaries' Schematron (Phase 10), the cohort's data
labels fail no rule.

Every float array declares the configured masked value as the
`missing_constant` of a `Special_Constants` block, read from
`backplanes.masked_value` rather than written as a literal (section 3.13).
A test holds the shipped value to a finite number a 32-bit float holds
exactly, since every masked pixel holds it as one; nothing checks the
configuration at run time. `BODY_ID_MAP` declares none, because its `0` is the mask rather than a
missing measurement; its `description` says instead that `0` marks a pixel no
body claimed and every other value is the NAIF ID of the body that did. Every
float array carries a `description` too, as the reference's arrays do: the
plane's name, the `oops` backplane method the configuration names for it, its
unit, and a sentence saying that a pixel the plane does not cover holds the
`missing_constant` value -- nothing about the geometry the method computes.

The builder describes what `write_fits` writes and refuses nothing; the
cohort tests, which run the real writer and hold every stated offset to the
file's bytes, catch a change to the writer. `BITPIX` is mapped to its PDS4
data type by a plain lookup.

`Array_2D_Image` rather than the generic `Array_2D`, and `Line`/`Sample`
axis names, are what the F ring bundle's `data_reproj_img.lblx` uses for an
image-shaped array; there is no reason to differ.

`pdstemplate` supports `$FOR` / `$END_FOR` and `$IF` / `$ELSE` (verified in
the installed 2.4.0, on a rendered label), so the XML stays in `data.lblx`
and Python supplies the per-HDU descriptors as one template variable,
`BACKPLANE_FITS`.

**Display settings: one block per array.** `data.lblx` generates one
`disp:Display_Settings` per array, in a `$FOR` over the same descriptors,
each referencing its own array's identifier. A single block would declare
one array's orientation -- with the writer's order, `BODY_ID_MAP`'s whenever
a body is in view -- and leave every other array's undeclared. The display
dictionary allows several: `Discipline_Area` takes any number of
dictionary elements, `disp:Display_Settings` is a global element of
`PDS4_DISP_1O00_1510`, and that dictionary's Schematron constrains each
block -- its display axes must name the referenced array's, its reference
must resolve -- and not their count.

A frame with no ring backplanes has no ring HDUs. The `$FOR` handles that
without a special case, which is the point of generating from the file
rather than from the config.

The label describes the supplemental file as well, as a `Stream_Text` over
the whole file, from offset 0 for the length the label states: `7-Bit ASCII
Text` with `Line-Feed` records, holding one JSON object.
The Schematron allows both values. The file is written as the ASCII bytes
`json_as_string` produces, which escapes every character outside ASCII and ends
lines in a line feed, so the line feeds stay line feeds on any platform. The
reference describes its supplemental text files as a `Header` of `UTF-8 Text`
over their heading, followed by a table; ours is JSON with no heading. It ends
without a line feed after its last line, which section 2.2 row 7 leaves to
Phase 10's `validate` run.

### 3.4 Epochs

Every exposure time the bundle states comes from the navigation document's
`observation` block (the operator's direction of 2026-09-14): `start_time_et`
and `end_time_et`, the exposure the instrument host publishes
(`get_public_metadata`), in TDB seconds past J2000. `build_metadata_from_result`
writes the block for every image whose navigation ran to a result, pointing or
not (`navigate_image_files.py` 636-641), and the supplemental file carries the
whole document, so both passes read the same block. `spindoctor/support/time.py`
is the one rule that turns one into UTC: `et_to_utc` writes the plain ISO spelling the observation
metadata and the statistics report use (the report's `date_from_image_et` and
`datetime_from_image_et` in `spindoctor/nav_records/derived.py`), and
`et_to_pds4_utc` the spelling a PDS4 label takes, `ASCII_Date_Time_YMD_UTC`
with its trailing `Z`, to a given number of decimals, rounded to the nearer or
down or up. `julian` agrees with SPICE's `et2utc` to the millisecond over the
Cassini mission, both leap seconds included, and an integration test,
`tests/integration/test_cohort_cassini_epochs_against_kernel.py`, holds the
Cassini cohort's epochs to the leapseconds kernel. #519 asked for exactly this
and said so. The C-kernel report converts through `cspyce.et2utc` against the
kernel its generator furnishes, and is the one conversion outside the rule.

A data label states its exposure's start and stop to the millisecond, each
rounded to the nearest, and its midtime as their midpoint; the millisecond is
one constant, `PDS4_EXPOSURE_TIME_DIGITS` in `support/time.py`, which the index
tables' time columns share. A millisecond is
the precision the PDS3 label and index record an image's times to, and a
Cassini exposure is often shorter than a second, so whole seconds would state
a 5 ms exposure as a window of one or two. The nearest, rather than the start
rounded down and the stop up, because the epochs are computed from those
millisecond values -- oops takes the stop from `IMAGE_TIME` and the start as
the stop less the exposure (`oops/hosts/cassini/iss.py` 64-66) -- and the
float lands a few nanoseconds to one side of the millisecond or the other,
often enough that a floor or a ceiling moves it by one. Over the 10,194 rows
of the COISS_2001, 2057 and 2086 index tables, the floor puts 1,124 starts a
millisecond before the PDS3 `START_TIME`, and the ceiling 339 stops a
millisecond after `STOP_TIME`; W1630770594's start floors to `.761` where
PDS3 says `.762`. The nearest reproduces PDS3 on every stop and on every
start but 17, which are PDS3 rows whose `START_TIME` is not `IMAGE_TIME` less
the exposure. The midtime is the midpoint of the start and stop as written
(`pds4_utc_midpoint`), a half millisecond rounding up, and not the midtime
epoch rounded: an exposure an odd number of milliseconds long has its
midtime on a half millisecond, where the epoch's float lands to either side
and PDS3's `IMAGE_MID_TIME` takes the half up. Rounded from the epoch, 303
of the 10,177 rows whose `START_TIME` is `IMAGE_TIME` less the exposure came
out a millisecond before `IMAGE_MID_TIME` (W1629783475's `.826` against
`.827`); taken from the written start and stop, all 10,177 match. The
nearest is also the rule the reference applies to its millisecond start and
stop; section 3.13 says what it does at whole seconds and for the midtime,
and which of its rules this bundle follows for which element.

An image whose navigation never reached a result is skipped; section 3.11
says what happens to it, and the answer is that it never reaches a label. A
document written since #624 records the exposure times in its `observation`
block for every image whose navigation reached a result, pointing or not: the
navigation records a success with no pointing when `compute_pointing` raises
`NavPointingError` or the instrument has no SPICE camera frame mapped, and the
block holds the host's exposure times all the same (section 2.2 row 6). Such an
image is bundled like any other. Until Phase 7 the labels read
`navigation_result.times`, which the navigation writes only beside a solved
pointing, and the labels pass failed such an image with nothing written.

A document an earlier version wrote carries no times in its `observation`
block, which holds only the image's path, name and shape, the instrument, the
camera and the shutter mode; its times are under `navigation_result.times`,
beside a solved pointing. The 17 success documents the Phase 7 review sampled
under `/data/nav-offset-results`, written 2026-08-10 to 08-27, are all of that
vintage. The labels pass fails such an image before anything is written for
it, in one line saying the navigation document records no exposure times in
its observation block and the image has to be navigated again. It checks only
`start_time_et`, since the host publishes the start, the midtime and the end
together. That is a document of a real, earlier vintage rather than a
malformed one, as a backplane root of mixed vintage is to the units check. It
is no fallback to `navigation_result.times`: by the operator's direction every
time the bundle states comes from the observation block, and a navigation of
the current version records them there. The summary pass checks nothing, since
a supplemental file the labels pass wrote always holds the times and a bundle
is written into an empty directory (the ruling of 2026-09-11). The epochs are
read as recorded, with nothing else checked about them (the operator's ruling
of 2026-09-11 that nothing guards against our own files). The empty string is
not reachable.

The data collection label states the cohort's earliest start and latest stop,
at whole seconds as the reference's collection and bundle labels do, the start
rounded down and the stop up. The summary pass already read
every supplemental file to build the global index, so the range is taken
there, in that same read, by an `EpochRangeScan`. The index therefore runs
before the collection files in `main_summary`, and returns the range in a
`GlobalIndexOutcome`, which the driver hands to `generate_collection_files`
and holds for `bundle.lblx`, which Phase 6 renders, without a second
computation. With no range to state -- no data label in the data tree has a
supplemental file beside it -- the data collection is not written, neither
its inventory nor its
label, and counts as a label not written, with an error saying so: one rule
with the empty collection's (section 3.5). The supplemental files
are read as the labels pass wrote them, with nothing checked in them but the
statistics (the same ruling); a data label that failed to render is reported
by the labels pass, and by the summary pass as an image whose products
disagree (section 3.5), and the bundle is cleared and regenerated. Running
first, the index generator refuses a bundle with no data
directory itself, as the collection generator does, rather than write its
tables into a root the labels pass would then refuse.

### 3.5 Inventories

`.csv`, no header row, `\n` line terminator (`csv.writer(f,
lineterminator='\n')`, file opened with `newline=''`), one `P,<lidvid>` line
per member. `<records>` counts members. The `FILE_RECORDS` template function
counts lines, so with no header it counts the members.

A collection with no member has neither an inventory nor a label.
`PDS4_PDS_1O00.xsd` requires `<records>` in `File_Area_Inventory/Inventory`
to be at least 1, so no label can describe an empty inventory; with the
header row counted, as before Phase 5, an empty collection's label stated one
record, valid only by accident. It is one rule with the data collection's
range (section 3.4): a generated collection is written only when its label
can state everything the PDS4 standard requires of it -- at least one member, and for the
data collection the range of its members' epochs. One that cannot is not
written at all, neither its inventory nor its label, whatever an earlier run
left at either path is removed, and it counts once among the labels not
written, with an error naming the collection and every reason, so the summary
pass exits 1. Each generated collection takes its members from the labels of
its own kind on disk -- the data collection from the data labels, the browse
collection from the browse labels, the miscellaneous collection from the index
labels -- and the summary pass holds each image's
products against each other, since every data product has a browse product
(#602, Phase 5): an image with a data label and no browse label, or a browse
label or supplemental file and no data label, disagrees, is logged by name
and counted, and the pass exits 1. The check and the empty-collection rule
are one rule at two scales -- each image holds all its products, and each
collection at least one member -- and the collection rule refuses an empty
collection even over a bundle with no image, where the check has nothing to
count. Over
a bundle with no data label and no browse label the summary pass writes
neither collection and counts two labels; over supplemental files and no label -- what a labels pass leaves
when every label fails to render, since it writes the supplemental file
first -- it writes neither, counts two labels, and counts every image as one
whose products disagree. A collection label that fails
to render is Phase 1's case, not this one: its inventory is written and
stays, as the index tables do.

A summary that exits 1 does not repair the bundle: the generators report
what they cannot describe, and a bundle is written into an empty directory
(the operator's ruling on #605, recorded in Phase 1's text). What they wrote
stays, but for two things: a collection that cannot be written has whatever
an earlier run left at its paths removed, and the bundle label, rendered and
then found to name a collection the bundle does not hold, is removed in the
same run. After one, the browse
labels can name a data collection that was not written, and the data labels
a browse collection that was not written; a user guide whose label failed is
in `document/user_guide/` unlabeled and unlisted, and a metakernel whose label
failed is in `spice_kernels/` with neither label nor collection. The bundle
label names no collection the bundle does not hold, except after a run refused
before it cleared anything -- over a bundle with no data directory -- which
leaves an earlier run's products, its bundle label among them, as they were;
and a run refused after clearing can leave empty collection directories. The
directory is cleared and the bundle regenerated into it, as after a labels
pass that exits 1.

Three inventories are **generated**, because their membership depends on
what the run produced: `collection_data.csv`, `collection_browse.csv`, and
`collection_miscellaneous.csv`, whose primary members are the global-index
products of section 3.1 whose labels are on disk, and whose secondary members
are the document inventory's (below). The miscellaneous collection is written
only when it has a primary member: its secondary members are cited beside a
product of its own, and do not make it a collection by themselves. With
neither index table written with its label -- no image gives either table a
row, or neither label renders -- neither its inventory nor its label is
written, whatever an earlier run left at either path is removed, it counts
once among the labels not written, and the bundle label, which declares it, is
removed by its member check, so the summary pass exits 1.

Three are **rendered** from the template directory (section 3.2), because
their membership is fixed: document, spice_kernels and schema, the last from the
configured schemas (section 3.9). The fourth,
context, is **written** from the members the template directory ships and every
target the data labels name (Phase 8). An inventory lists its `P` line, a
product of this bundle, only when that product's label is in the bundle: the document inventory leaves out the user
guide when the template directory holds no guide or the guide's label failed
(section 3.6), and the SPICE kernel collection, whose one member is the
metakernel, is by the rule above not written when the metakernel's label is
not. Their `collection_*.csv` templates follow the same rules as the
generated ones -- no header, LF, a line feed after the last line -- and hold
no comment line, since the label would count one as a record.

Members carry an explicit version: the reference writes
`P,urn:...:miscellaneous:global_mosaic_index::1.0` and
`S,urn:nasa:pds:context:instrument:issna.co::1.2`, naming the actual
published version of each secondary product rather than leaving it open. Our
`collection_context.csv` had no `::` on any line; Phase 6 gave each line its
context product's version as the PDS registry held it on 2026-09-14.
`GET https://pds.nasa.gov/api/search/1/products/<lid>/latest` answered
`mission.cassini-huygens::1.5`, `spacecraft.co::1.4`, `issna.co::1.2` and
`isswa.co::1.2`, each the newest its `/all` lists (1.0-1.5, 1.0-1.4, 1.0-1.2
and 1.0-1.2), and the versions the reference's document inventory cites. The
one exception the reference itself makes is its own `collection_context.csv`,
which is LID-only; both forms appear to pass, so prefer the versioned one
and let validation say otherwise.

Our `collection_context.csv` lists the mission, the spacecraft and the two
cameras, from the template directory, and then every target the data labels
name, one `S` line each at the version the targets table gives it (section
3.7), as the reference's lists every target its labels name. The summary pass
writes it from those four lines and the targets its one read of the
supplemental files takes, so its label's record count is its lines; it is
cleared with the other run-level products before that read; and a test holds
every target LID a data label references to a line of it.

A collection inventory also carries **`S` members**, not only `P`. The
reference's document and miscellaneous inventories both list the context
products and the external ISS data user guide as secondaries alongside their
own primaries, so an inventory is a statement about everything the
collection references, not just what it owns. Ours do the same: the document
inventory the template directory ships lists the four context products and the
ISS data user guide at `::2.0` as `S` members, and the miscellaneous inventory
takes its `S` members from that one file (`bundle_products.secondary_members`),
so the two cannot disagree about them. Neither lists the target context
products, which the reference's both do: their `S` members are the context
products their own collections' labels reference, and no document or
miscellaneous label names a target. The data and SPICE kernel inventories,
whose labels do name the targets, list their own products alone, as the
reference's do (Phase 8, section 3.13).

### 3.6 The document collection

`collection_document.csv` lists the backplanes user guide, the PDS3 ISS
Data User's Guide, and the context products of the mission, the spacecraft
and the two cameras at their versions (section 3.5), as the reference's
does. The first is a product this bundle owns and must therefore contain;
the others are external references and stay `S`. It cites the ISS guide as
`urn:nasa:pds:cassini_iss_saturn:document:iss-data-user-guide::2.0`, the
version the reference bundle cites, by the operator's decision of
2026-09-14. The PDS registry returned only version 1.1 of that product on
2026-09-14, and `::2.0` returned 404 when Phase 6 asked the same day.

The user-guide PDF is an operator deliverable, tracked as #595 (a shared
LaTeX template for all four instruments' guides) and #596 (the Cassini guide
written from it; #597, #598 and #599 are the other three, which wait on
their instrument's half of #53). The code path is written so that the PDF's
presence in the template directory decides whether the guide and its label
are written, and the label, written or not, whether the document inventory
lists the guide:

- PDF present: it is copied into `document/user_guide/`,
  `cassini-iss-saturn-backplanes-user-guide.lblx` is rendered beside it, and
  `collection_document.csv` lists it `P`.
- PDF absent: none of those, the inventory has no `P` line, and the run logs
  one warning naming the missing file.
- PDF present and its label not written: the PDF is copied, the label counts
  as not written, and the inventory has no `P` line, since an inventory lists a
  product of the bundle only when its label is there (section 3.5).

The document collection exists either way, and `bundle.lblx` declares it
either way: it holds the bundle's documents and lists the external ones the
bundle cites. The references to the user guide in the data, browse, data
collection, metakernel and bundle labels stay in every case, since they carry
the LID the delivered bundle will contain; with the PDF absent they do not
resolve, and the readme's sentences about the guide are untrue of the
draft. A draft is acceptable so, as acceptance criterion 9
has it, given the one warning and a delivery note recording the absence; a
bundle delivered to the Node is not, since criterion 6 requires every
reference to resolve.

The guide delivered has to be a PDF that `validate`'s VeraPDF library can
read, and presumably a PDF/A: `validate` reads the user guide's content with
VeraPDF, and over the cohort bundle built with the tests' stand-in guide, an
18-byte file that is not a PDF, it reported an internal error for that reason
(Part A of Phase 10). The stand-in is for label tests only.

The LID has its `:document:` segment in every label and inventory that names
it (Phase 6).

### 3.7 Targets and the mission area

**The targets table** is `backplanes.target_lids` in `config_900_backplanes.yaml`,
which had reserved it empty.  Part A of Phase 8 filled it by hand for every body the
backplane stage can produce backplanes for in a Saturn image -- Saturn and the nineteen
satellites `config_100_satellites.yaml` lists for it, the list
`backplanes_bodies.backplane_body_names` gives -- and for the ring target Saturn's ring
backplanes are computed for, `SATURN_MAIN_RINGS` (`backplanes_rings.ring_target`; #618
would move that choice into configuration).  That is more than the bodies the cohort
holds, so a real run meets no body without an entry.  Each entry is keyed by the name
the backplane metadata gives the target and gives its context product's LID and
version, and the name and type the product gives the target.  The values are the PDS
registry's answers of 2026-09-14 to

```bash
curl -sL -H 'Accept: application/json' \
  https://pds.nasa.gov/api/search/1/products/urn:nasa:pds:context:target:<id>/latest
```

reading `lidvid`, `pds:Target.pds:name` and `pds:Target.pds:type` from its `properties`.
The main rings' product was found by searching for every Saturn ring target,

```bash
curl -sL -G -H 'Accept: application/json' https://pds.nasa.gov/api/search/1/products \
  --data-urlencode 'q=(lid like "urn:nasa:pds:context:target:ring.saturn*")'
```

which returned two, `ring.saturn.rings` ("Saturn Rings") and `ring.saturn.f_ring`
("F Ring of Saturn"); `ring.saturn.main_rings` does not exist.

| Key | Context product, `urn:nasa:pds:context:target:` ... | Version | Name | Type |
|---|---|---|---|---|
| `SATURN` | `planet.saturn` | 1.4 | Saturn | Planet |
| `ATLAS` | `satellite.saturn.atlas` | 1.2 | Atlas | Satellite |
| `CALYPSO` | `satellite.saturn.calypso` | 1.2 | Calypso | Satellite |
| `DAPHNIS` | `satellite.saturn.daphnis` | 1.2 | Daphnis | Satellite |
| `DIONE` | `satellite.saturn.dione` | 1.2 | Dione | Satellite |
| `ENCELADUS` | `satellite.saturn.enceladus` | 1.2 | Enceladus | Satellite |
| `EPIMETHEUS` | `satellite.saturn.epimetheus` | 1.2 | Epimetheus | Satellite |
| `HELENE` | `satellite.saturn.helene` | 1.3 | Helene | Satellite |
| `HYPERION` | `satellite.saturn.hyperion` | 1.2 | Hyperion | Satellite |
| `IAPETUS` | `satellite.saturn.iapetus` | 1.2 | Iapetus | Satellite |
| `JANUS` | `satellite.saturn.janus` | 1.2 | Janus | Satellite |
| `MIMAS` | `satellite.saturn.mimas` | 1.2 | Mimas | Satellite |
| `PAN` | `satellite.saturn.pan` | 1.2 | Pan | Satellite |
| `PANDORA` | `satellite.saturn.pandora` | 1.2 | Pandora | Satellite |
| `PHOEBE` | `satellite.saturn.phoebe` | 1.2 | Phoebe | Satellite |
| `PROMETHEUS` | `satellite.saturn.prometheus` | 1.2 | Prometheus | Satellite |
| `RHEA` | `satellite.saturn.rhea` | 1.2 | Rhea | Satellite |
| `TELESTO` | `satellite.saturn.telesto` | 1.2 | Telesto | Satellite |
| `TETHYS` | `satellite.saturn.tethys` | 1.2 | Tethys | Satellite |
| `TITAN` | `satellite.saturn.titan` | 1.1 | Titan | Satellite |
| `SATURN_MAIN_RINGS` | `ring.saturn.rings` | 1.1 | Saturn Rings | Ring |

The guard is a test over the shipped configuration, holding the table to the stage's
own body list and ring target for Saturn; nothing checks it when a run starts, and a
name with no entry raises, naming it, which fails that image.  Scraping the context
products to maintain the table stays #79.

**Which labels name which targets.**  A data label names one `Target_Identification` for
each body its image's backplane metadata names that has geometry -- a statistic at
least, as `targets.has_geometry` decides; a body the inventory found in the field of
view that shows at no pixel is named with no statistic, and is no target (Phase 8's last
round; Part A named it, and W1479724035 names nine bodies, seven with no pixel) -- and
one for the rings when the metadata holds a ring statistic, each with
the name and type its context product gives and an `Internal_Reference` to its LID of
type `data_to_target`, in the table's order.  An image whose metadata names no body with
geometry and holds no ring statistic has no target, which `PDS4_PDS_1O00.xsd` requires of a data
label, and its backplanes hold no geometry for a label to describe, so it is skipped,
with one log line, before anything is written for it and before the checks that fail an
image.  Under the provisional rule that an absent input is a skip, which #600 will
decide, a skip is not a failure, and the run's exit status is unaffected (the fix
round; Part A failed it, so no real run could exit 0).  Such images are many.  By the RMS
Node's summary tables, 71,246 of the 405,121 images of COISS_2001-2116, 17.6%, show no
Saturn, none of the nineteen satellites and no main-ring pixel: sky frames (16,043),
Saturn-targeted frames that miss it (12,520, 487 NAC F-ring frames among them), irregular
moons (Kiviuq alone 4,740), and Pallene and Methone; 2,086 of the reference's 20,584
reprojected images are among them.  The repository's own star-dominated library frames
are such images, and so are 24 of the 424 successes of the operator's 2026-09-11 run.  An
F-ring frame's ring pixels lie outside `SATURN_MAIN_RINGS` -- N1467350440's are at
139,630 to 140,612 km -- so the F-ring frames come in only if #618 changes the ring
target.  `targets.covers_a_target` decides the skip without reading a ring target, so
backplanes an earlier version generated, which record none, still reach the check that
fails them.  The bundle label (`bundle_to_target`), the data collection label and the
SPICE kernel collection label (`collection_to_target`) and the metakernel label
(`data_to_target`, in its `Context_Area`) name every target the data collection's
members name, which the summary pass takes in its one read of the supplemental files,
over the data inventory's members, as it takes the range of their epochs, and which
`GlobalIndexOutcome` returns beside the range.  A summary pass over data members an
earlier labels pass wrote, before the ring target was recorded, raises a bare
`KeyError: 'target'` from that read; nothing checks for it, since a bundle is written
into an empty directory and regenerated (the 2026-09-11 ruling), as Phase 7 decided for
the times.  The context inventory lists each target (section 3.5).  Stars are not
targets (section 3.13).

**The ring geometry and the incidence angle** (#75, #47).  The rings dictionary
describes an image's ring geometry in one class, `rings:Reprojection_Geometry` within
`rings:Ring_Reprojection`, which the reference's data labels fill for their reprojected
images.  Of the dictionary's other classes, `rings:Ring_Spectrum` holds every one of an
image's ranges of radius, longitude, angles and resolutions but the longitudinal
resolution, but it describes ring spectra and spectrograms, and the dictionary's
Schematron requires it to identify the observation's wavelengths; section 3.13 records
the question this leaves open.  A data label of an image with ring backplanes fills
it from the ring statistics, each written as the global index tables write it and
stated in the unit its attribute takes: `minimum_` and `maximum_phase_angle`,
`_emission_angle`, `_inertial_ring_longitude` (deg), its range wrapped at the prime
meridian (below), and `_ring_radius` (km), and, in its
`Reprojection_Grid_Parameters`, `_radial_resolution` (km) and
`_longitudinal_resolution` (deg), each a size per pixel stated in the length or the angle
a pixel spans.  The incidence angle is one angle over an image, so no backplane holds
it: the backplane stage now records it in the backplane metadata's `rings` block as
`incidence_angle`, a value in degrees with its unit, beside `target`, the ring target
the ring backplanes were computed for, which the targets table keys the rings by.  It is
`oops`'s `ring_center_incidence_angle` on the observation's full-frame backplane and the
ring target the ring backplanes use: the incidence at the ring system's center, for the
light that reaches the camera at the observation's midtime, measured from the normal on
the sunlit side.  On the real ring frame N1863267861 it is 63.334 deg, 0.00006 deg from
90 less the Sun's elevation above Saturn's equator as SPICE gives it, and within
0.003 deg of every one of the frame's 1048576 ring pixels' `ring_incidence_angle`.  The
stage also keeps `ring_incidence_angle` at each pixel, on the same target and measured
the same way, which no plane holds either, so #47 stands: the writer records its least,
greatest and mean over the ring pixels the merged planes hold, as `min`, `max` and
`mean` beside the center's `value`, and the label states those three as the mean, the
minimum and the maximum incidence angle, where the reference states one value as all
three (the fix round).  On N1671602206 the center's angle is 82.57158 deg and its ring
pixels' 82.57085 to 82.57104, mean 82.57096, the rings about the center being hidden;
on W1626850595, 89.67993 at the center and 89.67993 to 89.67996 over the rings.  The class's required elements say what the product is:
`reprojection_plane` `Equator`, `corotating_flag` `N`, and `epoch_reprojection_basis_utc`
the image's midtime, which with no co-rotation the longitudes do not depend on; its
description says the arrays are the image's own lines and samples, not a reprojection,
and how the longitude range is wrapped.  An image of backplanes an earlier version
generated -- ring statistics without the angle's range over the ring pixels, whether
with no angle at all or with the center's alone -- is failed before anything is written
for it, until they are regenerated: those backplanes took their statistics before the
merge, too (below).  The cohort's backplane fixtures carry the angle SPICE gives at each
cohort epoch, as the stage writes it -- 64.59619, 64.68149 and 64.59625 deg -- and at
each ring pixel an angle ramping a thousandth of a degree either side of it.

**The wrapped longitude range** (the fix round).  The rings dictionary defines
`minimum_` and `maximum_inertial_ring_longitude` as a range wrapped at the prime
meridian, the minimum above the maximum where it crosses zero, so the label states the
ring longitude statistic's `wrapped_min` and `wrapped_max`, which the backplane stage
records beside its plain `min` and `max`, in degrees like them, from the merged plane.
Taken on the circle, the widest gap between the longitudes, the gap across zero among
them, is the part the image does not cover, and the arc runs from the longitude after it
to the one before it (`statistics.wrapped_range`).  When the widest gap is the one
across zero the arc is the plain least and greatest; of two gaps equally wide the one
across zero is taken, so the plain range stands; a longitude rounded to 360 is the one
at zero; and longitudes leaving no gap wider than the image's coarsest longitudinal
resolution, the greatest value of `ring_longitudinal_resolution`, cover the circle,
stated as 0 to 360.  N1591060671's rings, 359.687 across zero to 0.402 deg, are stated
so, where the plain range stated the whole circle.  The index tables keep the plain
least and greatest under their own names (section 3.13).  The cohort's ring image
records the range the stage writes, 216.0 across zero to 204.706, since Saturn's disc
interrupts its synthetic ramp of longitudes; its plain range is 0 to 360.

**Statistics over the product's own pixels** (the fix round).  The body and ring stages
took each plane's statistics before the merge, which then masks a ring pixel a nearer
body covers and gives a pixel two bodies share to the nearer, so the metadata, the data
label and Phase 7's index tables stated ranges over pixels the product's own arrays have
no value at: N1671602206 was labeled with a ring longitude of 10.767 to 258.852 where
its FITS spans 241.932 to 258.852, and a greatest radial resolution of 3818.0 km against
1151.4.  The writer now takes every statistic from the planes the FITS holds, over the
pixels where each has a value: a body's over the pixels the body identity map gives it,
the rings' over every pixel; a body a nearer body hides entirely is recorded with no
statistic.  The wrapped range and the incidence range are taken over the same merged
planes.  The index tables' values change with it for any image where a body covers the
rings or another body; the cohort's do not, since none of its bodies covers either.
Regenerated with this code, every ring value the data labels of N1671602206 and
W1626850595 state equals its FITS array's, thirty of thirty.

**The mission area** is Part B of Phase 8.  `cassini:ISS_Specific_Attributes` is to be
filled from the Cassini facts the navigation document's `observation` block records,
which #684 adds to what `ObsCassiniISS.get_public_metadata()` publishes, on a branch
against `main`, by the operator's direction; the bundle's supplemental file already
carries the whole document.  It reaches this stack after #684 merges and `main` is merged
into `rf_pds4_draft_bundle`.  The `cassini:*` variables `pds4_template_variables` reads
from the image's PDS3 index row are not the source, since a run's row holds two columns
(Phase 8's text).

### 3.8 Units: radians in the arrays, degrees in the tables

The FITS arrays are radians and say so in `BUNIT`. The statistics -- and
therefore the global index tables -- are degrees, and each statistic records
its unit beside its minimum and maximum. Both per-source stages reduce their
planes through `spindoctor/cli/backplanes/statistics.py`, whose
`statistics_units` holds the rule: if the part of a unit before any `/` is
exactly `rad`, it becomes `deg` and the rest is kept, so `rad/pixel` becomes
`deg/pixel`; every other unit is left alone. The index tables are read by
people, and the operator's ruling of 2026-09-09 is that everything in them is
degrees.

The tables are written with a format per unit (#607), from
`INDEX_VALUE_FORMATS` in `global_index.py`: three decimals for `deg`, one for
`km`, eight for `deg/pixel`, and five significant figures for `km/pixel`,
written positionally, never in exponent form. The arrays are float32, so a
statistic carries about seven significant digits; each format is chosen within
that from what one pixel resolves, the eight decimals of `deg/pixel` reaching
its edge. No format fixes a column's width, so each field is sized from the widest
value its column holds, the masked value included where the column holds it
(Phase 7). Nothing checks the configured units when a bundle is
written (the operator's ruling of 2026-09-11): two tests over the shipped
configuration are the guard, one holding each measure to the ones
`statistics_units` converts or passes through and the other each unit to
`INDEX_VALUE_FORMATS`, so an angular unit other than `rad` (`mrad`, `arcsec`)
fails them until `statistics_units` handles it.

Both passes hold every statistic to what an index column can hold, through
`spindoctor/cli/pds4/statistic_checks.py`, since every column is in one unit
and holds only finite numbers: the unit a statistic records has to be its
plane's configured unit restated through `statistics_units`, and neither its
minimum nor its maximum may be NaN or infinite. A statistic in another unit, or
with a NaN or infinite minimum or maximum, fails its image in the labels pass,
before anything is written for it, and fails the run in the summary pass,
before either index table is written. The remedy is to regenerate the
backplanes, and for the summary pass then the bundle into an empty directory.
A plane the document holds that the configuration does not declare is not
checked. Both are the operator's rulings of 2026-09-10. The documents are
written by this package's own software, so nothing else about them is checked:
no value a writer of ours cannot produce is guarded against (the operator's
ruling of 2026-09-11). The summary pass renders every cell of both tables
before it opens either, so no failure of any kind leaves a table half-written.

`sd_backplane_viewer` has its own rule: it converts a plane whose `BUNIT` is
`rad` in any letter case, or whose name contains an angle's name, so it
displays the one plane declared `rad/pixel` in radians per pixel. Nothing this
plan generates goes through it, so it is #611 rather than a phase: it needs to
call `statistics_units` instead of testing for a literal, and a ruling on the
name heuristic.

What follows for the labels, and what a later reader must not "fix":

- The `Array_2D_Image` blocks Phase 4 generates state `unit` from the HDU's
  `BUNIT`, so an angular plane is labeled `rad`. The label describes the
  array, and the array is radians.
- The `Field_Character` blocks Phase 7 generates for the global index take
  their `unit` from the same config entry the column was built from, mapped
  through `statistics_units`, the function that produced the column's values,
  so a label and the column it describes cannot disagree: an angular column is
  labeled `deg`, a resolution in radians per pixel `deg/pixel`. Its missing
  constant is the masked value in the column's own format (section 3.13).
- So one bundle carries `unit="rad"` on an array and `unit="deg"` on the
  table summarizing it, deliberately: each label is correct about the file it
  describes.

The backplanes user and developer guides and the docstring of
`statistics.py` state the rule; section 3.6's operator deliverable, the
user-guide PDF, repeats it for the bundle's own readers.

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

Phase 6 asked the registry about both spellings of all five
`collection_xml_schema.csv` lines on 2026-09-14, as
`GET https://pds.nasa.gov/api/search/1/products/<lidvid>`, and neither
resolves for any of them. Ours --
`urn:nasa:pds:system_bundle:xml_schema:pds-xml_schema::1.24`,
`...:disp-xml_schema::1.15`, `...:geom-xml_schema::1.19`,
`...:rings-xml_schema::1.15` and `...:cassini-xml_schema::1.18` -- return 404,
and so do the reference's, `...:pds-xml_schema_1.24.0.0::1.0`,
`...:disp-xml_schema_1.24.0.0_1.5.1.0::1.0`,
`...:geom-xml_schema_1.24.0.0_1.9.11.0::1.0`,
`...:rings-xml_schema_1.24.0.0_1.15.0::1.0` and
`...:cassini-xml_schema_1.24.0.0_1.8.0.0::1.0`, whose LIDs return 404 for
`/latest` and `/all` as well. Our five LIDs resolve, but at none of the
versions our lines name: `/all` lists `pds-xml_schema` 1.0-1.6, 1.9-1.13 and
1.15-1.21, `disp-xml_schema` 1.0, 1.9, 1.10, 1.12, 1.13 and 1.16,
`geom-xml_schema` 1.0, 1.3 and 1.16, `rings-xml_schema` 1.0, 1.5, 1.8 and 1.9,
and `cassini-xml_schema` 1.0, 1.14 and 1.17. The lines are unchanged, since
the reference's spelling resolves no better, and the question stays the
Engineering Node's.

**Declared once, since Phase 9.** The information model version and the schema of each
dictionary are set in the dataset's entry in `config_950_pds4.yaml`:
`information_model_version`, and under `schemas` each dictionary's location less the
extension and its `xml_schema` LIDVID, keyed by the prefix its namespace takes in a
label. Every template takes them as variables (section 3.13), and the XML schema
inventory is a template listing the LIDVIDs, their spelling unchanged. The bump above
would now be one edit, of the `rings` entry; a move to another information model build,
one edit of the entry. The information model version is there because the
`PDS4_PDS_1O00` Schematron requires `1.24.0.0` of every label, so it moves with the
`pds` schema. Tests over the shipped configuration hold it to the shipped templates, a
schema given for exactly the dictionaries they declare, and to itself: the `pds` schema's
file name carries the information model version's code (`1O00` for `1.24.0.0`) and its
LIDVID the version's first two parts (`::1.24`), and every other dictionary's file name
carries the same build's code.

### 3.10 Bundle name and version

`bundle_name` and `bundle_version` sit side by side in the dataset's entry in
`config_950_pds4.yaml`, read through `pds4_bundle_name()` and `pds4_bundle_version()`;
neither has a default. This was #71, which Phase 9 closes; it landed late because
doing it early would have meant re-editing every template the earlier phases touched.

**Where the entry lives, decided in Phase 9's fix round.** The dataset's entry stays in
`config_950_pds4.yaml`, a registry of PDS4 settings keyed by dataset, and is not moved into
the instrument's `config_400_inst_coiss.yaml`, although the loader reads the files in order
and deep-merges each section, so the configuration would be the same either way. Every
`config_4*` file's bytes are hashed into each navigation document's `static_data_hashes`
(`nav_orchestrator/provenance.py`), which records them as the instrument's static data: a
new bundle version or a moved schema kept there would read as a change of Cassini
instrument data. The resolved configuration's hash covers the `pds4` section wherever it
is kept.

Following #71's single version number, `bundle_version` is the `version_id` of the
bundle, of every collection and of every product the bundle writes, and the version in
every LIDVID citing one of them: in labels, in inventories, and in the metakernel's
line of the SPICE kernel inventory. External references keep their own versions: the
context products, the ISS data user guide at `::2.0`, the calibrated source products
and the schemas. No template spells the name or the version: each writes `$BUNDLE_LID$`,
which it extends into each collection's and product's LID, and `$BUNDLE_VERSION$`. Only
the bundle's own name moves; the source bundle `cassini_iss_saturn`, the template
directory `cassini_iss_saturn_1.0` and the user guide's file name keep theirs (the Phase 9
record gives every occurrence).

On a later revision each product would carry its own version: a product unchanged between
two deliveries keeps its version while its collection's and the bundle's move, and its
`Modification_History` lists each. This draft gives every product the bundle's version,
and one `Modification_Detail` at it, "Initial version".

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
incomplete. Choosing a cohort for the run over a real volume (#708) is
exactly this question. Part A of Phase 10 added
it: `sd_create_bundle labels --check-only` prints one line per selected image
and then a count, writes nothing, neither needs nor creates a bundle root,
and takes the four paths from `image_inputs`, the one function the labels
pass takes them from.

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

That package is the nav half of every cohort, and it holds the cohorts in sets
of their own rather than by extending the set it already held. The two sets are
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

So the package holds a **cohort per bundle**, each a `Cohort` subclass in a
module named for the bundle and registered in `COHORTS`, built from the same
`shared.py` primitives and written to a cohort root rather than into
`RESULTS_TREE`. The Cassini ISS Saturn cohort, `CohortCassiniISSSaturn` in
`cohort_cassini.py`, is the one that exists. `results_tree_documents()` and the
stats fixture tree are untouched by it.

**No cohort is checked in.** Each is built once a session into a directory
`tmp_path_factory` makes, and torn down with it; no cohort bytes live under
`tests/`, and the `python -m tests.mini_nav_results cohort <bundle> <outdir>`
form writes wherever the operator points it. The builders are the artifact, not
their output. That is what keeps a second bundle's cohort from costing the
repository anything -- adding one is one module, a `Cohort` subclass supplying
the bundle's images, holdings layout, camera reading, plane bounds and
registered dataset, and one entry in `COHORTS` -- and the FITS, the PNGs and
the tables it implies exist only while a test is running.

The two sets differ on this deliberately, and it is worth saying why rather
than leaving it to look like an inconsistency. The stats tree stays stored
because `test_results_tree_documents.py` holds the stored tree against what
the builders emit today, and that comparison is what reports a change in the
production writer instead of absorbing it. Generate that tree at test time
and the test compares the builders to themselves and can never fail. The
cohort has no such frozen counterpart -- what it feeds is a schema
validator, which is an external judge -- so storing it would buy nothing and
cost the repository a growing pile of binary fixtures.

Build each once per session rather than once per test: the session-scoped
`mini_nav_cohorts` fixture writes a cohort class into a temporary directory the
first time a test asks for it, since a dozen label tests should not each
rewrite a FITS. The self-tests every cohort is held to run over each registered
cohort; what only one bundle's cohort can state is tested in a module named for
the bundle.

What the Cassini ISS Saturn cohort holds beyond the documents:

- **A summary PNG per successful image.** A real PNG, small; the browse
  label states its byte size and checksum.
- **A backplane FITS per successful image.** A *real* FITS written by
  `astropy.io.fits`, 16x16 per plane, because Phase 4 reads `hdrLoc` and
  `datLoc` out of it through `fileinfo()` and the label states its size and
  MD5. A few bytes standing in for a FITS cannot serve that.
- **Backplane metadata beside each FITS**, whose `bodies` and `rings`
  statistics name the same backplanes the FITS carries, since the global
  index columns come from one and the arrays from the other.
- **`ImageFile`s carrying `index_file_row`**, because sixty-six of the
  seventy `cassini:*` template variables are read from the PDS3 index row, a
  source Part B of Phase 8 replaces with the navigation document's
  `observation` block (#684). No index file is parsed; the row is a dict,
  keyed by the index file's own column names.
- **Coverage the bundle stage cares about**, which the stats corpus has no
  reason to carry: two images whose numbers shard into *different*
  `1234xxxxxx/123456xxxx` directories, one image with ring backplanes and one
  without, and one image whose navigation did not succeed.

Three things about the Cassini cohort's products are known not to hold, and are
recorded here rather than only in a docstring, because each is a
property of the product a later phase describes rather than of the code
that writes it.

**A 16 by 16 frame records `INSTRUMENT_MODE_ID = FULL`.** `FULL` is a
claim about size and no Cassini mode value names a frame this small, so
there is no truthful value to record instead and the simplification
stands. It stops being purely internal at Part B of Phase 8, which puts
`cassini:instrument_mode_id` into the same label as Phase 4's 16-element
`Array_2D_Image` blocks: re-check it there, and decide whether the
cohort grows a full-size frame for one image or the label carries the
mode the row holds.

**One body per image, where a real frame often has several.** The merge
resolves overlapping bodies by nearest distance and aggregates a plane's
statistics over all of them, and a cohort with one body per frame cannot
tell a correct aggregation from one that reports the first body it
found. A second body is what Phase 7 needed, since that is
where a per-body inventory and the global index columns over it are
written; the backplane fixture takes a tuple of bodies already, so
adding one is a line in the image's declaration. Part A of Phase 8 held a
data label to two bodies by giving the limb image's backplane metadata a
second body in the test itself, under a backplane root of its own, and left
the cohort as it is, so the aggregation over overlapping bodies still has
no cohort image to be tested on.

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
a constructor in `host_cassini.py`, the Cassini host's module, that takes an
epoch and returns the clock triple, so a document built through it cannot
carry an invented one, and a second beside it derives the image number from
the same epoch. The Cassini cohort is built entirely through both; another
host's cohort brings its own. Routing the existing four through it as well
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

What holds it there is an `integration`-marked test,
`tests/integration/test_cohort_cassini_clock_against_kernel.py`, that furnishes
the kernel and converts every cohort epoch again. It has to be that test and
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
and Phase 4 states every array's byte offset against that order.

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

The cohort builders live in the same package rather than beside it, so there
is one name and one entry point. That entry point takes the set to write, for a
cohort the bundle whose cohort it is, and where to write it, in that order --
an argument that changes *which* files are written depending on whether a later
argument is present is exactly the surprise a fixture tool should not hold:

```bash
PYTHONPATH=src python -m tests.mini_nav_results results_tree \
    tests/spindoctor/cli/stats/data/results_tree
PYTHONPATH=src python -m tests.mini_nav_results cohort cassini_iss_saturn <outdir>
```

Every argument is required, a cohort's bundle being one of the names in
`COHORTS`. The stats path is spelled out
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
and an image stub per row -- belongs to the run over a real volume (#708),
which is where the CLI workflow over a whole volume is actually needed, and
is tracked as an issue of its own. Part A did not need it: its gate builds
the cohort's bundle through the library entry points and runs the check
over it.

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
and the templates substitute them. Our templates hardcoded each URL in each
file, which is why the section 3.9 dictionary bump had to be a search and
replace across two files rather than a one-line edit. Adopted in Phase 9, under the
reference's names -- `PDS4_<PREFIX>_SCHEMA` for a Schematron, `PDS4_<PREFIX>_SCHEMA_XSD`
for an XML schema, and `INFORMATION_MODEL_VERSION` -- with the values in the dataset's
configuration entry beside the bundle's name and version rather than in a module: each
bundle's templates are written for its own dictionaries (the reference's declare rings
`1E00` where ours declare `1F00`), and the Cassini dictionary is then in the Cassini
entry (section 3.9).

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

Phase 4 made the declaration: every float `Array_2D_Image` carries a
`Special_Constants` block whose `missing_constant` is the configured value,
read from the configuration, and `BODY_ID_MAP`'s array carries none and says
what its `0` means instead (section 3.3).

`collections.py`'s "TODO Need an appropriate sentinel value for missing
data" was the table-cell half of the same question, and Phase 7 answered it
with the same value, which the operator accepted on #601: a cell whose
image has no statistic for its plane holds `backplanes.masked_value` written in
the column's own format -- `-999.000` in a `deg` column, `-999.0` in `km`,
`-999.00000000` in `deg/pixel`, `-999.00` in `km/pixel` -- and every statistic
`Field_Character` declares that text as its `missing_constant`. The evidence,
gathered 2026-09-14:

- **The schema allows it.** `PDS4_PDS_1O00.xsd` (the cached copy, the
  `Field_Character` type at line 1285) gives `Field_Character` an optional
  `Special_Constants` (`minOccurs="0"`) after its `description`, so a table
  field can declare a missing constant.
- **`validate` accepts a blank, and a value of the field's type without
  comparing it with the constant.** In the NASA PDS `validate` tool
  (https://github.com/NASA-PDS/validate,
  at `fe7e30ba`), `FieldValueValidator.validate`
  (`src/main/java/gov/nasa/pds/tools/validate/content/table/FieldValueValidator.java`)
  reports a field of a fixed-width record whose trimmed value is empty at debug
  level only, "Field is blank." (lines 352-357), so a blank `ASCII_Real` would
  pass. Any other value is checked against its data type first (line 361);
  `-999.000` matches `asciiReal` (lines 84-85), so it passes as a real and is not
  compared with the constant at all. A value that fails its type is still
  accepted when it equals a declared constant as text:
  `SpecialConstantChecker.isNonConformantSpecialConstant`
  (`src/main/java/gov/nasa/pds/tools/validate/SpecialConstantChecker.java`, lines
  31-46) compares `value.equals(constants.getMissingConstant())` (called at
  `FieldValueValidator.java` line 367). Where a field declares a minimum or a
  maximum, `checkSpecialMinMax` (`FieldValueValidator.java` lines 457-594)
  compares it as a number through `SpecialConstantChecker.sameContent` (lines
  165-213): the text first, then `BigDecimal.compareTo` when the constant has a
  decimal point. The record is read by field location and length: `pds4-jparser`
  (https://github.com/NASA-PDS/pds4-jparser, at `a5d61745`) builds a
  `FixedTableRecord` for a `Table_Character` (`objectAccess/TableReader.java`
  line 283), each field's description carrying its `Special_Constants`
  (`objectAccess/table/TableCharacterAdapter.java` line 95).
- **The reference's blank column is a string.** The reference declares no
  `Special_Constants` anywhere. Its `global_mosaic_index.tab` column 54, blank in
  151 of its 305 rows, is `notes`, an `ASCII_String` of 4 bytes, so it shows
  `validate` accepting a blank string field and says nothing of a blank real.

So a declared constant can work and a blank is not required. `validate`
accepts a cell written as the constant as a real, as it would any number, and
never compares it with the constant; the declaration is what tells a reader
that the value means the plane has no statistic there. Written exactly as its
field declares it, the cell is the constant under a textual comparison and a
numeric one alike, and keeps every value of a column in the column's format.

**Accepted by the operator on 2026-09-14 (#601):** "-999 is good as long as it's
not a valid data value for that column." It is no valid value of any column.
Each plane is evaluated by the `oops` backplane method its configuration names,
with that method's defaults, and its statistic is the least and the greatest
value over the pixels the method leaves unmasked, a plane in radians converted
to degrees (`plane_statistics`), so a masked pixel never reaches a statistic and
each column's range is its method's:

| Plane | Columns | Range, as `oops` computes it | `-999` as written |
|---|---|---|---|
| `body_longitude` | `minimum_body_longitude`, `maximum_body_longitude` | 0 to 360 deg: `longitude`'s default `minimum=0` takes the value modulo 2π | `-999.000` |
| `body_latitude` | `geom:minimum_latitude`, `geom:maximum_latitude` | -90 to 90 deg, planetocentric (`lat_type='centric'`) | `-999.000` |
| `body_incidence_angle` | `geom:*_incidence_angle` | 0 to 180 deg: π less the separation of the surface normal and the arriving photons | `-999.000` |
| `body_emission_angle` | `geom:*_emission_angle` | 0 to 180 deg: the separation of the normal and the departing photons | `-999.000` |
| `body_phase_angle` | `geom:*_phase_angle` | 0 to 180 deg: π less the separation of the departing and the arriving photons | `-999.000` |
| `body_finest_resolution`, `body_coarsest_resolution` | `*_body_finest_resolution`, `*_body_coarsest_resolution` | 0 or more km/pixel: `Surface.resolution` gives the lengths of two perpendicular derivative vectors | `-999.00` |
| `ring_radius` | `rings:*_ring_radius` | 0 or more km: the cylindrical radius of the ring-plane intercept | `-999.0` |
| `ring_longitude` | `minimum_ring_longitude`, `maximum_ring_longitude` | 0 to 360 deg: the cylindrical longitude from the ring plane's J2000 ascending node, `arctan2` modulo 2π | `-999.000` |
| `ring_emission_angle` | `rings:*_emission_angle` | 0 to 180 deg: the emission angle, measured from the sunward pole | `-999.000` |
| `ring_phase_angle` | `rings:*_phase_angle` | 0 to 180 deg, as the body's | `-999.000` |
| `ring_radial_resolution` | `rings:*_radial_resolution` | 0 or more km/pixel: the norm of the radius's derivatives across the pixel | `-999.00` |
| `ring_longitudinal_resolution` | `rings:*_longitudinal_resolution` | 0 or more deg/pixel: the norm of the longitude's derivatives, from radians | `-999.00000000` |

The two time columns are never missing: every row is of an image the data
collection holds, whose supplemental file carries the navigation document's
`observation` block with its start and end, so they declare no missing
constant. Nor do `pds:logical_identifier`, `body_name` and `file_spec`, which
every row has.

**DOIs are products of their own.** The reference carries `BUNDLE_DOI` and a
separate `USERGUIDE_DOI`, and its user-guide label fills a real `<doi>`
where ours has `TODO DOI`. Registering both is an operator step with the
node, not a coding step, and it should be started early rather than
discovered at delivery. Until then `bundle.lblx` and the user-guide label,
which the summary pass renders, carry `<doi>TODO DOI</doi>`, which the XSD's
DOI pattern (`10\.\S+/\S+`) refuses. The prototype keeps them (section 0),
so they are among the errors section 5 expects; a bundle `validate` accepts
with zero errors needs the DOIs registered, which is #708's. The readme's
citation is the same operator step: it gives the bundle's title, "Backplanes
for Navigated Images from Cassini ISS at Saturn", and "DOI TBD" until the
bundle's DOI is registered, when the citation is completed with it.

**Authors and editors are template variables**, not prose: the reference
carries an `AUTHORS` string and an `EDITORS` string naming the node staff
who reviewed the bundle. Ours has a single hardcoded `List_Author` block.

**The source product has no PDS4 counterpart.** A data label's
`Source_Product_Internal` is to name the Cassini ISS product its backplanes
were computed from, and the navigation reads the calibrated image:
`DataSetPDS3CassiniISS` enumerates `<holdings>/calibrated/COISS_2xxx/`
(`_VOLUMES_DIR_NAME`) and maps each index row's `..._1.IMG` to `..._1_CALIB.LBL`
and `..._1_CALIB.IMG`, and `ObsCassiniISS.from_file` reads that file under its
`cassini_iss_calib` configuration. The file is the RMS Node's CISSCAL 4.0beta
product in I/F, as its PDS3 label says, so the readme's "generated from
Cassini ISS calibrated images" is true. The PDS4 Cassini ISS archive has no
calibrated counterpart. `urn:nasa:pds:cassini_iss_saturn::1.1` holds
`browse_raw`, `context`, `data_raw`, `document` and `xml_schema`, and the
registry, asked on 2026-09-14 at
`https://pds.nasa.gov/api/search/1/products/<id>` and its `/latest` and `/all`,
returns 404 for `urn:nasa:pds:cassini_iss_saturn:data_calibrated`,
`...:data_calib`, `...:data_cal` and `...:calibrated`, and for
`...:data_calibrated:1454725799n` and `...:data_calibrated:1455327968w`. The
raw products are there: `urn:nasa:pds:cassini_iss_saturn:data_raw:1454725799n`
(NAC) and `...:data_raw:1455327968w` (WAC) each resolve at `::1.0`, with
`/latest` and `/all` both giving `::1.0` alone, in
`urn:nasa:pds:cassini_iss_saturn:data_raw::1.0`; the LID is
`urn:nasa:pds:cassini_iss_saturn:data_raw:<image number><camera letter, lower
case>`. The cohort's image numbers are made from epochs and name no real
image, so the query took two real COISS_2001 images. The raw label names its
own PDS3 source through `Source_Product_External`,
`external_source_product_identifier`
`CO-S-ISSNA/ISSWA-2-EDR-V1.0:COISS_2001:data/1454725799_1455008789:N1454725799_1.IMG`,
`reference_type` `data_to_raw_source_product`. With nothing calibrated to
name, Phase 6 stopped rather than choose, leaving `SOURCE_IMAGE_LIDVID` the
product's own LIDVID under `data_to_calibrated_source_product` and its `TODO`
in place (section 2.2 row 1). The choices are the operator's, tracked as
#678: name the raw product, `data_to_raw_source_product` at `::1.0`, which is
what the calibrated image was made from but not what the navigation read; name
the calibrated PDS3 product through `Source_Product_External`, as the raw PDS4
label names its EDR; or both.

**The ruling, 2026-09-14 (#678):** cite the calibrated image the navigation read
as an external source product, not the raw PDS4 product. A PDS4 bundle of
calibrated images is coming, and the label switches to `Source_Product_Internal`
once it exists (#687). Phase 7 applies it. `data.lblx` carries a
`Source_Product_External` whose
- `external_source_product_identifier` is the image's volume and the file
  specification of its label within that volume, as in
  `COISS_2001:data/1454725799_1455008789/N1454725799_1_CALIB.LBL`, built in
  `pds4_template_variables` from the image's results path stub, so no label is
  opened for it. The calibrated label's own `DATA_SET_ID`,
  `CO-S-ISSNA/ISSWA-2-EDR-V1.0`, is the EDR's, and its `PRODUCT_ID`,
  `1_N1454725799.122`, the EDR image's, so either would name the raw product,
  where the `_CALIB` file specification under the volume names the calibrated
  one;
- `reference_type` is `data_to_calibrated_source_product`, one of the five
  values the `PDS4_PDS_1O00` Schematron allows on
  `pds:Source_Product_External/pds:reference_type`, and the one for a calibrated
  source;
- `curating_facility` is `PDS Ring-Moon Systems Node`: the XSD types it as a
  string of 1 to 255 characters with no list of values, and the Schematron rule
  on `pds:Source_Product_External` requires it or a `doi`;
- `description` says it is the calibrated image the backplanes were computed
  from.

**Smaller differences Phase 6's product review found**, each adopted or
declined:

- **Science facets.** The reference's bundle label carries `Science_Facets`
  (Visible; Rings; Ring-Moon Systems) and its data collection label and data labels
  (Visible; Ring-Moon Systems). Decided in Phase 8: the bundle, data collection and
  data labels carry one `Science_Facets`, `wavelength_range` Visible and
  `discipline_name` Ring-Moon Systems, as the reference's data collection and data
  labels do, both values in the lists the `PDS4_PDS_1O00` Schematron allows, and
  Ring-Moon Systems taking no `facet1`. Declined: the bundle label's `domain` Rings,
  since this bundle's backplanes cover bodies as well as rings. The comments that only
  asked about them are gone.
- **The SPICE kernel collection label.** Adopted: a `Context_Area` with the
  reference's `Primary_Result_Summary` (`purpose` Observation Geometry,
  `processing_level` Derived) and a `Collection/description`. Its targets are
  named since Phase 8's fix round: every target the data labels name, with
  `collection_to_target`, where the reference's names its two ring targets. Its
  inventory lists the metakernel alone, as the reference's does.
- **The user guide's LID and file name.** Declined. The reference names both
  `f-ring-mosaics-user-guide`; ours has the LID
  `...:document:backplanes-user-guide`, which section 3.1's tree gives it and
  every label and the readme cite, and the file
  `cassini-iss-saturn-backplanes-user-guide.pdf`, the name the template
  directory ships it under, which names the instrument's guide among the four
  #596-#599 write. A label names its file by `file_name`, so the two need not
  match.
- **Optional elements.** Adopted: the readme's `creation_date_time` in
  `bundle.lblx`, a `description` in `kernels.lblx`'s
  `Primary_Result_Summary`, and `records` in `collection_document.lblx`'s file
  entry. Declined: the time range in the bundle's description, since its
  `Time_Coordinates` state it.
- **Cosmetic.** Adopted: every label writes its first modification as "Initial
  version", with no period; the document collection's title has no trailing
  period; and every label lists the narrow-angle camera before the wide-angle
  one.
- **Line endings.** Five templates were CRLF, so five labels were; every
  template and label is LF, as the reference's are.

**Differences Phase 7's product review found** in the index tables and their
labels, against the reference's `global_mosaic_index.lblx`, each adopted or
declined:

- **The exposure times.** Adopted: `pds:start_date_time` and
  `pds:stop_date_time`, after `file_spec` in both tables, each an
  `ASCII_Date_Time_YMD_UTC`: the exposure's start and stop, as the data label
  states them, to the millisecond with a trailing `Z`, which the type's
  patterns in the cached `PDS4_PDS_1O00.xsd` (line 8283 on) accept. Time is the
  usual key a pipeline selects images by.
- **The reference's other columns.** Declined, each for what it means in the
  reference's label:
  - `cassini:observation_id` ("The Cassini Observation ID this product is
    associated with") and `cassini:spacecraft_clock_start_count` and
    `..._stop_count` (a clock reading, the partition omitted): mission
    columns, named from the Cassini dictionary and spelled in the mission's
    own clock format, which the generic index generator could name and write
    only through a per-dataset hook. The supplemental file's navigation
    document records the clock readings (`sclk_start`, `sclk_stop`), so the
    values are there if such a hook is added.
  - `pds:creation_date_time` ("The date and time when the product was
    created"): each data label states its own files' `creation_date_time`,
    and the supplemental files the index reads record none.
  - `nav_quality` ("Subjective quality of the navigation of the images that
    were used to create this mosaic") and `notes` ("Any notes about the
    product"): hand-assigned to each mosaic, and a navigated image carries
    neither.
  - `num_valid_longitudes`, `percent_coverage`, `num_images`,
    `min_image_name` and `max_image_name`: properties of a mosaic's longitude
    grid and of the images it was made from, where a backplane product is one
    image.
  - The `mean_*` columns: the backplane stage records each plane's least and
    greatest value and no mean (`cli/backplanes/statistics.py`).
  - `rings:minimum_corotating_ring_longitude` and
    `rings:maximum_corotating_ring_longitude`: the ends of the range of
    co-rotating longitude over the reference's valid data, computed on the
    circle, in a frame turning with the F ring's core at its mean rate
    (581.964 deg/day, after Albers et al. 2012) and at one with inertial
    longitude at 2007-01-01T00:00:00Z. That frame is the one the reference's
    mosaics are reprojected into, the F ring's science, and a backplane defines
    none. The reference's `rings:minimum_inertial_ring_longitude` and
    `..._maximum_...` are not declined with them: they are inertial longitudes,
    the co-rotating pair plus the frame's rotation since its epoch (row
    `1874525875w` of its `global_reproj_img_index.tab` gives co-rotating
    338.04 to 77.60 and inertial 14.095 to 113.655, the frame having turned
    36.053 deg), and so the quantity our `minimum_ring_longitude` and
    `maximum_ring_longitude` give. Ours differ in the wrap convention, a plain
    least and greatest where the reference's range is taken on the circle,
    which is why they carry names of their own (the Phase 7 record's naming
    table), and in ranging over every pixel with a ring intercept rather than
    the reference's F ring reprojection.
  - The F ring's own columns, its core radius, its node, pericenter and true
    anomaly, and Prometheus's and Pandora's longitudes and radii: that
    bundle's science, not a backplane's.
- **`Header/parsing_standard_id`.** Declined: the reference's is `UTF-8
  Text`, ours `7-Bit ASCII Text`. Every table is written as ASCII, and the
  narrower standard is the one that says so; both are in the schema's list.
- **`Header/description`.** Declined: the reference's is the header line
  itself, wrapped, and ours says what the line is. The `Field_Character`
  blocks name every column, in order, from the same list the header line is
  written from, so a copy of the line in the label would be a second
  statement of the names to keep in step with the first.
- **Column naming.** Adopted, with two exceptions: a column takes a PDS4
  dictionary attribute's name, with its prefix, where it holds the quantity
  that attribute defines, as the reference's `rings:minimum_ring_radius`
  does. The longitudes are the exceptions, since both dictionaries define a
  longitude range as wrapped at the prime meridian and these statistics are a
  plain least and greatest, and so are the body resolutions, which no
  dictionary names. The Phase 7 record gives every column.
- **The title.** Adopted: the reference's is "Global Mosaic Index", and ours
  are "Global Bodies Index" and "Global Rings Index", each label's citation
  description saying which bundle and images the table indexes.

**Differences Phase 8 made**, each deliberate:

- **Stars are not targets.** The reference's bundle label names stars as targets (R
  Lyrae among them), and its context inventory lists nine `star.*` context products:
  they are the stars its occultation data were measured against. A backplane bundle's
  targets are what its images are of, the bodies and the rings, and not what the
  navigation used: an image navigated on stars names no star, and the targets table
  holds none.
- **The document and miscellaneous inventories list no target.** The reference's both
  list its four target context products as `S` members, though no document or
  miscellaneous label of it names a target. Here an inventory's `S` members are the
  context products its own collection's labels reference, so neither lists one, and the
  context inventory lists every one (section 3.5); `bundle_products.secondary_members`
  is unchanged.
- **The ring geometry of an image that is not reprojected.** The reference fills
  `rings:Reprojection_Geometry` for images it reprojected onto a radius and longitude
  grid; ours fills it for images laid out as their own lines and samples, since no other
  class of the rings dictionary fits an image's ring ranges (the next item), with
  `corotating_flag` N, `reprojection_plane` Equator, the midtime as the basis epoch, grid
  parameters holding the resolutions alone, and a description saying so (section 3.7).
  Its longitude range is the one the dictionary defines, wrapped at the prime meridian,
  its minimum above its maximum across zero, where the index tables keep the plain least
  and greatest. Its incidence angle is the mean, the least and the greatest over the
  image's ring pixels, where the reference states one value as all three. Both are the
  fix round's: Part A stated the plain range, and the angle at the ring center as all
  three.
- **Open, for the operator or the Rings Node before delivery: the ring geometry's
  class.** The product review reads the class's definition -- "the parameters describing
  reprojection geometry when the ring(s) is reprojected based on a fixed grid of
  coordinates (e.g., radius vs. longitude)" -- and `epoch_reprojection_basis_utc`'s, "the
  basis epoch for the corotating frame", with `reprojection_plane` "required in labels of
  ring reprojection products", as meant for reprojected rings. The values fit
  their definitions, and the description tells a reader that the arrays are the image's
  own lines and samples, but a harvester would file every product as a reprojection. No
  other rings class fits better: `rings:Ring_Spectrum` holds every one of the ranges but
  the longitudinal resolution, but it describes ring spectra and spectrograms, and the
  dictionary's Schematron requires it to identify the observation's wavelengths. The code
  review's alternative is geom's `Illumination_Geometry`, whose `Illumination_Min_Max`
  holds the least and greatest emission, incidence and phase angles over a target or the
  whole field of view, and whose emission angle geom defines for rings too; the radius,
  longitude and resolution ranges would then have no class. Nothing changes until the
  question is answered.
- **The satellites' `alternate_designation`s and NAIF description.** Declined. The
  reference's satellite `Target_Identification`s carry `alternate_designation`s --
  Prometheus's are "Saturn XVI (Prometheus)", "S/1980 S 27" and "NAIF ID 616" -- and a
  `description` giving the NAIF ID, the center of motion, and its LID and NAIF ID. Ours
  give the name, the type and the reference, from the targets table, which holds each
  target's LID, version, name and type as the registry gives them; the rest is in the
  context product each label refers to, and holding it here would be four more fields
  for each of the table's 21 entries, which scraping the context products (#79) would
  fill.
- **The metakernel label names the union.** Declined: the reference's `kernels.lblx`
  names its two ring targets, what its data are of; ours names every target the data
  labels name, since a backplane bundle's data are of the bodies as well as the rings,
  and the SPICE kernel collection label names the same.
- **The ring block's mean phase and emission angles, `local_identifier`s and
  `Local_Internal_Reference`.** Declined. The statistics record no mean of any plane
  (#253 records the dev guide's promise of one), so the label states the incidence
  angle's mean alone, which the writer takes for the range #47 asked for; a mean for
  each ring plane would be a new statistic for every plane, the index tables' columns
  included. The reference's `local_identifier`s on `Reprojection_Geometry` and
  `Reprojection_Grid_Parameters`, and its `Local_Internal_Reference` from
  `Ring_Reprojection` to its reprojected image's array, tie its one reprojected array to
  its geometry; ours describes every ring plane of the FITS together, so no one array is
  the one the geometry is of, and nothing refers to either class by identifier.
- **Versioned context members.** Declined: the reference's context inventory lists LIDs
  without versions; ours lists each member at its version, `S,<lidvid>`, as section 3.5
  writes every member of an inventory.

Six places where this plan deliberately does **not** follow the reference:

- `populate_template` there discards `template.write`'s `(errors, warnings)`
  return exactly as ours does. Phase 1 fixes that here; it is a defect the
  reference shares, not a convention to copy.
- The reference declares `PDS4_RINGS_1O00_1E00`. Section 3.9 moved us to
  `1F00`, so on this one point we are ahead of it, and the F ring bundle may
  want the same bump.
- The reference writes a product's times twice. Its label's
  `START_DATE_TIME` and `STOP_DATE_TIME` are whole seconds, the start floored
  and the stop ceiled (`generate_pds4_files.py` 2127-2129), which its delivery
  changelog (item 13) explains: its review copy rounded to the nearer second,
  and the intervals it stated missed their exposures. Its `START_DATE_TIME_3`,
  `STOP_DATE_TIME_3` and `MIDTIME_DATE_TIME_3` are the same times rounded to
  the nearest millisecond (2128-2133), which it writes into every product's
  supplemental file header (1827-1842). A data label here follows the second
  for its start and stop: the millisecond is the precision a Cassini exposure
  needs, and the nearest gives back the PDS3 value where a floor or a ceiling
  moves it (section 3.4). For the midtime it follows neither: the reference
  rounds the midpoint of its two epochs to the nearest millisecond, which on
  an exposure an odd number of milliseconds long goes whichever way the float
  falls, and a data label here takes the midpoint of the start and stop it
  writes, a half millisecond rounding up, as PDS3's `IMAGE_MID_TIME` does.
  The collection range follows the first, whole seconds with the start
  floored and the stop ceiled, which contains every product's written start
  and stop.
- The display direction. Every `disp:Display_Settings` in a data label here
  displays `Line` Top to Bottom and `Sample` Left to Right, and both reference
  labels display `Line` Bottom to Top. Each is right for its own array. The
  reference's is a reprojected ring grid whose `Line` 0 is the innermost
  radius. A backplane array is laid out as the calibrated image it describes.
  `oops` reads that image from its VICAR file in record order, the first record
  its first line (`vic.data_2d` in `oops/hosts/cassini/iss.py`); the VICAR File
  Format document (`documents/COISS_0xxx/VICAR-File-Format.pdf` in the PDS3
  holdings) puts one image line in each record of a band-sequential file, and a
  calibrated ISS image is one (`ORG='BSQ'`). The backplane stage evaluates every
  plane over the image's lines by its samples: the body planes on a
  `Meshgrid.for_fov(..., swap=True)`, whose indices are (v, u)
  (`backplanes_bodies.py`), and the ring planes on the snapshot's full-frame
  backplane, whose meshgrid follows the observation's axes, which the Cassini
  ISS host declares as `('v', 'u')`. The writer writes each array as it is,
  unflipped. The Cassini ISS User's Guide
  (`documents/COISS_0xxx/ISS-Users-Guide.pdf`, page 13) says to "display the
  image such that the (line, sample) origin point is at top left", and the
  Cassini ISS PDS4 archive declares exactly that for its raw images: `Line`
  Top to Bottom and `Sample` Left to Right in
  `urn:nasa:pds:cassini_iss_saturn:data_raw:1454725799n`. Decided on that
  evidence; the data label states Top to Bottom.
- The missing constant is written `-999.0`: the value every masked pixel
  holds, as the shortest decimal that reads back as that value when parsed as
  a 64-bit float, so a reader comparing in either precision finds it; the reference writes `-999` for
  its float array. They are the same number. The index tables write it in each
  column's own format instead, since a cell is text: `-999.000` in a column of
  degrees, and each `Field_Character` declares it in that spelling.
- Every `Header` carries a `<name>`, the HDU's `EXTNAME` (`PRIMARY` for the
  first), which is the optional first child the schema gives a `Header`, so
  that a reader can tell which HDU a header belongs to. The reference's image
  file has no `Header`, and the `Header`s of its supplemental tables carry no
  `<name>`.

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
write rather than one per run. `generate_global_index_files` returns a
`GlobalIndexOutcome` carrying the number of labels that failed, beside the
epoch range it gained in Phase 3, and `generate_collection_files`, since
Phase 5, a `CollectionOutcome` carrying that number beside the images whose
products disagree. The index tables are written either way, and so is the inventory
of a collection whose label fails to render; a collection that cannot be
written at all gets neither inventory nor label (section 3.5).

`main_labels` counts the images whose labels it did not write -- including an
image whose generation raised and a batch that did not hold exactly one image
-- carries on to the next image either way, and closes with a line giving
that count beside the images it labeled and the images it skipped, so a
selection that matched nothing reads as zero. A dry run reports what it would
have processed and counts nothing. `main_summary` sums the two failed-label
counts and, since Phase 5, counts the images whose products disagree beside
them; each pass exits 1 when a count it keeps is not zero.
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
nothing. That is #602, which the operator's ruling of 2026-09-14 gave to
Phase 5 (see Phase 5). A skipped image is not that
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
tree, unchanged and still stored under `tests/spindoctor/cli/stats/data/`,
composed from each host's own documents; the cohorts, one `Cohort` subclass per
bundle registered in `COHORTS`, are what bundle generation is asserted against,
and are never stored. The one that exists, `CohortCassiniISSSaturn`, is three
Cassini images.

The Cassini cohort is two navigated images and one that is not. The two shard into
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
`host_cassini.py`'s `cassini_sclk_triple` and `cassini_image_number`, both counted
from a line through two correlation points the mission clock kernel gives --
calibrating where the clock started and the rate it runs at -- and stamped onto
a result by `with_pointing_from_epoch`, which takes no clock argument at all.
Routing the four existing Cassini documents of the statistics set through the
same constructor is #530's own work: it is a coordinated change to four
documents, four filenames, the `filtered` variant's image-number bounds and
both goldens, and it belongs in a PR about the statistics fixtures.

`cohort.py` holds what every cohort shares: `CohortImage`, and the `Cohort`
base, which writes the two roots and the holdings directory and builds the
`ImageFiles` list from what a bundle's subclass supplies -- its images, holdings
layout, camera reading, plane bounds and registered dataset.
`tests/conftest.py` serves the cohorts through the session-scoped
`mini_nav_cohorts` fixture, which writes each cohort class the first time a
test asks for it. `python -m tests.mini_nav_results` takes the set to write,
for a cohort the bundle, and where to write it, all required.
`tests/spindoctor/cli/pds4/conftest.py` keeps `FakePds4DataSet` for the
plumbing questions and gains `CohortBundleEnv`, which runs the registered
dataset a cohort's bundle is built with over the templates it ships, and is
what the phases after this one assert against. Each host's camera frames,
exposure and clock are in a module named for the host (`host_cassini.py`,
`host_voyager.py`), described to `with_pointing` by a `Host`; each host's clock
reader is in a module named for it under `tests/sclk_readings/`; and the tests
only one bundle's cohort can state are in modules named for the bundle.

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
(`et_to_utc`, `et_to_pds4_utc` for the PDS4 spelling and `pds4_utc_midpoint`
for the midpoint of two times so written), used by the
statistics report's `date_from_image_et` and `datetime_from_image_et` and by
`pds4_template_variables`. `START_DATE_TIME` and `STOP_DATE_TIME` read
`navigation_result.times`, each to the nearest millisecond, and
`IMAGE_MID_TIME` is the midpoint of the two as written, a half rounding up
(section 3.4); a navigated image whose navigation recorded no pointing was
failed before anything was written for it, until Phase 7 took the times from
the navigation document's `observation` block (section 3.4). The data
collection range is taken in the global index's read of the supplemental files,
which runs first, and written at whole seconds, rounded outward; with no
range the data collection label is counted as not written. The bundle label's
range is Phase 6's, from the same `GlobalIndexOutcome`.

Tests: known epochs to the strings `et2utc` writes for them, leap seconds
included, and an integration test converting every Cassini cohort epoch
through the kernel (`test_cohort_cassini_epochs_against_kernel.py`); the
shipped data label over the cohort states each navigated image's
start and stop; W1630770594's start, computed a few nanoseconds short of its
millisecond, is written as its PDS3 label states it, and W1629783475's
midtime, on a half millisecond, as its `IMAGE_MID_TIME`; the range contains a
product's written times where they meet its whole seconds; a cohort success
document with no `times` fails its image with nothing written; the collection
range over three supplemental files is the min and the max; the shipped
`collection_data.lblx` over the cohort states the range of its two navigated
images; a summary over no supplemental file writes no data collection label.
The tests only the Cassini ISS Saturn bundle can state are in modules named
for it.

Closes #519, by hand when its PR merges into `rf_pds4_draft_bundle` (section 8).

### Phase 4 — The FITS in the bundle, with its data objects

Done on `rf_pds4_phase4`. The labels pass copies the FITS into `data/` beside
its label and points `BACKPLANE_PATH` at the copy.
`spindoctor/cli/pds4/data_objects.py` builds a descriptor per HDU from the
source FITS before the copy is made, the copy being byte-identical, and
`data.lblx` renders them in two `$FOR` blocks: a `Header` per HDU and
an `Array_2D_Image` per image HDU in `File_Area_Observational`, and one
`disp:Display_Settings` per array in the `Discipline_Area` (section 3.3, which
records why each array has its own).

Each float array declares its masked value: a `Special_Constants` block whose
`missing_constant` is the sentinel from `config_900_backplanes.yaml`, per
section 3.13, so a reader masks on the label rather than on a convention;
`BODY_ID_MAP` declares none and says what its values are instead. The draft
bundle should be built from backplanes that carry `-999`.

Tests: over the cohort, the shipped data label of each navigated image has a
`Header` per HDU and an `Array_2D_Image` per image HDU, each checked at the
byte level -- `SIMPLE` or `XTENSION` at a header's offset, its stated length
ending with the record that holds its `END` card, and an array read big-endian
at its offset equal to the array astropy reads -- with its data type from
`BITPIX`, its unit from `BUNIT`, the configured `missing_constant` on every
float array and none on `BODY_ID_MAP`, and every masked pixel holding it. The
ring image has ring arrays and the limb image none; every
`local_identifier_reference` resolves; every `<file_name>` is beside its label
(criterion 5), and the FITS's stated size and MD5 are the copy's; each data
object's children follow the schema's order. The cohort's frames are square,
so a plane 2 lines by 3 samples rendered through the shipped template holds
the two axes apart. Under the full Schematron rule set (Phase 10), first run
in Phase 6's fix round, the cohort's data labels fail no rule; the XSD wants
their `Target_Identification` (Phase 8).

Closes #69, by hand when its PR merges into `rf_pds4_draft_bundle` (section
8); contributes to #30.

### Phase 5 — Inventories that conform

Done on `rf_pds4_phase5`. The data and browse inventories are written as
`collection_data.csv` and `collection_browse.csv`, the names their labels
give, through one writer in `collections.py`: no header, one `P,<lidvid>`
line per member, each line ending in a line feed alone, the last included
(section 3.5), so the `<records>` that `FILE_RECORDS` counts is the number
of members. The three inventories the template directory ships were already
headerless and LF, with a line feed after the last line; the phase removed
`collection_context.csv`'s trailing `# TODO` line, which a label would have
counted as a record, cited the ISS data user guide in
`collection_document.csv` at `::2.0` by the operator's decision (section
3.6), and left their content otherwise as it was. Versioned
context members and `S` members in the generated inventories (section 3.5),
the `:document:` segment (Phase 6) and the `xml_schema` LIDVIDs (section
3.9) are not this phase's. The global index tables stay `.tab`, header and
all, as section 3.1 settles; Phase 7 moves and reshapes them.

Tests: over one data label an inventory is that one `P` line; every line of
each ends in a line feed and none holds a carriage return; over the cohort,
each shipped collection label names the inventory beside it and states the
two navigated images as its records; and every line of each inventory the
template directory ships is a member, ending in a line feed.

A product review of the phase found that an empty inventory is one no label
can describe. Over an empty `data/`, both inventories were written as empty
files, the data collection label was not written for want of a range, and
the browse collection label was rendered stating `<records>0</records>`,
which `PDS4_PDS_1O00.xsd` rejects. Over supplemental files and no data label
-- a labels pass whose every data label failed -- both labels stated zero
records and the summary pass exited 0. Before the phase the header row was
counted, so the same trees gave one record, valid by accident. The phase
settles it with section 3.5's rule, which takes in section 3.4's for a data
collection with no range: a collection whose label cannot state what PDS4
requires gets neither an inventory nor a label, and counts once among the
labels not written, so the pass exits 1. Tests: over an empty data tree
neither collection is written, the collection generator itself removing what
an earlier run left at the four paths, and the two count as two, the data
collection's error giving both its reasons; over a data label and no
supplemental file the data collection is not written and the browse
collection is; and a summary pass over supplemental files and no data label
exits 1, naming both collections, with neither written.

#602 is this phase's, by the operator's ruling of 2026-09-14 that every data
product needs a browse product. Each inventory lists the labels of its own
kind on disk -- the data inventory the data labels, the browse inventory the
browse labels, where before both were built from the data labels -- and the
summary pass holds each image's products against each other: a data label
with no browse label, or a browse label or supplemental file with no data
label, is an image whose products disagree. Each gets one error naming the
image, the files of it that are there and the label it lacks, and is
counted, and the pass exits 1, its closing line giving those images beside
the labels not written. The inventories still list what is on disk. The
check and section 3.5's empty-collection rule are one rule at two scales:
each image holds all its products, and each collection at least one member.
A data label with no
supplemental file is not checked, since the labels pass writes the
supplemental file first. Phase 7 makes the global index rows exactly the data
inventory's members, which settles #602's question of where the index takes
its images from. Tests: the browse inventory lists the browse
labels on disk and not the data labels; each collection is judged empty by
its own members; one test per disagreement, each counting the one image it
names with the label it lacks; and the summary pass exits 1 on a
disagreeing image alone. A re-review's mutation run then pinned six more
behaviors: an image holding a browse label and a supplemental file with no
data label counts once, its one error naming both; two disagreeing images
count two; a collection label that fails to render keeps its inventory; and
each collection template, when missing, raises over a data tree holding no
label. The global index tests moved to `test_global_index.py`, keeping every
test module under 1000 lines. Under the full Schematron rule set (Phase 10),
first run in Phase 6's fix round, the browse collection label fails no rule,
and the data collection label fails one, for its name and its type: a Mission
Science Data collection names its targets, which is Phase 8's.

Closes #602, by hand when its PR merges into `rf_pds4_draft_bundle` (section
8). The inventory-filename part of #265 is done, and #265 stays open for its
dev-guide output-layout part (Phase 10). The header and line-ending defects
it fixed were never filed as issues.

### Phase 6 — Bundle-level and static products

Done on `rf_pds4_phase6`. `src/spindoctor/cli/pds4/bundle_products.py` owns
the run-level products (section 3.2), and the summary pass calls it last,
after the global index and the data and browse collections; `collections.py`
keeps to collection inventories, and its index generator clears the run-level
products with its own, once it has found the bundle's data directory and
before it reads any supplemental file, through `clear_bundle_products`, which
also removes `document/user_guide/` when that leaves it empty in a bundle on
the local file system (a remote store holds no directory to remove, and
`FCPath` refuses `rmdir` on one), so a rerun leaves nothing stale. The module copies `readme.txt` to the bundle root; copies each static
inventory -- context, document, SPICE kernel, XML schema -- into its
collection's directory and renders the collection label beside it; copies the
metakernel `kernels.ker` and renders `kernels.lblx`; copies the user guide into
`document/user_guide/` and renders its label beside it when the template
directory holds the PDF; and renders `bundle.lblx` last, stating the range the
data collection label states (section 3.4). Every label goes through
`write_label`, every one not written is counted, and the pass then exits 1, as
Phase 1 and Phase 5 had it.

**The bundle label is kept only over a bundle holding every collection it
declares.** It is rendered, its `Bundle_Member_Entry` LIDs are read, and it is
removed and counted when one names no collection label one directory below the
bundle root declaring that LID: a collection whose label failed or was not
written, or one the template declares before any pass writes it. With no range
to state it is not rendered at all. It declares six collections: browse,
context, data, document, spice_kernels and xml_schema. The
`bundle_has_miscellaneous_collection` entry is Phase 7's, with the collection
it names, so that at every phase every entry the bundle label declares
resolves to a collection label in the bundle.

**The `spice_kernels` collection** takes the reference's form: `kernels.lblx`,
a `Product_SPICE_Kernel` with `kernel_type` `MK` over `kernels.ker`, and a
static `collection_spice_kernels.csv` listing
`P,urn:nasa:pds:<bundle>:spice_kernels:kernels::1.0`. Which kernels the
metakernel lists is not settled (#677), so `kernels.ker` is a
placeholder: the `KPL/MK` identification word and a comment saying it lists no
kernels, with no `KERNELS_TO_LOAD` assignment, since SPICE refuses an empty one
(`SPICE(BADVARASSIGN)`) and loads the comment-only file. For want of that
assignment SPICE classifies the file as a text kernel, not a metakernel
(`ktotal` counts it TEXT, not META), which #677 records. When the metakernel's
label is not written, the SPICE kernel collection has no member, so by section
3.5's rule neither its inventory nor its label is written and it counts, and
the bundle label, which declares it, goes too. `kernels.lblx`'s descriptions
and the data label's `geom:SPICE_Kernel_Files` comment say the file lists no
kernels; the data label no longer claims the file lists the kernels used or names
a C-kernel, and its `TODO` is gone. `kernels.lblx` has no
`Target_Identification`, which the Schematron requires of a
`Product_SPICE_Kernel`; that is Phase 8's.

**The user guide** follows section 3.6: its presence in the template directory
decides whether it and its label are written, and its label, written or not,
whether the document inventory lists it. The dataset names it, through
`DataSet.pds4_user_guide_file_name`,
and its label renders from the template of the same stem ending in `.lblx`.
The document inventory the template directory ships lists the guide as its one
`P` line; the pass writes it as it is when the guide's label is in the bundle,
and without its `P` lines when it is not: the PDF absent, which logs one
warning naming the missing file, or its label failed to render. The
guide's label and the XML schema collection label read their files through
path variables, `USER_GUIDE_PATH` and `COLLECTION_XML_SCHEMA_CSV_PATH`, where
they had named them relative to the working directory.
`pds4_required_templates('summary')` names every file the pass takes from the
template directory, the guide's label template among them, and not the guide.

**The `:document:` segment** is back wherever a label or inventory names the
user guide: `data.lblx`, `bundle.lblx` and `collection_document.csv`, and
`collection_data.lblx` and `browse.lblx`, which dropped it too.

**Versioned members** (section 3.5): `collection_context.csv` lists the mission
at `::1.5`, the spacecraft at `::1.4` and both cameras at `::1.2`, and
`collection_document.csv` lists the same four as `S` members beside the ISS
data user guide at `::2.0`, as the reference's does. Targets are Phase 8's.

**The readme** no longer places index files in the document collection, and
its citation gives the bundle's own title, with no `# TODO` line and "DOI TBD"
until the DOI is registered (section 3.13); Phase 7 adds what it says of the
miscellaneous collection. Five templates were CRLF and are LF, as every label
now is, and the smaller differences from the reference the product review
found are adopted or declined in section 3.13.

**The source product** (section 2.2 row 1). The navigation reads the
calibrated image, and the PDS4 Cassini ISS archive has no calibrated product
for it to name (section 3.13, with the registry's answers), so Phase 6 stopped
rather than invent one and left the choice to the operator, tracked as #678.
The operator ruled on 2026-09-14, and Phase 7 applies the ruling: a data
label cites the calibrated PDS3 image as a `Source_Product_External`, built in
`pds4_template_variables` (section 3.13).

Tests, over the cohort: the summary pass adds exactly the files section 3.1
lists for the collections the bundle holds, and the bundle's top level is
exactly those; the bundle label's member entries are exactly the six
collection labels' own LIDs; with a user guide in the template directory,
every label naming a document of the bundle, the data and bundle labels among
them, names the guide's own LID, the document inventory lists it as its one
`P` member, and the readme gives it; the SPICE kernel inventory lists the
metakernel by its label's LID and version, and `kernels.lblx` states the size,
checksum and time of the `kernels.ker` beside it, and `bundle.lblx` the time
of the `readme.txt` beside it; `kernels.lblx`'s `kernel_type` is `MK`, the
SPICE kernel collection label's `collection_type` `SPICE Kernel`, and the
bundle label's member entry for that collection
`bundle_has_spice_kernel_collection`, three values no schema rule can check;
each of the six collection
labels states the lines of the inventory beside it as its records; every
shipped inventory names each member at a version, and no shipped template holds
a carriage return; and both passes run over a template directory holding only
what the dataset declares and a stand-in guide. Over stand-ins: the copied
products are the template directory's bytes; a guide present is copied,
labeled and listed; a guide absent is none of those, with one warning, and a
guide whose label failed is not listed; a SPICE kernel collection whose
metakernel label failed is not written, what an earlier run left at its paths
removed, with one error; a rerun without the guide leaves no
`document/user_guide/`; clearing takes a bundle root relative to the working
directory, and one in a remote store, served from a `fake://` URL; the bundle
label states the bundle's LID and range; a
bundle label declaring a collection not written, or with no range, is not
written, counts, and draws one error; each run-level label that fails is
counted, and every other run-level product is on disk; the driver exits 1 on a
run-level count, hands the run-level products the index's range and logs the
reason one raises; and a summary refused after the index generator's `data/`
check leaves none of the run-level products an earlier one wrote (one refused
by that check, over a bundle with no `data/`, clears nothing and leaves them
in place).

Schema checks over the cohort bundle, with and without a stand-in guide,
offline: xmlschema, with the cartography dictionary the Cassini schema imports
mapped to its cached copy, and every Schematron rule of the five dictionaries
under XSLT match semantics (Phase 10). The XSD finds each data label without
`Target_Identification` (Phase 8) and the `TODO DOI` placeholders in
`bundle.lblx` and the guide's label (section 3.13), and nothing else. The
Schematron finds five failures, all targets and all Phase 8's: two in
`bundle.lblx`; two in `collection_data.lblx`, whose
`pds:Product_Collection/pds:Context_Area` rule makes a Mission Science Data
collection name its targets; and one in `kernels.lblx`, under
`pds:Product_SPICE_Kernel/pds:Context_Area`. Every other label passes every
rule: the data and browse labels, and the browse, context, document, SPICE
kernel and XML schema collection labels. The phase's first checks ran
pyschematron, which runs 116 of the 923 rules and so reported only
`bundle.lblx`'s two and none of the others (Phase 10).

Closes #74, by hand when its PR merges into `rf_pds4_draft_bundle` (section
8). #72, the context collection, stays open for Phase 8's targets. The
`:document:` segment and the metakernel TODO were never filed as issues.

### Phase 7 — The miscellaneous collection and its global index labels

Done on `rf_pds4_phase7`. The global index tables are products of a
`miscellaneous` collection of their own (section 3.1), written by
`src/spindoctor/cli/pds4/global_index.py`, a module of their own, so that
`collections.py` keeps to the inventories: `global_bodies_index.{tab,lblx}` and
`global_rings_index.{tab,lblx}` in `miscellaneous/`, whose LIDs,
`urn:nasa:pds:<bundle>:miscellaneous:global_bodies_index` and
`...:global_rings_index`, are built from the bundle name as the bundle's own LID
is. The old location is no longer written.

**The rows are exactly the data inventory's members.** The index and the data
inventory take the data labels from one function, `collections.data_products`,
so the tables and the inventory cannot disagree about what the bundle holds. The
supplemental file stays the source of the values, and one with no data label
beside it adds no row, where before it added one naming a data label the bundle
does not hold. Every supplemental file is still read for the statistics check,
and the range of the epochs the data collection and bundle labels state is taken
over the members alone, the same images as the rows, so that a supplemental file
with no data label cannot widen it. That is the index half of #602.

A body the backplane document names gets a bodies row only when it has geometry,
a statistic at least, as `targets.has_geometry` decides: the backplane writer
records a body from the image's inventory alone when none of its planes has a
value, and a row for it would measure nothing. This phase gave such a body a row
of masked values; Phase 8's last round reversed that, since on real frames the
table would fill with rows that measured nothing (W1479724035 names nine bodies,
seven with no pixel). A body with statistics for some planes and not others still
has its row, the masked value in the columns it has none for. Rings with no
statistic get no rings row, since the writer records the rings only with their
statistics. The dev guide says both.

**The tables are fixed width**, as the reference's are: a comma-separated header
line, then rows each field of which is padded to the longest value written in
its column -- statistics right-justified, text left-justified -- with a comma
between fields. A field's length comes from its column's values, not from its
format (section 3.8). The tables are written as ASCII through
`FCPath.open('w')`, where the CSV writer went through `get_local_path()` and
`upload()`.

**The labels** replace the two zero-byte templates, following the reference's: a
`Product_Ancillary` whose `File_Area_Ancillary` holds a `Header` over the header
line (`7-Bit ASCII Text`, since every value is ASCII) and a `Table_Character` at
the header's length, with `records` the table's line count less one,
`record_delimiter` `Line-Feed`, and a `Record_Character` whose `fields`,
`record_length` and `Field_Character` blocks come from the same column list the
table was written from, through a `$FOR` over `FIELDS`. The generator computes
each field's number, location and length and hands them to the template with
`INDEX_LID`, `INDEX_TABLE_PATH`, `HEADER_LENGTH` and `RECORD_LENGTH`. Each
configured plane's entry in `config_900_backplanes.yaml` gained an `index` block
naming its two columns, their data type (`ASCII_Real`) and their descriptions; a
column's `unit` is the plane's restated through `statistics_units` (section 3.8)
and its format the one `INDEX_VALUE_FORMATS` gives that unit, both of which moved
with the tables to `global_index.py`. The fixed columns take the reference's
names: `pds:logical_identifier` (`ASCII_LID`), `file_spec` for the data label's
path (it was `path_to_image_file`), and `body_name`, local to the bodies table
since its values are the backplanes' body names rather than PDS target names.
After `file_spec` both tables give `pds:start_date_time` and `pds:stop_date_time`
(`ASCII_Date_Time_YMD_UTC`), the exposure's start and stop, which
`epochs.exposure_times` writes from the epochs the supplemental file records, to
the millisecond with a trailing `Z`, as the data label states them. Section 3.13
records the reference's other columns as declined, each with its reason. The
labels are titled "Global Bodies Index" and "Global Rings Index", after the
reference's "Global Mosaic Index".

**Naming.** A column takes a dictionary attribute's name, with the dictionary's
prefix, where it holds the quantity that attribute defines, and a name of its own
otherwise:

| Plane | Columns | Why |
|---|---|---|
| body latitude, incidence, emission and phase angles | `geom:minimum_*` and `geom:maximum_*` | geom's `Surface_Geometry_Min_Max` and `Illumination_Min_Max` attributes; the latitude is planetocentric (`oops` `latitude` defaults to `lat_type='centric'`) |
| ring radius, emission and phase angles, radial and longitudinal resolutions | `rings:minimum_*` and `rings:maximum_*` | as the reference names its columns; the rings emission angle is measured from the normal on the lit side, `oops` `ring_emission_angle`'s default `pole='sunward'` |
| body and ring longitudes | `minimum_body_longitude`, `minimum_ring_longitude` and their maxima | geom and rings define a longitude range as wrapped at the prime meridian, its minimum above its maximum across it, and these statistics are a plain least and greatest |
| body finest and coarsest resolutions | `minimum_body_finest_resolution` and so on | no dictionary names them |

The rings dictionary says of its radial and longitudinal resolutions "Not
intended to be used as a table field"; the reference uses them as table fields,
and so does this bundle, the quantity being the attribute's.

**The missing value** is decided, and the operator accepted it on #601 on
2026-09-14: where an image has no statistic for a plane, both of its cells hold
the masked value, read from `backplanes.masked_value`, in the column's own
format, and every statistic `Field_Character` declares that text as its
`missing_constant` (section 3.13, which holds the evidence and, column by
column, why no column can hold `-999` as a real value).

**A table with no row** is not written, nor its label, and is not listed:
`records` has a minimum of 1 in `PDS4_PDS_1O00.xsd`, so no label can describe
it. That is not a failure, since a cohort with no rings is a real state, and the
run says so at info level. Both guides state the rule. With neither table
written, though, the miscellaneous collection is not written either, and that
does count (below).

**The collection.** `collection_miscellaneous.lblx`, with `collection_type`
`Miscellaneous`, over a generated `collection_miscellaneous.csv` written after
the tables: a `P` line for each index product whose label is on disk, by its LID
and `INDEX_VERSION` (`1.0`), and then the `S` lines of the document inventory the
template directory ships -- the four versioned context products and the ISS data
user guide at `::2.0` -- read from that one file through
`bundle_products.secondary_members`, so the two inventories cannot disagree. The
three generated inventories share one writer, `collections.write_collection`,
which itself refuses a collection with no member (Phase 5's rule). As section
3.5 has it, the collection takes its members from the index labels, so it is
written only when at least one index product is on disk with its label; with
neither, it is not written whatever the document inventory cites, it counts once
as a label not written, and the bundle label, which declares it, is removed by
its member check, so the summary pass exits 1. The bundle label gained
`bundle_has_miscellaneous_collection`, so all seven of its entries resolve to
collection labels in the bundle, and the readme says what each index table has a
row for, that a table no image gives a row is left out, and that the tables are
in the miscellaneous collection. The summary pass clears the index products and
the collection with the rest of its products before it reads a supplemental
file. A bundle written before this phase keeps its `document/supplemental/`
tables, which no inventory lists, when the summary pass is rerun over it; a
bundle is written into an empty directory, so such a bundle is regenerated. The
rings table's `TODO Add planet name to rings table` stays.

Tests. `test_sd_create_bundle.py` was split first, the summary pass's tests
moving to `test_sd_create_bundle_summary.py` and the stub dataset to
`sd_create_bundle_helpers.py`. Over stand-ins (`test_global_index.py`): the rows
are exactly the data inventory's members; each field is as long as the longest
value in its column, statistics right-justified; a missing statistic is the
masked value in its column's format; a table no image gives a row is left out,
logged at info and not listed; a product whose label failed is not listed; the
inventory's `S` lines are the document inventory's, in its order; a collection
with neither index table is not written, whatever the document inventory cites,
and counts; a refused run leaves none of the index products or collection files
an earlier run wrote; each label is handed its LID; and the index takes a bundle
root given as a string. The range of the epochs is over the data inventory's
members, and a supplemental file with no data label does not widen it
(`test_collection_range.py`). Over the shipped templates
(`test_global_index_cassini_iss_saturn.py`): the `Header` is the header line and
the table follows it; `records` counts the rows and `fields` the columns; every
`Field_Character` is numbered by its position and lands on the column it names;
every statistic field states the data type and description of its column's own
configuration entry and the unit of its plane's statistic; each label names the
table beside it; each row's exposure start and stop are its data label's; a
plane added to the configuration adds a column and a `Field_Character`; a missing
statistic's cell holds the constant its field declares, spelled as declared, in
both tables; each `P` LIDVID of the inventory resolves to a label beside it; and
the collection and its bundle entry state the miscellaneous types, which the
schemas accept for other collections too. The bundle label's test says seven. A
configured plane being removed has no test of its own, being the added-plane
test's mechanism. Each new test was driven red by a mutation, the code review's
surviving mutations among them, all but the one removing the readme's sentence.

Schema checks over the cohort bundle -- with and without a stand-in guide, with a
statistic dropped from every body, and with no ring statistic -- offline, as in
Phase 6: the XSD finds the data labels' missing `Target_Identification` and the
`TODO DOI` placeholders, and nothing in the two index labels or the collection
label; every Schematron rule of the five dictionaries finds Phase 6's five
targets failures and nothing in the three new labels; and each table, read by
its label's field locations and lengths, gives every value as its data type, the
dropped statistic's cells equal to the constant their fields declare. The same
checks after the review fixes find the same, and the product review's
`check_misc.py`, taught the date-time type, finds nothing in the tables or the
miscellaneous collection: each row's start and stop equal its data label's, and
the bodies table has 19 fields in records of 290 bytes (293 with the dropped
statistic's four missing cells), the rings table 16 in records of 272.

**The last round**, after both verifications of the fix round:
- the fixed columns' data types are pinned, one assert each, in both tables;
- the millisecond rule is one constant, `PDS4_EXPOSURE_TIME_DIGITS` in
  `support/time.py`, which the data label and the index tables' time columns
  both import, the dataset still importing nothing from `spindoctor.cli`;
- section 3.13 declines the reference's co-rotating longitudes, whose inertial
  companions are the quantity our ring longitude columns give;
- by the operator's direction, every exposure time the bundle states comes from
  the navigation document's `observation` block (`start_time_et`,
  `end_time_et`) rather than `navigation_result.times`: the data label's start
  and stop, the index tables' time columns, and the data collection's and the
  bundle's ranges. An image whose navigation recorded no pointing is therefore
  bundled like any other, and the labels pass's refusal of it, with its test,
  is gone; the test's replacement bundles the cohort's limb image with its
  times and pointing taken out and holds its label's start and stop to the
  observation block's. The integration test holds the observation block's
  epochs to the leapseconds kernel. The cohort writes the same epochs into both
  places, and its four builds -- plain, with the guide, with a dropped
  statistic and with no rings -- rebuilt at `d63ce7b6` and after this round,
  differ only in the data labels' source-product block;
- by the operator's ruling on #678, a data label cites the calibrated image as
  a `Source_Product_External` (section 3.13), which a cohort test pins;
- by the operator's ruling on #601, the tables' missing value stands, section
  3.13 showing for each column that `-999` is no value it can hold.

**The review's fixes**, after the last round's adversarial review:
- an image whose navigation document records no exposure times in its
  observation block, as a navigation by an earlier version left it, fails the
  labels pass in one line, with no traceback and nothing written, until it is
  navigated again, with no fallback to `navigation_result.times` (section 3.4);
  a test pins it over a document in that version's shape;
- the source-product block's element order, the XSD's, and its description are
  pinned;
- the note on the block's future is a `$NOTE`, which `pdstemplate` does not
  render, so no data label carries it;
- the identifier's comment and docstring say that the results path stub gives
  the volume and the directory and the label's URL the file name;
- `et_to_pds4_utc`'s default is `PDS4_EXPOSURE_TIME_DIGITS`, and the
  integration test passes the constant and derives its own precision from it,
  so a change to the constant changes what that test compares;
- section 3.13 no longer declines the reference's inertial longitudes.

Closes #76, #601 and #678, by hand when its PR merges into
`rf_pds4_draft_bundle` (section 8).

### Phase 8 — Targets, mission area, ring geometry

Split in two by the operator's direction.  **Part A**, the targets and the ring
geometry, is done on `rf_pds4_phase8`.  **Part B**, the mission area, is not started,
and waits on #684.

**Part A.**  Section 3.7 carries the design and the targets table; what landed:

- **The targets table**, `backplanes.target_lids`, filled from the PDS registry for
  Saturn, the nineteen satellites the configuration lists for it and
  `SATURN_MAIN_RINGS`.  `src/spindoctor/cli/pds4/targets.py` reads it (`target_table`),
  says which targets an image's backplane metadata names (`image_targets`), and takes
  them over the summary pass's read (`TargetScan`), each list in the table's order.  The
  stage's body list and ring target became one function each,
  `backplanes_bodies.backplane_body_names` and `backplanes_rings.ring_target`, which the
  writer uses too, so that the guard, a test over the shipped configuration, asks the
  stage's own rules rather than restating them.
- **The data label** names one `Target_Identification` per body its backplane metadata
  names and one for the rings when the metadata holds a ring statistic, each with a
  `data_to_target` reference.  An image naming none is skipped before anything is written
  for it (the fix round; Part A failed it), and a name with no entry raises, naming it,
  which the labels driver counts.  The
  XSD error each data label carried is gone.
- **The run-level labels.**  The bundle label (`bundle_to_target`), the data collection
  label (`collection_to_target`) and the metakernel label (`data_to_target`) name every
  target the data collection's members name, which the global index takes in its one
  read of the supplemental files and returns beside the range in `GlobalIndexOutcome`; the
  driver hands both on.  The evaluator's five target failures are gone.
- **The context inventory** is written, the template directory's four lines and then an
  `S` line per target at its version, and cleared with the other run-level products
  (section 3.5).  The document and miscellaneous inventories list no target (section
  3.13).
- **Science facets**: one, Visible and Ring-Moon Systems, in the bundle, data collection
  and data labels, with no domain (section 3.13).
- **The ring incidence angle** (#47): the backplane stage records `rings.target` and
  `rings.incidence_angle`, `oops`'s `ring_center_incidence_angle` in degrees, for every
  image with a closest planet, and the cohort's fixtures carry them as the stage writes
  them.  An image of backplanes generated before is failed before anything is written for
  it.
- **The ring geometry** (#75): `src/spindoctor/cli/pds4/ring_geometry.py` builds
  `rings:Reprojection_Geometry` from the ring statistics and the angle, which `data.lblx`
  states for an image with ring backplanes and not for one without.

Tests.  Over stand-ins: an image's targets are its bodies with geometry (Part A's test
held a body with no statistic a target too; the last round reversed it), and the ring
target only beside a ring statistic, in the table's order; a name with no
entry is refused by name, by the scan too as it reads; a scan's targets are every
product's, once each; the global index takes only the data collection's members'
targets; the table's entries become targets in order, a version as text; the data label
is handed the targets; an image naming no target is skipped, and one with ring
statistics and no incidence angle fails, each with nothing written; a body with no entry
raises with nothing
written; the ring geometry states each range and the incidence angle in the schema's
order, each resolution in the size a pixel spans, and leaves out a plane with no
statistic; the context inventory lists each target after the template directory's lines;
the driver hands both generators the index's range and targets; the ring stage takes the
incidence angle at the ring center on the ring target, in degrees; and the writer records
the target and the angle in the `rings` block.  Over the shipped configuration: every body
the stage looks for in a Saturn image, and its ring target, has an entry, and every
configured ring plane has a place in the ring geometry.  Over the cohort: each data label
names the targets its backplanes cover, with `data_to_target`; an image given two bodies
names both; the bundle, data collection and metakernel labels name every target, each
with its reference type; every target a data label references is in the context
inventory, each at its registered version; only the ring image states the ring geometry,
its values are the metadata's written as the tables write them, and its fixed elements are
the equator, no co-rotation and the midtime; and the bundle, data collection and data
labels declare the facets.  Each new or changed test was driven red by a mutation: fifty,
all killed.

Checks over the cohort bundle -- plain, with a stand-in guide, with a statistic dropped
from every body and with no ring statistic -- offline as in Phase 7: the XSD finds only
the `TODO DOI` placeholders, two in `bundle.lblx` and four more in the guide's label when
it is there; every Schematron rule of the five dictionaries finds nothing; and the table
check (`check_misc`) finds what it found at `758c09de` and nothing new -- the user guide's
LID resolving to nothing where the PDF is absent, and, in the two builds that alter
supplemental files after the labels pass, the data labels' sizes and checksums of those
files.  Against the `758c09de` builds, nine files of each differ, as they should: both
supplemental files, 123 bytes longer for the `rings` block's target and angle, so both data
labels' statements of their size and checksum; both data labels for the facets and the
targets, and the ring image's for its ring geometry; the bundle, data collection and
metakernel labels for their targets, and the first two for the facets; and the context
inventory, three target lines longer, with its label's size, checksum and records.  In the
build with no ring statistic the ring target is in neither the run-level labels nor the
context inventory, since that build strips the statistics from the supplemental files after
the ring image's data label names it.

**The fix round**, after the product and code reviews of Part A.  Section 3.7 carries the
design; what changed:

- **An image whose backplanes cover no target is skipped**, not failed, before the checks
  that fail an image, with one log line, and the run's exit status is unaffected: 17.6%
  of COISS_2001-2116 are such images, the F-ring frames among them, which come in only if
  #618 changes the ring target (section 3.7).
- **Statistics over the product's own pixels.**  The writer takes every statistic from
  the merged planes, a body's over the pixels the body identity map gives it and the
  rings' over every pixel, and the stages compute none; `merge.body_naif_id` is the one
  NAIF lookup the merge and the writer share.  The index tables' values change for any
  image where a body covers the rings or another body; the cohort's do not.
- **The wrapped ring longitude range**, `statistics.wrapped_range`, recorded as the ring
  longitude statistic's `wrapped_min` and `wrapped_max` and stated by the label.
- **The incidence range.**  The ring stage keeps `ring_incidence_angle` at each pixel, the
  writer records its `min`, `max` and `mean` over the merged ring pixels beside the
  center's `value`, and the label states those three; the check for backplanes of an
  earlier version asks for the range.
- **The SPICE kernel collection label** names the targets, with `collection_to_target`.
- **Stand-in values** in the generic ring tests, three of which held Saturn's.
- **Pins** for Part A's two surviving mutations: the summary pass reads each supplemental
  file once (M15), and only the context inventory lists a target (M18).
- **Section 3.13** records the ring geometry's class as open, for the operator or the
  Rings Node, with geom's `Illumination_Geometry` as the code review's alternative, and
  the four other differences from the reference the product review found, each declined
  with its reason.  The dev guide and section 3.7 say that `rings:Ring_Spectrum` holds the
  ranges but describes spectra, and that nothing checks a summary pass over an earlier
  labels pass.

Tests.  An image whose backplanes cover no target is skipped with nothing written,
whatever its navigation document records, and a labels run over one, generating for
real, counts it skipped, writes nothing and exits 0.  The writer's statistics leave out
the rings, and the part of a body, that a nearer body covers, and a body hidden entirely
has none; the wrapped range and the incidence range leave out the covered rings too; the
incidence range leaves out a ring pixel with no angle, and its mean is not its median;
and the wrapped range measures gaps against the coarsest pixel.  `wrapped_range`: an arc
across zero starts at the greater longitude, one not across it is the plain range, a tie
keeps the plain range, 360 is zero, no gap wider than the resolution is the whole
circle, and a plane in radians is wrapped in degrees.  The ring stage keeps the angle at
each pixel, on the ring target and from the sunlit side, in radians, the masked value
off the rings.  The label states the wrapped range and the incidence's mean, least and
greatest; backplanes with no incidence angle, and with the center's alone, both fail;
the kernel collection label names the targets; the summary pass reads each supplemental
file once; and only the context inventory lists a target.  Over the shipped
configuration, the ring longitude and its resolution are in radians.  Each new or changed
test was driven red by a mutation: forty-one, all killed.

Real frames, re-navigated so that their documents record the exposure times, then
backplaned and labeled with this code: every ring value the data labels of N1671602206
and W1626850595 state equals its FITS array's, thirty of thirty, N1671602206's ring
longitude now 241.932 to 258.852 deg and its greatest radial resolution 1151.4 km.  The
code review's star-only frame N1607625633 is skipped, the labels pass exiting 0 with
nothing written.

Checks over the cohort bundle, offline as before, in six builds: Part A's four, one
whose ring image's rings block is N1591060671's as the stage now writes it, and one
whose limb image's metadata is the F-ring frame N1467350440's.  The XSD finds only the
`TODO DOI` placeholders, two in `bundle.lblx` and four more in the guide's label when it
is there; every Schematron rule of the five dictionaries finds nothing; and the table
check finds what it found for Part A.  Against Part A's builds, four files of each of its
four cases differ: the ring image's supplemental file, 178 bytes longer in the plain
build for the wrapped range and the incidence range, and so its data label, which also
states them in its new description's terms; the SPICE kernel collection label, for its
targets; and the readme, for its sentence on the images left out and the Phase 7 fix
merged at `8d704fc6`.  The limb image's products and the context inventory are
unchanged.  The crossing build's ring label states the longitude from 359.687 to
0.402 deg; the cohort's own ring label states 216.000 to 204.706, and the incidence
64.681, 64.680 and 64.682.  In the last build the limb image is skipped with the log
line, the bundle holds the ring image alone, and the context inventory lists Saturn and
its rings but not Enceladus.

**The last round**, after the product and code verifications of the fix round:

- **Statistics in double precision.**  `plane_statistics` restated a float32 plane's
  radians in degrees in float32, so a size per pixel in degrees, written to eight
  decimals, lost its last digit: W1626850595 stated 0.27105078 where its FITS array,
  converted in double, gives 0.27105080.  It now converts in float64, and the wrapped
  range is found over the same values.
- **NaN in the wrapped range** is ignored, as `nanmin` and `nanmax` ignore it for the
  plain range, and longitudes that are all NaN give NaN for both ends.
- **One rule for a body with geometry.**  A body counts only when its backplanes hold at
  least one statistic, as `targets.has_geometry` decides, and the one rule serves the
  skip, the targets, which the run-level labels and the context inventory follow, and
  the rows of the bodies index (section 3.7).  The last reverses Phase 7's row of
  missing values for a body with no statistic, which on real frames would fill the
  table with rows that measured nothing; a body with statistics for some planes and not
  others keeps its row.
- **Pins** for the fix round's two surviving mutations: the cohort's rings index row
  states the plain ring longitude, 0 to 360, where its data label states the wrapped
  range, 216.000 to 204.706 (N29); and the writer test's ring pixels span 41 to 49 deg,
  mean 43, about a center of 40, the least, the greatest and the mean each pinned on
  its own (N17).
- **The readme** says which images the bundle leaves out and why, naming the 19
  satellites, and five rst lines the fix round and this round left past 90 characters
  are wrapped.
- **Older body-only backplanes**: no change, since backplanes are regenerated before a
  delivery build.

Tests.  A float32 radian value whose degrees differ in the eighth decimal is stated as
its float64 conversion; a NaN among the longitudes leaves the arc as it is, and
longitudes all NaN give none; a frame naming only a body with no statistic is skipped;
of two bodies, one with a statistic and one without, the data label names one target,
the bodies index has one row, and the one without is no target; the rings index states
the plain longitude; and the writer's least, greatest and mean incidence each stand
apart from the center's.  Each new or changed test was driven red by a mutation:
eleven, all killed.

Checks.  At `6d86f0d5`, ruff, the format check (796 files) and mypy (797 source files)
are clean, and so are sphinx and pymarkdown; the unit suite passes, 12647 and 6
xfailed.  The plain cohort bundle, rebuilt, has only the `TODO DOI` placeholders for the
XSD, nothing for any Schematron rule, and the table check's seven findings of the fix
round, the user guide's LID with no PDF.  Against the fix round's build, both index
tables are byte-identical and every value a label states is unchanged: the statistics'
double-precision digits change the two supplemental files, 8154 to 8149 bytes and 9179
to 9229, and so each data label's statement of that file's size and checksum; the readme
is the other difference.

**Part B: the mission area.**  `cassini:ISS_Specific_Attributes` is to be filled from the
Cassini facts the navigation document's `observation` block records, which #684 adds, on
a branch against `main`, by the operator's direction.  It reaches this stack after #684
merges and `main` is merged into `rf_pds4_draft_bundle`, and it replaces the source the
`cassini:*` mapping of `pds4_template_variables` reads rather than declaring that
source's columns.  The template content it adds names the bundle, its version and any
schema through the variables every template is handed (sections 3.10 and 3.13), never by
spelling them.

*History: the index row.*  The mapping reads the image's PDS3 index row, and a run hands
over a row of two columns: the enumeration reads only the columns it declares --
`_INDEX_COLUMNS` plus `_INDEX_CAMERA_COLUMNS`, which for Cassini are
`FILE_SPECIFICATION_NAME` and `INSTRUMENT_ID`, plus the four a BOTSIM grouping adds --
and `PdsTable` returns those alone, so every one of the seventy `cassini:*` variables
takes its default on every real run.  Declaring the columns the label reads was to be
this phase's, with its second half: a dozen of those variables name columns the COISS
index has no such column for (`SPACECRAFT_CLOCK_COUNT_PARTITION` against the index's
`SPACECRAFT_CLOCK_CNT_PARTITION`, `FILTER1` and `FILTER2` against one two-element
`FILTER_NAME`, `GROUND_SOFTWARE_VERSION_ID` against `SOFTWARE_VERSION_ID`,
`START_TIME_DOY` and `STOP_TIME_DOY` against `START_TIME` and `STOP_TIME`, and the
`EXPECTED_MAXIMUM` / `VALID_MAXIMUM` / `INST_CMPRS_RATE` pairs against one array column
each), and `cassini:image_mid_time` is filled from `IMAGE_TIME`, the shutter-close time,
where the index carries an `IMAGE_MID_TIME` (`04:25:36.045` against `04:25:35.815` for the
cohort's limb image).  The cohort keys its rows by the index's own names, so both halves
are visible there.  Part B takes the facts from the observation block instead.

Part A closes #73, #75, #47 and #72, the context collection, which Phase 6 wrote and whose
targets Part A adds, by hand when its PR merges into `rf_pds4_draft_bundle` (section 8);
it contributes to #53's template list.  #79 and #618 stay open.

### Phase 9 — Parameterize the bundle name and version

Done on `rf_pds4_phase9`. The bundle's name and version, and the schemas its labels
declare, are each set in one place, the dataset's entry in `config_950_pds4.yaml`, and
every template takes them from there (sections 3.9, 3.10 and 3.13).

**One set of variables for every template.** `src/spindoctor/cli/pds4/bundle_variables.py`
gives the variables every template of a bundle is handed beside its own: `BUNDLE_LID`,
`BUNDLE_VERSION`, `INFORMATION_MODEL_VERSION`, `PDS4_<PREFIX>_SCHEMA` and
`PDS4_<PREFIX>_SCHEMA_XSD` for each dictionary, and `XML_SCHEMA_LIDVIDS`. Every render
takes them: the data and browse labels, the collection and index labels, the run-level
labels, and the readme and the four static inventories, which are now rendered from the
template directory rather than copied. A readme or an inventory that does not render
counts, as a label does, and a static collection whose inventory does not render is not
written, nor its label; the document inventory is the exception, since the fix round
(below).

**The version.** `bundle_version` beside `bundle_name`, read through
`pds4_bundle_version()`, which has no default, is the `version_id` of the bundle, of
every collection and of every product the bundle writes, each label's
`Modification_Detail` included, and the version in every LIDVID citing one of them: the
data and browse LIDVIDs the dataset builds, the index products the miscellaneous
inventory lists (`INDEX_VERSION` is gone), and the primary members of the document and
SPICE kernel inventories (section 3.10). The dataset's template variables build the
product LIDVIDs through its own hooks and no longer carry `BUNDLE_LID` or the unused
`BUNDLE_LIDVID`, which every template now takes from the bundle's variables.

**The name.** Every place a template spelled the bundle's name takes `$BUNDLE_LID$`: 27
spellings in 15 files, each collection's and product's LID, every reference to a
collection or to the user guide, the bundle label's seven member entries, the primary
members of the document and SPICE kernel inventories, and the readme's two identifiers.
The code spelled it as the configuration's value, which stays, and as
`_default_pds4_bundle_name`'s fallback, which the fix round removed. Lookalikes that stay as they are: the source bundle
`cassini_iss_saturn` in the ISS data user guide's LID, in five templates, an external
product at its own version; the template directory's name, `cassini_iss_saturn_1.0`; and
the user guide's file name, `cassini-iss-saturn-backplanes-user-guide.pdf`, which the
dataset names.

**The schema locations.** `information_model_version` and `schemas` in the same entry,
read through `pds4_information_model_version()` and `pds4_schemas()` (section 3.9). The
fourteen templates' PDS declarations, the data label's other four and the fourteen
information model versions are variables, and the XML schema inventory is a template over
`XML_SCHEMA_LIDVIDS`, its LIDVID spelling unchanged. The schemas sit with the bundle's
name in the dataset's entry rather than in a shared place or a module (section 3.13).

Tests. Over the cohort (`test_configured_bundle_cassini_iss_saturn.py`), built with a
stand-in user guide from a configuration naming the bundle
`saturn_backplanes_other_bundle` at version `3.7`: the bundle is written under that name
with every collection its label declares; no label, inventory, table, metakernel or
readme holds the shipped name; every LIDVID of the bundle's own products carries `3.7`,
every `version_id` is `3.7`, and every logical identifier is under the configured name;
and no template variable is left unrendered. Moving any one of the five dictionaries'
schemas in the configuration moves it in every label declaring that namespace, in the
`xml-model` instruction and in `xsi:schemaLocation`, leaves the shipped location in no
label, and puts the moved LIDVID in its place in the XML schema inventory; every label
states a configured information model version; and the shipped configuration gives a
schema for exactly the dictionaries the shipped templates declare. Over the shipped
configuration, `bundle_version` is set, as text a `version_id` can hold, and a LIDVID
carries a configured version. Over stand-ins, a readme and a static inventory that fail
to render are counted, and the inventory's collection is left with neither file, what an
earlier run left at either path removed. The stand-in datasets gained the three hooks,
the one rendering the shipped index templates handed the shipped schemas; the shipped
inventories are judged as they render; and the cohort bundle helpers moved from the
conftest to `cohort_bundle.py`, keeping it under a thousand lines. Each new test was
driven red by a mutation: twenty-five, all killed, one of them (a failed inventory's
label left from an earlier run) only once a test was added for it.

Checks. At `faa5dc32`, ruff and format clean (799 files), mypy clean (800 source files),
sphinx and pymarkdown clean; the unit suite passes, 12659 and 6 xfailed. Integration,
one file at a time: the cohort's epochs and clock files pass, and `test_cmatrix_readers.py`
(2 of 28) and `test_results_index_consumers.py` (4 of 21) fail as Phase 8 recorded, the
Galileo fitted-rotation tests. The cohort bundle, rebuilt with and without a stand-in
guide under the shipped configuration, is byte-identical to the `88f19f57` build but for
timestamps (17 files identical and 15 differing only in timestamps without the guide, 18
and 16 with it); the XSD finds only the `TODO DOI` placeholders, two and six; every
Schematron rule finds nothing; and the table check finds nothing with the guide and,
without it, the seven references to the guide's LID it found before. Built under the
other name and version, the bundle checks the same.

**The fix round**, after the review of `244507cd`, which found no high finding:
- **External references keep their versions.** The renamed-build test now also builds
  the bundle under the shipped configuration and holds every LIDVID outside the bundle's
  LID, file by file, to that build's; before, a context product or the ISS data user
  guide rendered at the bundle's version survived every test.
- **The document inventory's secondaries** are taken from that inventory as it renders
  with the bundle's variables, so the miscellaneous inventory cannot cite an `S` line the
  document collection writes otherwise. No shipped `S` line holds a variable. The global
  index step renders it and raises on one it cannot render, so a document inventory that
  cannot be rendered stops the summary pass there, with exit status 1 and only the index
  tables and their labels written, where the other static inventories are counted. The
  template being our own, that is kept, and the dev guide says so.
- **The information model version and the `pds` schema are tied** by a test over the
  shipped configuration: the schema's file name carries the version's code, its LIDVID
  the version's first two parts, and every other dictionary's file name the same build's
  code (section 3.9).
- **A failed readme or static inventory** costs the bundle label too, which the dev guide
  now says: the bundle label declares the collection and states the readme's time.
- **The configuration's comment** over the commented-out entries says what they give and
  when a dataset bundles. The entries are unchanged.
- **The name has no code default.** `_default_pds4_bundle_name()` is gone, and a missing
  `bundle_name` stops the run as a missing `bundle_version` does. Nothing depended on it:
  the one dataset with no entry, `coiss_cruise`, is refused by the template check first.
- **The entry stays in `config_950_pds4.yaml`**, as section 3.10 records.
- **`sphinx -n`** is left as it is: its nine unresolved references come from the API
  reference documenting `spindoctor.dataset.dataset` as `:noindex:`, which is older than
  this phase.

Tests. The renamed-build test holds the external references to the shipped build's; a
stand-in document inventory whose `S` line names the bundle through its variables is
cited by the miscellaneous inventory as the document collection writes it; the shipped
configuration keeps the information model version and the schemas of one build; and the
test over the shipped configuration that pins the name is named for it. Each new or
changed test was driven red by a mutation: nine, all killed -- the review's E1-E3 and
X1-X3, the secondaries read unrendered, a rings schema of another build, and the shipped
name removed, which survived while the name had a default.

Checks. At `66967ad2`, ruff and format clean (799 files) and mypy clean (800 source
files); the unit suite passes, 12661 and 6 xfailed. Integration, one file at a time, as
before: the epochs and clock files pass, and the two files holding the six Galileo
fitted-rotation tests fail as Phase 8 recorded. The plain cohort bundle, rebuilt, is
byte-identical to the `88f19f57` build but for timestamps; the XSD finds only its two
`TODO DOI` placeholders, and every Schematron rule finds nothing. With the guide, and
under the other name and version, it checks as the round before did.

Closes #71, by hand when its PR merges into `rf_pds4_draft_bundle` (section 8).

### Phase 10 — Validation, the integrity pass, and the draft run

In two parts. Part A is what the synthetic cohort alone makes possible: the
bundle check, the test that gates it, and `--check-only`; it is done, and its
PR is #707. Part B, the draft run over a real COISS volume, is deferred to
#708 (section 0): it needs the DOIs and a fresh navigation.

#### Part A — The bundle check, its gate and `--check-only`

Schema validation as a repeatable command: `sd_create_bundle check <dataset>`,
a third subcommand, the `spindoctor.cli.pds4.check` package. It checks a
bundle tree that has already been written, reading that tree and the schemas
its labels name, and writes nothing in the tree, not even a log. It prints one line per
finding -- the file, whether it is an error or a warning, the check that found
it, where in the file, and the message -- then the number of each, and exits
non-zero on any error. A warning is a finding `validate` also reports as a
warning: an unresolved reference to a product of the bundle
(`reference_not_found`), a product no inventory lists (`unreferenced_member`),
and a Schematron assert or report whose `role`, or whose rule's, is `warning`
or `WARN`, the two spellings the labels' Schematron use (72 rules and one
assert of the common dictionary's). Every other finding is an error. It takes the
dataset, `--config-file` and `--bundle-results-root`, since one of its checks
reads the configuration. `lxml`, `elementpath` and `xmlschema` are runtime
dependencies, their floors the versions tested on Python 3.11 and 3.12: 6.0.4,
5.0.4 and 4.1.0.

**The schemas are fetched, not shipped,** by the operator's decision of
2026-09-15: "Fetch schemas, don't ship them. A smaller package. The check
needs the network, and the tests keep local copies." Every URL the check reads
-- a label's `xsi:schemaLocation` URLs and `xml-model` `href`s, and each
schema's `xs:import` `schemaLocation` -- is resolved one way. By default it is
fetched through `filecache` into the cache `_filecache_spindoctor_pds4_schemas`,
under `$FILECACHE_CACHE_ROOT` when that is set and otherwise in the user's own
`$XDG_CACHE_HOME` or `~/.cache`, which keeps
each download, so a later check fetches nothing it already has; with
`sd_create_bundle check --schema-dir DIR` it is the file of the URL's name in
`DIR`, and nothing is fetched. `xmlschema` is allowed only local files and
reads every URL through that rule, so an import resolves by its own URL, as
`validate` resolves it. `xmlschema` reads a namespace once in a set, though:
every data label declares GEOM `19B0` itself, and that build serves the
Cassini schema's import of `19A0`, which is not read. `validate` reads both,
to the same verdicts (Part A's code review), and a fetch-mode check from an
empty cache fetches eleven files, `19A0` not among them; `19A0` serves only a
label declaring the Cassini schema and no geometry build. Every label's
findings are the same as they were. A `--schema-dir` given as a relative path
is read from where the check runs. The namespace catalog is gone, and no
namespace is named in code. A URL that cannot be resolved -- one that cannot be
fetched, or with no file of its name in the directory -- is a finding naming
it, and the check goes on; every warning `xmlschema` raises while building a
schema set is a finding, so an import that fails cannot pass silently. A
schema set that cannot be built at all -- a URL paired with a namespace its
file does not define, say -- is one finding, and the label is checked without
it. The package held the six XSD and six Schematron files, of the five
dictionaries the labels declare and of `PDS4_CART_1O00_1970`, in
`src/spindoctor/cli/pds4/schemas/` until then; they now live, byte for byte
the files at their URLs, in `tests/spindoctor/cli/pds4/check/schemas/`, with
GEOM `19A0`'s XSD fetched once from pds.nasa.gov beside them, and the check's
tests read them there, so the suite needs no network (criterion 4). Two
tests hold every schema a dataset's `pds4.<dataset>.schemas` names, and every
import of a copy, to having a copy; one test, marked `integration`, runs the
check in fetch mode over the plain cohort bundle and holds it to the offline
findings. The copies are NASA's as published, misspellings included, so the
spelling gate on its way to `main` has to skip `.xsd` and `.sch` (#705).

The rules cannot be
run by `lxml`'s ISO Schematron: on 2026-09-14 the product reviewer found
that `lxml.isoschematron` refuses the PDS4 1O00 Schematron, reporting that
it "does not work with schemas using the xslt2 query language".
Nor can pyschematron 1.2.1 be the gate, though a control label with a wrong
`offset` failed the rules it ran. It matches a rule by evaluating the rule's
context from the node's parent (`_node_matches_context`,
`pyschematron/direct_mode/xml_validation/validators.py` 306-325), so a rule
whose context is a relative path of more than one step -- `a/b`, `p:c/p:d`,
`a//b`, `b/@x` -- never fires: of the 923 rules in the five cached `.sch` files
it runs 116, skipping 267 of the PDS dictionary's 343, 101 of the Cassini
dictionary's 102, 5 of the display dictionary's 7, 225 of the geometry
dictionary's 251 and 209 of the rings dictionary's 220. 57 of the rules it
skips match something in our labels, and a control `kernels.lblx` whose
`kernel_type` is `XX` passes both it and the XSD. The check therefore evaluates
the rules with XSLT match semantics, the ISO skeleton's: a node matches a rule
when it is in `//(context)` from the document node, only the first matching
rule in a pattern fires, schema and pattern variables are evaluated at the
document node, and a rule's variables at the node it matched. Its evaluator is
a port of the one written that way for Phase 6's reviews, which agrees with
pyschematron on the 116 rules pyschematron runs and catches every control.
For a context that is a union of path expressions joined by `|` -- every
context of the labels' Schematron -- it selects a rule's nodes as `//` before
each branch, the same node set as `//(context)`: over the Phase 9 review's two
cohort builds the two agree for all 12,953 (label, rule) pairs, and the code
review found them identical for all 19,470 pairs of its builds and controls.
Any other context, one holding the `union`, `intersect` or `except` keyword or
a comment, is selected as `//(context)` itself, since a split at `|` would
change what it selects; a test holds each such shape to that. Over the two
data labels the code reviewer measured the check at 0.57 and 0.93 s, the
reference evaluator's method at 1.72 and 2.83 s, and literal `//(context)` at
21.3 and 54.0 s. The NASA PDS `validate` tool is the authority for the draft
acceptance and checks what the check does not (below), but it is Java and
does not belong in this repository's CI; the Python check is the gate that
runs on every PR, and `validate` is run by hand for the draft.

**The integrity checks**, from acceptance criteria 5-7 and both reviews:
every `<file_name>` names a file beside its label, by its name alone and once,
with the `file_size` and `md5_checksum` the label gives it; every file in the
tree is a label or is named by exactly one label; no label holds a `[[[`
marker; no element is empty unless its `xsi:nil` is true (`true` or `1`, white
space trimmed); no two labels declare one LID; and every `lid_reference` and
`lidvid_reference` under the bundle's own LID resolves to a product in the
tree, and a LIDVID to that product's version, one that does not being a
warning. Every `P` member of every inventory resolves to a product in the
tree at its version, an error otherwise, and every product whose label lies
in a collection's directory is listed by that collection's inventory once:
twice is an error and not at all a warning
(`spindoctor.cli.pds4.check.inventories`). And each global index table's
records name products the tree holds, with each product's label as their
`file_spec` and their start and stop times as that label states them, and
each product whose label names a supplemental file has the records the file
calls for (`spindoctor.cli.pds4.check.index_tables`). That is bounded by what
the Phase 7 reference tool's `main_bundle` compared, less the statistics a
record holds, which the summary pass's own formats write and its tests pin.
None of it knows the Cassini layout: the index's columns are
`global_index.py`'s, which every dataset shares, so the module is generic.

A label's references to context products are left to `validate`, which checks
them against the registered context products. Neither tool checks the
versions an inventory's secondary members give products outside the bundle:
`ring.saturn.rings::1.9` and `mission.cassini-huygens::1.0` in the context
inventory pass the check, `validate` and the Phase 7 reference tool alike.
The run over a real volume (#708) checks those versions against the
registry by hand.

**The gate is a test in the default suite**,
`tests/spindoctor/cli/pds4/check/test_check_cassini_iss_saturn.py`, so pytest
runs it both in `scripts/run-all-checks.sh` and in CI, with no script step of
its own. It builds the cohort bundle twice through
`tests/spindoctor/cli/pds4/cohort_bundle.py`, plain and with the stand-in
guide, runs the check over each, and asserts that the findings, each by its
file, check, location and message -- or, for an XML schema error, its kind,
the class of the xmlschema validator that failed, since `pyproject.toml` sets
only a floor and a release may reword xmlschema's messages -- are exactly
these, errors and warnings apart:

- errors: the `TODO DOI` placeholders, from the XSD, until the DOIs are
  registered (section 3.13) -- `bundle.lblx`'s one in both builds, and the
  guide label's two in the guide build; and each data label's empty
  `cassini:ISS_Specific_Attributes`, from the integrity check, in both builds,
  until Part B of Phase 8 fills it (section 2.2, row 2);
- warnings, in the plain build alone: the seven references to the user
  guide's LID -- from the two data labels, the two browse labels, the data
  collection label, the metakernel label and the bundle label -- which a
  bundle without the PDF does not hold (section 3.6, criterion 9), and which
  `validate` reports as warnings too. The operator ruled to keep the
  references, and criterion 9 allows a draft without the PDF, so a plain
  build passes the check with these seven warnings.

The check was expected to report the DOI placeholders alone. It reports the
other two because each is true of the bundle -- an element with nothing in it,
a reference to a product the tree does not hold -- and each leaves the list
when what it records changes. Part A of Phase 8 removed every other finding:
each data label's missing `Target_Identification`, and the Schematron's
five failures, all targets -- `bundle.lblx`'s two, for its targets' name and type;
`collection_data.lblx`'s two, under `pds:Product_Collection/pds:Context_Area`,
which requires a Mission Science Data collection's targets' name and type; and
`kernels.lblx`'s one, under `pds:Product_SPICE_Kernel/pds:Context_Area`.

The Schematron checks a value against its rule's vocabulary, not against what
the product is: over Phase 6's cohort bundle the evaluator refuses a
`kernel_type`, a `collection_type` or a member entry's `reference_type` of
`XX`, and accepts `kernel_type` `FK` over the metakernel, the SPICE kernel
collection typed `Document`, and its member entry typed
`bundle_has_document_collection`, each a value the vocabulary holds. A value
that is wrong but allowed is left to the tests and to `validate`; a cohort
test pins those three. Part A's controls repeat all six over the cohort's
bundle: each `XX` fails, `kernel_type` `XX` by the rule on
`pds:SPICE_Kernel/pds:kernel_type`, and each wrong-but-allowed value passes.

**The command reads each table through its own label.** Neither schema can see
whether a table agrees with the label describing it. Of eleven deliberate
breaks of the index labels in Phase 7's product review, the XSD caught one
(`records` 0) and the Schematron three (`fields`, `parsing_standard_id`,
`record_delimiter`); the other seven -- `field_length`, `Header/object_length`,
the `Table_Character`'s `offset`, a row one byte short, `missing_constant`
respelled `-999`, `unit` `rad` and `data_type` `ASCII_Integer` -- were caught
only by reading the table through its label. `validate` would pass the
respelled constant too, since `-999.000` is a valid real and is never compared
with the constant. So the command gains a check that reads each table by its
label alone: the `Header` and `Table_Character` offsets and lengths, `records`,
`fields`, each field's location, length and data type, its unit against the
configuration, and each missing cell against its declared constant.
`spindoctor.cli.pds4.check.tables` is a port of the product review's
`check_table` and `check_units`, over every table-bearing label in the tree,
not only the two global indexes: the index tables' `Table_Character`, with its
`Header`, and the inventories' `Inventory`, a `Table_Delimited`. The objects
of the file area are held to tile the file, and each value to the simple type
of its `data_type` in the common dictionary the label declares. The fields of
a group, a `Table_Binary` -- none is in the tree -- and a file area that also
holds an object of another class are left to the XSD and to `validate`. The
seven breaks are its controls, each failing its test, with an inventory whose
`records` is one too many and a missing cell spelled otherwise than its
constant. From the code review: a delimited record ends in its declared
delimiter byte for byte, so an inventory with CR-LF endings under a
`Line-Feed` label is found; each field's `field_number` is its position in a
record holding no group; and a global index table, whose header and whose
bytes between fields PDS4 does not constrain, is held to the layout
`global_index.py` writes -- a header line naming the label's fields, and a
comma alone between two fields, the last ending at the record delimiter -- as
the reference `check_table` held it.

`--check-only` on the labels pass, per section 3.11. The navigation
document's path and read are the navigation records' own, `document_path()`
and `read_document()`; the other three paths are `image_inputs.py`'s
statement of the rules the navigation and backplane stages write by, each of
which builds its path inline and exposes no function for it
(`navigate_image_files.py`, `cli/backplanes/backplanes.py`,
`cli/backplanes/writer.py`), a duplication older than this phase. A batch the
labels pass would fail, of other than one image, is incomplete, by the one
rule the two share. An enumeration that fails ends the report with the
traceback it ends the labels pass with. It writes no label, log or bundle
file; the file cache the roots are read through makes a temporary directory
and removes it.

Part A reconciled `docs/dev_guide/dev_guide_pds4.rst` and
`docs/user_guide/user_guide_pds4_bundle.rst` to section 3.1: both trees are
section 3.1's, in its order, and neither names a file it does not hold.

**NASA PDS `validate` over the cohort, 2026-09-14.** `validate` 4.2.0, from
the release's `validate-4.2.0-bin.tar.gz`, needs Java 17 or newer: its
changelog says so, and the system `java`, OpenJDK 1.8, is too old, so it runs
with `JAVA_HOME` naming a newer JDK, which its launcher reads first. The
command, over each bundle root:

```text
JAVA_HOME=<JDK 17 or newer> validate -R pds4.bundle -e lblx \
    [--pdf-error-dir DIR] -r REPORT -t BUNDLE_ROOT
```

`--pdf-error-dir` has to name a directory that exists; otherwise `validate`
stops with "Could not parse dir ... as a directory" and writes no report. A
run over the cohort takes about 5 s, with data-content validation on, and
takes the schemas from the labels' `schemaLocation` URLs, which needs the
network. Over the Phase 9 review's builds of the cohort bundle:

- **Plain:** 15 products, 14 passing. Two errors, both from `bundle.lblx`'s
  `TODO DOI` placeholder, one element giving two messages (`cvc-pattern-valid`
  and `cvc-type.3.1.3`). Seven `warning.integrity.reference_not_found`, the
  seven labels citing the user guide's LID, which a build without the PDF does
  not hold. Referential integrity: 15 checks, all passing, every context
  product reference registered.
- **With a stand-in guide:** 16 products, 14 passing. Six schema errors from
  the three `TODO DOI` placeholders, and one `error.validation.internal_error`:
  `validate` reads the PDF with VeraPDF, which cannot parse the 18-byte stand-in
  (section 3.6). No warnings. Referential integrity: 16 checks, all passing.

Over these two builds, then, `validate`'s only errors are the DOI
placeholders, and it raises nothing `sd_create_bundle check` does not. The
supplemental file's last line (section 2.2, row 7) is accepted by `validate`
4.2.0: no product failed content validation. A plain build's guide references are warnings, not
errors, in line with criterion 9: they are the missing-PDF signal. Zero
errors under `validate` is then blocked only by the DOIs, in both builds,
which #708 registers.

Over Part A's own builds of the cohort bundle, at `06247cd9`, `validate` gives
the same results: plain, 15 products, 2 errors (the one placeholder, twice),
7 warnings, and 15 integrity checks passing; with the stand-in guide, 16
products, 6 schema errors from the three placeholders and the VeraPDF error,
no warnings, and 16 integrity checks passing. `sd_create_bundle check` makes
10 findings over the plain build -- the placeholder, the two empty
`cassini:ISS_Specific_Attributes`, and the seven references `validate` warns
of -- and 5 over the build with the guide: the three placeholders and the two
empty blocks. Over these two builds the two agree on everything both look
at: each of `validate`'s schema errors is a placeholder the check reports
once, and each of its warnings is a reference the check reports. The empty
blocks are the check's alone, since the XSD lets
`cassini:ISS_Specific_Attributes` be empty, and the unreadable PDF is
`validate`'s alone, since the check does not read a document's content. They
did not agree in general: the product review's 55 controls found sizes,
checksums and inventory members that `validate` caught and the check did not,
which the fix round below closes. What `validate` adds after it is a FITS
data object's offset, the PDF's content and the registered context products;
what neither checks is the versions an inventory gives products outside the
bundle, and whether what a label states is right rather than allowed.

**The fix round, 2026-09-14.** Both adversarial reviews of Part A
(`65499926`) were reconciled on `rf_pds4_phase10`. The check now grades each
finding an error or a warning; compares each labeled file's `file_size` and
`md5_checksum`; resolves the inventories' members and finds two labels
declaring one LID; holds the global index tables' records to the tree and
their layout to `global_index.py`'s; reads delimited records byte for byte
and checks `field_number`; reports a schema set it cannot build as one
finding and goes on; matches a context that is not a union of paths as
`//(context)` itself; and takes the navigation document's path and read from
the navigation records' seam. Over fresh builds from the fix round's code:

| Build | Check errors | Check warnings | `validate` errors | `validate` warnings |
|---|---|---|---|---|
| Plain | 3 | 7 | 2 | 7 |
| Guide | 5 | 0 | 7 | 0 |
| Drop | 7 | 7 | 6 | 7 |
| No rings | 5 | 7 | 4 | 7 |
| Crossing | 3 | 7 | 2 | 7 |
| No target | 2 | 5 | 2 | 5 |

The check's errors are the DOI placeholders, one each where `validate` gives
two messages, and each data label's empty `cassini:ISS_Specific_Attributes`,
which the XSD allows and `validate` passes; and, in the drop and no-rings
builds, the supplemental files' sizes and checksums, stale by construction
since `phase8/build_bundle.py` rewrites those files after the labels pass: 4
in drop and 2 in no rings, one for one with `validate`'s `filesize_mismatch`
and `checksum_mismatch`. The guide build's seventh `validate` error is the
VeraPDF one. The warnings are the guide references, the same in both tools;
the no-target build skips its limb image, so it has one data product fewer.

Of the product review's 55 controls, both tools now find 34, the check alone
6 (a `[[[` marker, a header one byte short, `-999`, `unit` `rad`, an index
time 1 ms off, and an orphan index row), `validate` alone 3 (a FITS array's
offset and two context LIDs no product is registered under), and neither 12:
5 that must pass -- the two builds with every DOI filled, whose remaining
findings are the empty ISS blocks and the guide warnings, and the three
wrong but allowed values -- and 7 that neither covers: the two external
versions in the context inventory, two targets dropped from labels that keep
another, two units the vocabulary allows but the values are not in, and a
FITS axis length. With every DOI filled, a plain build now fails the check
for the empty ISS blocks alone, with the seven warnings, where `validate`
passes it with the same seven. Of the code review's 34 breaks the check
catches all but the two that make a guide reference external, which it
leaves to `validate`. The code review's surviving mutations X1, X11, X12,
X13, X14 and X16 are killed by tests, and X19 fails its three table tests by
assertion; X20 still passes the check's tests, since the check reuses the
helper it changes, and fails the summary pass's own.

**The last round, 2026-09-14.** Both re-checks found the fix round holding,
each with low findings. An XML schema error now carries its kind, the class
of the xmlschema validator that failed, and the gate compares that instead of
xmlschema's reason text: a reworded message leaves the gate passing, and a
changed kind fails it. Two tests kill X2 and X3: a reference to a bundle
whose LID begins with this one's but lacks the colon is not resolved as the
bundle's own, and a bundle label below the top of the tree is not the
bundle's. The user guide says that `--check-only` also exits 1 when the
selection hands the labels pass a batch it refuses, an empty one included.
The tests' long docstring lines are rewrapped. X20, and the reuse of
`has_geometry` in the index records check, stay by design: each rule is
stated once and the summary pass's tests pin both helpers. The records check
compares a column named for a PDS4 attribute with the first element of that
name in the product's label, which is the element the summary pass copies in
every label the templates write; its docstring says so.

#### Part B — The draft run over a real volume: deferred to #708

Deferred by the operator's ruling of 2026-09-15 (section 0): the DOIs and a
fresh navigation of a COISS volume will not come soon, and this work
finishes before them, as a prototype over the synthetic cohort. #708 carries
the run, whose steps were these: choose one COISS volume with
`--check-only`; navigate it afresh, since a navigation by an earlier version
recorded no exposure times in its `observation` block (section 3.4);
generate its backplanes and its bundle, with the DOIs registered and, for a
delivered bundle, the user-guide PDF (section 3.6); run
`sd_create_bundle check` and `validate` and record the result; check by hand
the version the context inventory gives each context product against the
PDS registry, which neither tool does (Part A); and reconcile both guides to
what it produces.

Part A closes the dev-guide output-layout part of #265, the last of its three
parts, and #66; #708 contributes to #53.

---

## 5. Acceptance criteria

The work is a prototype over the synthetic cohort (section 0), and these are
its criteria; the ones that need a real volume or the DOIs are #708's.

1. `sd_create_bundle labels` followed by `sd_create_bundle summary` over the
   synthetic cohort produces a tree matching section 3.1 exactly — asserted
   by a test that walks the tree, not by inspection.
2. Over the synthetic cohort, the NASA PDS `validate` tool reports as errors
   exactly the `TODO DOI` placeholders -- and, over a stand-in guide PDF,
   that VeraPDF cannot read it -- and as warnings the references to the user
   guide in a build without the PDF. The command and its output are recorded
   in the branch's final PR. Zero errors over a real COISS volume, its DOIs
   registered, is #708's criterion.
3. `sd_create_bundle check` reports, over a bundle built from the synthetic
   cohort, exactly the findings Part A of Phase 10 lists as known -- the
   `TODO DOI` placeholders, which the prototype keeps, each data label's
   empty `cassini:ISS_Specific_Attributes` until Part B of Phase 8, and, in a
   build without the user guide, the references to it, as warnings -- each by
   its message or, for an XML schema error, its kind, and nothing else. A
   test in the default suite asserts it over a plain build and one with a
   stand-in guide, so pytest runs it in `scripts/run-all-checks.sh` and in CI.
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
10. `--check-only` over the synthetic cohort's images reports each one's four
   inputs and exits non-zero if any selected image is incomplete.
11. Both guides describe the tree in section 3.1 and nothing else.

For the prototype, criteria 2 and 3 are the ones that matter: over the
synthetic cohort, `validate` and the check report the known findings and
nothing else. The other nine are how you get there without discovering at
the end that you cannot. `validate` accepting a real volume's bundle with
zero errors, which this plan once called the one that matters, is #708's.

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
are, and every declaration moves together. Since Phase 9 that is one edit of the
dataset's configuration entry (section 3.9), not a redesign, but it invalidates any
label generated before it.

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

**Two guides and four plan files change.** Every PR in this branch edits
`plans/PROGRAM_PLAN.md`, so each merge re-conflicts the rest. Take both
removals on a two-sided conflict.

---

## 7. Follow-ups

**No issues are filed for section 2.2's rows other than 1 and 6.** Each is assigned to a
named phase of this plan, which holds the evidence, the location and the
disposition in one place; a tracking issue whose content is "see Phase 5"
adds a close to reconcile and no reader. Row 6 was the navigation's, and was
tracked as #619, which #624 closed by recording the host's exposure times in
the `observation` block, from which Phase 7 takes every time the bundle states.
Row 1 was a decision Phase 6 could not make for want of a source product to
name, tracked as #678, which the operator ruled on and Phase 7 applies.

The one row that would have outlived this plan was the angular-unit
difference between the arrays and the tables, and the difference itself
turned out not to be a defect: section 3.8 records it as the design, decided
2026-09-09, and both backplane guides now say so where a reader will meet it.
What was a defect was the conversion recognizing only the literal `rad`,
leaving `ring_longitudinal_resolution` in radians per pixel in a table of
degrees; that is fixed, and the formatting half of #607 closes on the same
branch, with a format per unit that section 3.8 records; Phase 7 sized its
columns from the values those formats write. Nothing else there is left for someone else to
pick up.

If this branch is abandoned, section 2.2 is where the findings live. That is
a deliberate trade against five issues that would each close within the same
branch.

**Deferred, with the issue that carries them:**

- #708 — the draft run over a real COISS volume, which `validate` has to
  accept with zero errors: Part B of Phase 10, deferred by the operator's
  ruling of 2026-09-15 (section 0). It needs the DOIs registered and a fresh
  navigation of the volume, and carries the by-hand registry check of the
  context inventory's versions.
- #705 — the spelling gate on its way to `main` reads the local copies of
  the PDS4 schemas the tests keep (Phase 10, Part A), whose misspellings are
  NASA's; its skip list takes `*.xsd` and `*.sch`, in whichever of its two
  changes lands second.
- #595, #596, #597, #598, #599 — the backplanes user guides: one shared
  LaTeX template and one guide per instrument. #596 is what section 3.6's
  acceptance criterion 9 turns on for this bundle; the other three wait on
  their instrument's half of #53. #595 carries an open decision on where the
  LaTeX sources and the built PDFs live relative to the template directory.
- #600 — what the bundle says about images that did not navigate. Replaces
  section 3.11 when it is decided; nothing before Phase 10 depends on it.
- #601 — masked backplane values are `-999` as of 2026-09-09, applied on
  this branch (section 3.13), so one comparison masks every plane, and Phase 4
  declares it on every float array of the data label. Phase 7 decided the
  index tables' missing value the same way: a cell whose image has no
  statistic for its plane holds `-999` in the column's own format, and every
  statistic field declares that spelling as its `missing_constant`. `validate`
  accepts such a cell as a real, and the declaration tells a reader what the
  value means (section 3.13). The operator accepted it on 2026-09-14, since no
  column can hold `-999` as a real value (section 3.13 argues each), and #601
  is closed by hand when Phase 7 merges. The ring
  half of the original finding turned
  out to duplicate #251, which is the `xfail`-pinned record that ring-won
  pixels get no `BODY_ID_MAP` entry; the sentinel makes that gap harmless
  for consumers without closing it.
- #79 — scrape the PDS4 context products so `target_lids` is maintained
  rather than hand-written; Phase 8 filled it by hand from the PDS registry
  (section 3.7).
- #618 — choose each planet's ring target from configuration. The ring target
  the backplane metadata names, which keys the rings in the targets table, is
  `backplanes_rings.ring_target`'s, and moving its rule for Saturn into
  configuration leaves the table's key as it is.
- #684 — the Cassini ISS label facts the navigation document's `observation`
  block is to record, from which Part B of Phase 8 fills
  `cassini:ISS_Specific_Attributes` (section 3.7).
- #677 — which SPICE kernels the bundle's metakernel lists, a
  navigation question. Phase 6 shipped the `spice_kernels` collection with a
  metakernel that lists none and says so.
- #678 — what a data label names as its source product, ruled on 2026-09-14:
  the calibrated PDS3 image, as a `Source_Product_External`, until a PDS4
  bundle of calibrated images exists, when the label switches to
  `Source_Product_Internal` (section 3.13). Phase 7 applies the ruling, and
  #678 is closed by hand when Phase 7 merges.
- #687 — cite each image's calibrated PDS4 product through
  `Source_Product_Internal`, by its LIDVID, once a PDS4 bundle of calibrated
  images is registered, and drop the external citation (section 3.13).
- #530 — the stats corpus's own Cassini clock seconds, which do not follow
  from their epochs. Phase 2 builds the epoch-first constructor that makes
  the defect unrepeatable and uses it for every cohort document. Routing the
  existing four through it renames four fixtures, moves the `filtered`
  variant's image-number bounds and regenerates both goldens; that is #530's
  own work and belongs in a PR about the statistics fixtures, not on a PDS4
  branch. It can land before, after or independently of this plan.
- #67 — cloud-aware bundle generation. Phase 4 added a second
  `get_local_path()`/`shutil` copy, for the FITS; both copies, and the
  label's `FILE_*` functions that read `BACKPLANE_PATH` as a local file, are
  #67's work.
- #424 — remove `sd_create_bundle_cloud_tasks`.
- #53's generalization half — Voyager, Galileo and New Horizons template
  trees and `pds4_*` hooks, against this plan's validated Cassini tree as
  the reference.
- #232 — whether the label's geometry values are *correct*, as opposed to
  present and schema-valid. This plan makes a bundle that validates; it does
  not check a single number against independent truth.
- #30 — the backplane label design as a whole, of which Phase 4 implemented
  the part the file forces: the data objects and their display settings.

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
  than no test, so check that each new one fails when the behavior it
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

The draft run over a real volume, which needs the operator's cohort choice
and, for a delivered bundle, the user-guide PDF (section 3.6), is #708's
(section 0), and blocks nothing that remains. One question
is open and is not the operator's to answer alone: whether this information
model build's dictionaries are registered (section 3.9), which the
Engineering Node is being asked. It bears on acceptance criterion 6 and on
nothing before Phase 10. The angular-unit question that section 3.8 once
held open was settled 2026-09-09 in favor of degrees in the tables, which is
what the products did for every angular column but one.

The branch merges to `main` as a merge commit, not a squash, so the
individually reviewed phase PRs survive in the history.
