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
`main` at `bc103ffb`.

No phase has run. One change has landed ahead of them, because it is
mechanical, self-contained and must precede any label generated against a
schema: the rings dictionary bump recorded in section 3.9.

This plan is the "finish and validate the Cassini path" half of #53, which
`plans/ENGINEERING_PLAN.md` (Track D, "PDS4 output bundles") lists as the
prerequisite for the generalization half. It does not generalize to the
other three instruments; that work begins when this plan's acceptance
criteria hold.

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

- **The other three instruments.** Voyager and Galileo have partial `pds4_*`
  hooks and New Horizons has none; `DataSetPDS4` raises throughout. Their
  template trees and hooks are the second half of #53 and are mechanical
  once a validated reference tree exists. This plan produces that reference.
- **PDS4 input** (#34). Unrelated to output bundles despite the shared
  acronym; no such archive exists to read.
- **The backplane set and HDU content decisions** (#55, #57, #54, #77).
  This plan describes whatever the generator writes; it does not decide what
  the generator should write. Section 3.8 is the closest it comes: it records
  why the arrays and the tables carry different angular units, because the
  labels have to state both and a later reader will otherwise take one of
  them for a mistake.
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
| 18 | Every `template.write` discards its `(errors, warnings)` return. An unresolved variable is reported through that return, not raised, so a bad label is written and the run reports success. | `bundle_data.py:121,138`, `collections.py:83,111,287,298` | #265 |
| 19 | Nothing validates. No `validate` invocation, no schema check in CI, no `xmlschema` or `lxml` dependency in `pyproject.toml`. | — | #53 list |

None of these gets its own tracking issue. Every row is fixed by a named
phase of this plan, which carries the evidence and the disposition together;
an issue whose content is "see Phase 5" has no reader, and five more entries
in Track D's index means five more closes to reconcile on a branch where
every PR already re-conflicts `plans/PROGRAM_PLAN.md`. Defect 1 additionally
has an `xfail` and belongs to #265 and #69. The rows that *would* have
outlived this plan -- the ones true of shipped products whether or not a
bundle is ever built -- were the units pair, and section 3.8 records that as
settled design rather than a defect.

### 2.3 What this implies about order

Defect 18 hides the rest. Until `template.write` failures are surfaced,
every subsequent phase is working blind: a template edit that mistypes a
variable produces a label with an embedded `[[[ ]]]` error marker and a run
that says it succeeded. It is Phase 1 for that reason and no other.

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
TDB seconds past J2000. `spindoctor/cli/stats/classify.py` already converts
with `julian.iso_from_tai(julian.tai_from_tdb(...))`. That conversion moves
to one shared function — `spindoctor/support/` is the right home, since two
CLI packages now need it — and both callers use it. #519 asks for exactly
this and says so.

An image whose navigation never reached a solution has no `times` block.
Section 3.10 says what happens to it, and the answer is that it never
reaches a label, so the empty string stops being reachable rather than being
made deliberate.

The collection and bundle labels need the cohort's earliest start and latest
stop. The summary pass already reads every supplemental file to build the
global index; it takes the min and max there, in the same pass, and hands
them to both labels. `collections.py:80-81` stops being a TODO.

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
therefore the global index tables -- are degrees, converted at
`backplanes_bodies.py:184` and `backplanes_rings.py:95` when the configured
unit is `rad`.

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
- The `Field_Delimited` blocks Phase 7 generates for the global index take
  their `unit` from the same config entry the column was built from, mapped
  through the same `rad` to `deg` rule the statistics use. An angular column
  is labelled `deg`.
- So one bundle carries `unit="rad"` on an array and `unit="deg"` on the
  table summarizing it, deliberately. Both labels are correct about the file
  they describe, which is the only thing a label is required to be correct
  about.

Two places must say so in prose rather than leaving it to be rediscovered:
the backplanes user guide (section 3.6's operator deliverable), and a
comment at the conversion site in the statistics path, which currently reads
as an incidental unit fix rather than as a deliberate difference between two
audiences. Phase 8 adds both.

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

The integrity pass (#66) is the right answer and this plan includes a
minimal form of it: a `--check-only` flag on the labels pass that reports,
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

Most of what is needed already exists and must not be rebuilt.
`tests/spindoctor/cli/stats/results_tree_documents/` is a package that builds
navigation documents **through the production writer** --
`build_metadata_from_result` and `build_timing_section` from
`navigate_image_files` -- over hand-constructed `NavResult` objects. It
touches no SPICE and no holdings, it covers three instruments, three
outcomes, BOTSIM pairs and gated features, and it derives every spacecraft
clock reading from the epoch beside it rather than inventing one. It is
regenerated with:

```bash
PYTHONPATH=src python -m tests.spindoctor.cli.stats.results_tree_documents
```

That package is the nav half of the cohort, but not by extending the set it
already holds. The two sets are selected for different things. The eight
documents in `results_tree_documents()` are chosen for what they make the
*statistics report* exercise -- three outcomes, four feature sources, a
BOTSIM pair, a suspect offset -- and the package docstring states that every
document earns its place there. An image added for bundle sharding earns
nothing in that report; it adds rows to a fixture whose whole value is that
each row is deliberate.

The stored golden output is cheap to regenerate and no one has signed it
off, so the cost of touching it is not the argument. The argument is that a
fixture selected for two unrelated criteria stops being legible for either.

So the package gains a **second document set**, `cohort_documents()`, built
from the same `shared.py` primitives and written to a cohort root rather
than into `RESULTS_TREE`. `results_tree_documents()` and the stats fixture
tree are untouched.

**The cohort is never checked in.** It is built at test time into `tmp_path`
and torn down with it; no cohort bytes live under `tests/`, and the
`python -m tests.mini_nav_results cohort <outdir>` form writes wherever the
operator points it. The builders are the artifact, not their output. That is
what keeps a fifth instrument's cohort from costing the repository anything
-- adding one is a module beside `cassini.py`, and the FITS, the PNGs and
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
- **`ImageFile`s carrying `index_file_row`**, because thirty of the
  `cassini:*` template variables are read from the PDS3 index row and Phase 8
  puts them in the label. No index file is parsed; the row is a dict.
- **Coverage the bundle stage cares about**, which the stats corpus has no
  reason to carry: two images whose numbers shard into *different*
  `1234xxxxxx/123456xxxx` directories, one image with ring backplanes and one
  without, and one image whose navigation did not succeed.

Two rules bind the additions.

**Epochs first, everything else derived.** #530 is the open record of what
happens otherwise: four Cassini documents in the existing set carry clock
seconds taken from the image number rather than converted from the epoch
beside them. The response here is not a test that exempts those four. It is
a `shared.py` constructor that takes an epoch and returns the clock triple,
so a document built through it cannot carry an invented one. The cohort set
is built entirely through it. Routing the existing four through it as well
is #530's own work -- a coordinated change to four documents, four
filenames, the `filtered` variant's image-number bounds and both goldens,
which belongs in a PR about the statistics fixtures rather than on a PDS4
branch.

**Production writers write the fixture.** `writer.py:write_fits` writes the
FITS and its metadata sidecar for real code, so it writes them for the
fixture too, called with synthetic arrays. A fixture built by a second,
parallel writer is a fixture that stops describing the product the moment
the real writer changes.

The package moves to `tests/mini_nav_results/`, beside `tests/shims/` and
`tests/cmatrix_helpers.py`, because a package two suites import should not
live inside one of them, and takes the shorter name on the way: it is a
miniature of what a navigation run leaves behind, which is what every
consumer of it wants it for. The move and rename are import-only -- the
documents it emits are unchanged, so the stats suite's stored golden output
does not move.

The backplane products it grows are not, strictly, navigation results: they
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

The `cohort` form is what an operator points `sd_create_bundle` and the Java
`validate` tool at without waiting for a navigation run, and what Phase 10's
schema gate runs over.

---

## 4. Implementation phases

Each phase is one pull request onto `rf_pds4_draft_bundle`, with tests, and
with `docs/` and the plan files reconciled as the standing convention
requires.

### Phase 1 — Surface label-write failures

Capture `(errors, warnings)` from all six `template.write` call sites.
Warnings log at warning level; `errors > 0` fails that product — the label
is not left on disk, the image is counted as failed, and the run's exit
status reflects it. `pdstemplate` offers `mode='repair'`, which saves only
when there are no errors; prefer it to a post-hoc unlink.

The two `main_*` functions gain a failed-product count and exit non-zero
when it is not zero.

Tests: `test_undefined_template_variable_error_is_swallowed`
(`test_bundle_data.py:283`) currently asserts the silent behavior as
characterization; it is inverted here rather than added to, and renamed.
Plus: a clean template still writes, and a run with one failed product
exits non-zero.

Closes the first half of #265.

### Phase 2 — The synthetic cohort

Build the fixture cohort section 3.12 describes, so that every phase after
this one can assert against the shipped templates without holdings, without
SPICE and without navigating anything.

Three commits:

1. **Move and rename** `tests/spindoctor/cli/stats/results_tree_documents/`
   to `tests/mini_nav_results/`. Three things follow it: its one importer,
   `tests/spindoctor/cli/stats/test_results_tree_documents.py` (whose own
   name still describes what it tests and stays); the regeneration command
   in the package docstring; and the docstring itself, which gains a line
   naming the roots the package writes. The `results_tree_documents()`
   function keeps its name -- it returns the documents of the results tree,
   which is still what it does. Import-only: the emitted documents are
   byte-identical, so the stats golden report is untouched. Verify that by
   regenerating the tree and confirming `git diff` over
   `tests/spindoctor/cli/stats/data/results_tree` is empty.

2. **Add a second document set**, `cohort_documents()`, over the same
   `shared.py` builders -- not more entries in `results_tree_documents()`.
   The stats fixture tree and both goldens are untouched, and a test asserts
   that: regenerate, and `git diff` over
   `tests/spindoctor/cli/stats/data/` is empty. The cohort set holds the
   images the bundle stage needs and the report has no use for -- two
   sharding into different `1234xxxxxx/123456xxxx` pairs, one with ring
   backplanes, one that did not navigate -- plus what bundle generation
   reads: a `backplanes` module calling `writer.py:write_fits` with
   synthetic 16x16 arrays for a real FITS and its metadata sidecar, a
   summary PNG, and `index_file_row` dicts carrying the COISS index columns
   `pds4_template_variables` reads. Every clock triple built through the
   epoch-first constructor described in section 3.12, which is added here
   and which the existing four documents do not yet use.

3. **The cohort builder**, as `tests/mini_nav_results/cohort.py`: it
   assembles nav root, backplane root and `ImageFiles` list into one object
   the tests take as a session-scoped fixture, written into a temporary
   directory and torn down with it. Nothing it produces is committed.
   `__main__.py` becomes the two-argument entry point of section 3.12 -- set
   name, then output directory, both required for both sets. The statistics
   regeneration command changes shape with it, so update the one place it is
   written down (the package docstring) and any test docstring that quotes
   it.

Then repoint the PDS4 suite. `tests/spindoctor/cli/pds4/conftest.py` keeps
`FakePds4DataSet` and its tiny templates — they exist to test the *plumbing*
with variables the tests control, and that is still worth having. What it
gains is a second environment built on the cohort, using the real
`DataSetPDS3CassiniISSSaturn` hooks and the real shipped templates, which is
what every phase after this one asserts against.

Tests: the cohort writes a FITS `astropy` reopens with the expected HDU
names; the two Cassini images land in different shard directories; each
image's backplane metadata names the same backplanes its FITS carries; the
non-success image is present and is skipped by the bundle stage; every
cohort clock triple spans the epochs beside it; and the stats fixture tree
regenerates byte-identical. Plus one guard that the cohort stays ephemeral:
`git status` is clean after the suite runs, so a cohort written into the
working tree by mistake is reported rather than committed.

No issue closes here. This phase exists because the nine after it are
untestable without it.

### Phase 3 — Epochs

Shared ET-to-ISO conversion in `spindoctor/support/`, used by
`classify.py` and by `pds4_template_variables`. `START_DATE_TIME`,
`STOP_DATE_TIME` and `IMAGE_MID_TIME` read `navigation_result.times`.
Collection and bundle date ranges computed in the summary pass from the
supplemental files.

Tests: a document with `times` yields the expected ISO strings; the
collection range over three supplemental files is the min and max.

Closes #519.

### Phase 4 — The FITS in the bundle, with its data objects

Copy the FITS into `data/`. Build the per-HDU descriptor list from the
copied file. `data.lblx` grows the `$FOR` block described in section 3.3,
with `local_identifier` `image` on the first array.

Tests: a two-HDU fixture FITS produces two `Array_2D` blocks with the
offsets `fileinfo()` reports; a frame with no ring planes produces no ring
arrays; the display settings' referenced identifier is defined in the label.

Closes #69; contributes to #30.

### Phase 5 — Inventories that conform

`.csv`, headerless, LF, correct record counts, for the two generated
inventories and the three copied ones. `.tab` removed everywhere including
the global index tables.

Tests: the generated inventory has no header, ends every line with `\n`, and
`FILE_RECORDS` equals the member count.

Closes the inventory half of #265 and the new format issue.

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

Also carries section 3.8 into prose: the user-guide paragraph on why arrays
and tables use different angular units, and the comment at the conversion
site that currently reads as an incidental fix.

Tests: an image with two bodies emits two `Target_Identification` blocks; an
image with rings emits the ring geometry block and one without emits none.

Closes #73, #75, #47; contributes to #53's template list. #79 stays open.

### Phase 9 — Parameterize the bundle name and version

`bundle_version` in `config_950_pds4.yaml`; every hardcoded bundle name and
`version_id` in every template becomes a variable.

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

Closes the second half of #265 and #66; contributes to #53.

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

**`sd_create_bundle` crashes inelegantly on a missing metadata file**, noted
on #519 and true of the backplane metadata read at `bundle_data.py:73` as
well. `--check-only` makes it avoidable rather than fixing it; the exception
policy merged as #580 is the pattern if it is fixed here instead.

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
difference between the arrays and the tables, and it turned out not to be a
defect: section 3.8 records it as the design, decided 2026-09-09, and Phase
8 writes it down where a reader will meet it. Nothing there is left for
someone else to pick up.

If this branch is abandoned, section 2.2 is where the findings live. That is
a deliberate trade against five issues that would each close within the same
branch.

**Deferred, with the issue that carries them:**

- #595, #596, #597, #598, #599 — the backplanes user guides: one shared
  LaTeX template and one guide per instrument. #596 is what section 3.6's
  acceptance criterion 9 turns on for this bundle; the other three wait on
  their instrument's half of #53. #595 carries an open decision on where the
  LaTeX sources and the built PDFs live relative to the template directory.
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

Cut `rf_pds4_draft_bundle` from `main`. One PR per phase onto it. For each:
implement, run `./scripts/run-all-checks.sh`, run the unit suite at `-n 4`
with the BLAS and OpenMP thread counts pinned to 1, open the PR, wait for
CodeRabbit to settle and reply on every comment with a disposition, then
merge. Update `docs/` and the four plan files in the same PR as the change
they describe, not afterwards.

Phase 10's draft run needs the operator's cohort choice, and the user-guide
PDF if the draft is to be delivered rather than reviewed internally
(sections 3.6 and 10 of Phase 10). Neither blocks Phases 1-9. One question
is open and is not the operator's to answer alone: whether this information
model build's dictionaries are registered (section 3.9), which the
Engineering Node is being asked. It bears on acceptance criterion 6 and on
nothing before Phase 10. The angular-unit question that section 3.8 once
held open was settled 2026-09-09 in favour of what the products already do.

The branch merges to `main` as a merge commit, not a squash, so the
individually reviewed phase PRs survive in the history.
