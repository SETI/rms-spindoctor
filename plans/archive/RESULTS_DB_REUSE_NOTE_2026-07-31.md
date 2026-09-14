# Reusing the Stats Database as a Shared Results Index, and Backend Selection

Status: pre-decision analysis, 2026-07-31. No code has been changed. Companion to plans/CK_KERNEL_DESIGN_NOTE.md.

Scope, as set by the operator: the JSON metadata files remain the authoritative record and are not being replaced. The database is ingested from them (once per nav batch) and downstream programs -- bundle generation, backplanes, reprojection, CK generation, stats -- then consult the database instead of reading per-image JSON. The cost being eliminated is not directory listing; it is the one-object-read-per-image pattern that every program using `yield_image_files_from_arguments` plus a per-image metadata read exhibits, which in a cloud environment is a paid round trip per image. A further hard requirement: **the current scanning/reading paths stay and remain the default** -- every program must operate with no database at all; the database is a strictly optional alternative the user selects. Second question: can the backend be selectable (SQLite vs MySQL/PostgreSQL)?

## 1. Who touches the results tree today, and what they actually need

| Program | Access pattern | Fields actually consumed |
|---|---|---|
| `sd_stats_ingest` | full `rglob('*_metadata.json')` walk per root | everything it keeps (see section 2) |
| `ResultsFilter` (used by every PDS3 CLI via `--has-offset-file` etc.) | one `FCPath.walk()` per selected volume, collecting metadata + summary-PNG paths; error filters then read files in batches of 64 | file presence (metadata, PNG); `status`, `status_error` |
| backplanes (`sd_backplanes`) | direct read of `<root>/<stub>_metadata.json` per image, no walk | `status`, `status_error`, top-level `offset` (full precision) |
| reproj/mosaic (`cli/reproj/offsets.py`) | direct read per image, no walk; highest-volume reader in the system | `status`, top-level `offset` |
| bundle gen (`cli/pds4/bundle_data.py`) | direct read per image, no walk | `status` gate, then **the entire JSON document** (serialized verbatim into the PDS4 supplemental product) |
| `sd_consolidate_metadata` | direct `read_bytes` per image | raw file bytes (metadata + PNG) |
| CK generation (planned, `sd_create_ck`) | would enumerate successful images | `status`, offset/C-matrix, sigma, confidence, rotation, times/SCLK |
| `util/cohort_curation/triage_stage_b.py` | one `rglob` **per image name** over the whole root -- the worst pattern in the repo | `status`, `status_reason`, `offset_px`, `image_classifier` |

The pattern that matters is common to all of them: whether a program walks the tree or addresses each file by stub, it ends up reading one JSON object per selected image -- order 400k reads for a Cassini-scale run, each a paid round trip on `gs://`. The redeeming fact is that the fields consumed are narrow -- `status`, `status_error`, `offset`, and quality numbers -- for every consumer **except** bundle generation (whole document) and consolidation (raw bytes). So one ingest pass (which pays the N reads once) can feed every narrow-field consumer from then on.

## 2. The stats DB as it stands: right shape, five real gaps

The DB (`cli/stats/schema.py`) already has one row per image with `status`, `status_reason`, `offset_dv/du`, `sigma_dv/du`, `confidence`, `confidence_rank`, `image_et`, camera/instrument, timing, and child tables for per-technique results and feature sources. As an index it is the right skeleton. The gaps:

1. **Keying.** `images.image_name` (source basename) is the sole primary key. Every consumer addresses results by `results_path_stub` (`<volume>/<filespec>`), which is in neither the metadata JSON nor the DB -- only implicitly inside `source_file`. Two images with the same basename in different volumes/roots silently overwrite each other. The index must key on the stub (or stub + root), with the volume/dataset extractable.
2. **Precision and provenance of the offset.** The DB stores `navigation_result.offset_px`, rounded to 4 decimals; downstream consumers are defined to use the full-precision top-level `offset`. 1e-4 px is physically negligible, but if the DB becomes the source of record the ingest should store the top-level value so there is exactly one authoritative number. Same for `status_error`, which ingest currently merges into `status_reason`, conflating two vocabularies (`ResultsFilter`'s SPICE-error filter needs `status_error == 'missing_spice_data'` verbatim).
3. **Dropped fields.** Not stored today: image-level `rotation_deg`/`sigma_rotation_deg` (Galileo twist is invisible), fused `covariance_px2`, `sigma_along_unobservable_px`, provenance (kernels, versions, config overrides), classifier details beyond class/noise, full feature inventory. CK generation and the planned C-matrix/SCLK fields would need columns added. Bundle generation needs the *whole document*, which no column set will satisfy.
4. **Freshness.** Ingest is full-rescan only: no mtime/size tracking, no `--since`, no record of which roots were ingested when. One giant transaction covers the whole run (crash = lose everything); the DB is documented as disposable. As a private stats cache that is fine; as a shared index it means every consumer inherits an unquantified staleness risk, and "absent from the DB" is ambiguous between "never navigated" and "not yet ingested".
5. **Artifacts other than metadata.** Summary-PNG and backplane-product existence (the `--has-png-file` filters) are not modeled.

Also worth fixing regardless of this decision: ingest's cloud-root read path uses `FCPath.get_local_path()` + `read_text()`, which does not download -- on a `gs://` root every file lands in the per-file error counter. And the stats CLIs bypass the config system (bare `--db nav_stats.sqlite3` in CWD, `print()` instead of loggers), which must change before anything pipeline-critical depends on them.

## 3. Reuse designs

### A. Rebuildable index, JSON stays authoritative (RECOMMENDED -- this is the operator's stated model)

The tree of `_metadata.json` files remains the record; the DB is a derived index that consumers *may* use. Work: fix the keying (stub-based), store the top-level offset and `status_error` verbatim, add the missing quality columns (rotation, fused covariance) plus PNG/backplane presence, make ingest incremental (per-file mtime/size, per-root ingest bookkeeping, batched retrieval, chunked transactions), add a config key (`environment.results_db`), and give consumers an opt-in flag (`--results-db`). The existing scan/read code paths are not replaced: with no flag, every program behaves exactly as today; the flag merely swaps the lookup implementation. When given, `ResultsFilter` becomes a single SQL query instead of a per-volume walk; triage becomes one query; backplanes/reproj/CK gen read one row instead of one JSON each.

*Pros:* no new source of truth; DB corruption or staleness costs nothing (rebuild); programs gain the option independently and lose nothing when the DB is absent -- operation never depends on it; SQLite remains sufficient because the writer stays single-process.
*Problems:* dual representations can drift between ingests -- consumers must treat the DB as a snapshot and the freshness policy must be explicit (e.g. refuse if the newest ingest predates the newest nav run, or re-verify hits against the tree); absence filters only work over roots the DB has fully ingested, so ingest bookkeeping is load-bearing, not cosmetic; bundle gen and consolidation still pay per-image reads (they need documents/bytes -- design B removes bundle gen's share); someone must run ingest -- either operators remember after each nav batch, or the batch driver invokes ingest as its final step.

### B. Index plus full-document column

As A, but ingest also stores the raw metadata JSON per image (SQLite TEXT / PostgreSQL `jsonb`). Bundle generation then reads its supplemental payload from the DB too, and ad-hoc queries can reach dropped fields via `json_each`/`jsonb` operators without schema churn.

*Pros:* every metadata consumer except consolidate-PNG stops reading per-image JSON -- in particular this is the **only** way bundle generation (which needs the whole document for the PDS4 supplemental product) gets off the N-reads pattern; new analysis questions need no schema migration.
*Problems:* DB size grows to roughly the tree size (~10-40 KB/image; order 10 GB for a Cassini-scale archive -- still fine for either backend, but it weakens the ship-the-file-to-workers distribution trick in section 4 unless the document column lives in a separate optional database); JSON functions differ per backend, complicating portability. A worthwhile extension of A once A is proven -- not the first step, unless bundle-gen cloud cost is the immediate pain.

### C. Pipeline writes the DB directly (OUT OF SCOPE -- recorded only for completeness)

The operator has ruled this out: the JSON files stay, and the DB is populated by ingest, not by the pipeline. For the record, the reason it would be a poor move anyway: nav cloud workers are many spawned processes on many machines, so direct DB writes would *require* a networked server backend, per-worker credentials, and DB availability as a new failure mode for navigation itself.

### D. Status quo plus targeted fixes

Skip the shared index; just fix triage's per-name rglob and ResultsFilter's per-volume walk with a per-run in-memory scan cache.

*Pros:* minimal work.
*Problems:* leaves N-per-image remote reads in backplanes/reproj/bundle gen, leaves CK gen to invent its own enumeration, and leaves the 400k-image future unsolved. Falls short of the stated goal.

## 4. Selectable backend (SQLite / MySQL / PostgreSQL)

**Feasible, with a bounded but real porting cost.** Current code is raw `sqlite3` with `sqlite3.Connection` in public signatures, and several SQLite-only constructs: a Python UDF (`image_number`) used inside WHERE clauses, `TOTAL()`, `executescript`, `PRAGMA`s, and documented user idioms (`AVG(status='success')`, two-argument `MAX`, `json_each`). No ORM or non-SQLite driver is currently a dependency.

Options:

1. **(CHOSEN) SQLAlchemy Core (not ORM) with a connection-URL config key** -- `results_db: sqlite:///path.db | postgresql+psycopg://...`, the driver as an optional extra (`spindoctor[postgres]`). Core gives dialect-neutral DDL/DML, parameter-style handling, and pooling, without imposing an object model on what is fundamentally three flat tables. The UDF problem disappears by materializing `image_number` as an ingested column (worth doing even in SQLite-only life); `TOTAL()` becomes `COALESCE(SUM(...), 0)`; `executescript` becomes per-statement DDL. This is the standard answer and the recommended one.
2. **Hand-rolled DB-API adapter** (per-backend SQL templates, `?` vs `%s` param styles). Saves the dependency but recreates a worse SQLAlchemy over time; sustained maintenance cost; not recommended.
3. **Stay SQLite-only, hardened** (WAL, `busy_timeout`, chunked transactions). Superseded by the choice of option 1, but its hardening measures still apply whenever the SQLite backend is selected -- especially under the parallel ingest described in section 5.

A wrinkle the cloud goal adds: the *consumers* themselves run as distributed workers (`sd_mosaic_cloud_tasks`, `sd_backplanes_cloud_tasks`, ...), so the index must be reachable from every worker. Two viable distribution models:

- **Ship the SQLite file.** Publish the ingested file to the results bucket; each worker downloads it once at startup via the existing `FCPath` machinery. One object fetch of order 100 MB replaces hundreds of thousands of per-image fetches; no service to run, no credentials, no availability story; workers get an immutable snapshot, which matches the ingest-once model exactly. Costs: a per-worker download (and memory/disk for it), and staleness is frozen at download time.
- **One remote PostgreSQL instance** (e.g. Cloud SQL) that all workers query directly. No per-worker download, always reflects the latest ingest, and design B's document column has no size penalty for workers. Costs: a managed instance to run and pay for; network/auth plumbing into every worker environment; connection-count limits when hundreds of workers connect (pooling or a proxy sidecar); the DB becomes an availability dependency of every batch run; and it pulls the backend-abstraction work forward from "someday" to "prerequisite", since ingest must then be able to write to that backend too.

Neither model is wrong; the choice is operational. Ship-the-file is the smaller step and fits the snapshot semantics of ingest-once; the server is the better fit if the index is queried interactively from many places, updated between worker starts, or grows past convenient shipping.

Costs that come with multi-backend support regardless of mechanism: a real migration story replaces "delete the file and re-ingest" (though design A keeps rebuild-from-tree as the migration escape hatch); CI must run the test matrix against PostgreSQL (service container) alongside SQLite; type mapping discipline (booleans, REAL vs DOUBLE PRECISION, TEXT collation); the documented direct-SQL examples in the user guide need per-backend variants or neutral rewrites. PostgreSQL is the only server backend (decided) -- a good fit here: `jsonb` serves design B if it ever arrives, and transactional DDL keeps schema setup atomic.

## 5. Plan (decisions taken 2026-07-31)

The operator has decided: reuse design **A**; backend mechanism **option 1** (SQLAlchemy Core with a URL-selectable backend), landed up front rather than deferred, with **remote PostgreSQL supported from the beginning** (PostgreSQL is the only server backend -- no MySQL); ingest gets a **cloud-tasks variant**; ingest is **always manual**, never invoked automatically by batch drivers; consumers **trust the DB contents** as given (no staleness machinery); **no schema migrations**, but the DB carries a **schema version number** that programs check and error on when it does not match the version they were built for (the remedy is delete and re-ingest); and ingest targets the **current metadata schema only** -- the existing legacy-era results JSONs are explicitly not a concern.

1. **Make the DB a trustworthy index**, built on SQLAlchemy Core from the start: stub-based keying with volume/dataset columns; store top-level `offset` and `status_error` verbatim; add rotation/covariance/PNG-presence columns and (with the CK work) SCLK/start/stop; incremental batched ingest with per-root bookkeeping; fix the FCPath download bug; `image_number` as a column; a `schema_version` value stamped at creation and checked by every opener (error with "delete and re-ingest" on mismatch -- this formalizes the existing column-comparison check and replaces migrations entirely); connection-URL config key and logger integration for the stats CLIs. Adoption order (decided): the stats reporting itself moves onto the new schema/abstraction first, then backplanes, then CK generation -- each behind `--results-db`, with the existing scan/read path untouched as the default. `ResultsFilter` and `triage_stage_b` remain the biggest bang-for-buck conversions and can follow after those three.
2. **`sd_stats_ingest_cloud_tasks`.** Queue tasks partition the metadata set (by volume or file list); each worker task reads its share of JSONs and upserts rows. Concurrency model: when all workers run on **one machine**, they write the single local SQLite file directly -- WAL mode, `busy_timeout`, and short per-batch transactions make multiple local writer processes safe, and since there is only ever one file, **no merge step exists**. When workers span **machines**, they connect as ordinary clients to the remote PostgreSQL instance (a shared SQLite file across machines is not an option). Implementation cautions: schema creation must be race-safe (one-time init before task fan-out, not per-task DDL); today's whole-run single transaction must become per-batch transactions, or parallel local writers will just serialize behind one lock; the per-root ingest bookkeeping must tolerate concurrent completion records; and the DELETE+INSERT upsert must stay per-image-atomic so two tasks never interleave halves of one image's rows.
3. **Design B's document column** later, when bundle generation's per-image reads become the pain point -- it is the only consumer a column schema cannot serve.

For cloud *consumers*, the URL mechanism makes the distribution choice per-run rather than architectural: point `--results-db` at a downloaded SQLite snapshot or at the remote server, whichever the run warrants (trade-offs in section 4).

## 6. Decision record

All design decisions are settled; the note is ready to drive implementation issues.

- Reuse design **A**: JSON files authoritative, DB is an opt-in ingested index; the current scan/read paths stay as the default and operation never requires a database.
- Backends: SQLite (local) and **PostgreSQL only** as the server backend (no MySQL), via SQLAlchemy Core with a connection-URL config key, landed up front; remote use supported from the beginning.
- Ingest: manual only (never auto-invoked); cloud-tasks variant with same-machine-SQLite / remote-PostgreSQL-across-machines concurrency; current metadata schema only (legacy-era results JSONs out of scope).
- Consumers: trust DB contents (no staleness machinery); adoption order stats, then backplanes, then CK generation; `ResultsFilter`/triage after those.
- Schema evolution: **no migrations**; a stored schema version number, checked by every program at open, erroring on mismatch -- delete and re-ingest is the only upgrade path.
- Bundle generation: stays on direct JSON reads; a design-B document column is a possible future addition, not planned soon.
