# Archived critiques

Point-in-time review documents, dated in each filename, kept for reference. Each
is a snapshot of what its author found on its date, frozen as written: none is
maintained, and none carries a running status section. A critique is archived
once the work it reviewed has landed, which is why the parent `critiques/`
directory is empty whenever no review is open.

Internal file paths and line numbers reflect the tree as reviewed and have
drifted since. The 2026-06 and 2026-07-08 reviews all read the code merged as
commit `77a90f4`, and their paths predate the SpinDoctor rename, so `src/nav/`
is now `src/spindoctor/`, `tests/nav/` is `tests/spindoctor/`, and the CLI entry
points are `sd_*`.

For what is actually open, read the live plans and the GitHub issues, not these
records: `plans/PROGRAM_PLAN.md` is the plan of record, and every finding below
that remains open is tracked there or in an issue.

- `CODE_CRITIQUE_2026-06-10.md` — 190-finding source review with inline remediation
  status; the majority of findings are fixed, and every still-open
  Critical/High finding is tracked by a GitHub issue (see `ISSUE_CROSSREF_2026-06-19.md`
  and issues #132-#139).
- `TESTS_CRITIQUE_2026-06-10.md` — test-suite review. Much of what it reports has
  since been closed: the star-conflict logic gained direct coverage (#216), and the
  backplanes and PDS4 backends gained suites (#257). Regression baselines beyond the
  single real-image frame remain open as #174.
- `SCIENTIST_REVIEW_2026-06-19.md` / `SCIENTIST_REVIEW_CRITICAL_2026-06-19.md` — paired scientific
  reviews; every finding of the critical review is mapped to a workstream in
  `plans/VALIDATION_AND_CALIBRATION_PLAN.md` (see its traceability matrix).
- `ISSUE_CROSSREF_2026-06-19.md` — mapping of open GitHub issues to `CODE_CRITIQUE_2026-06-10.md`
  finding IDs; its "untracked candidates" were subsequently filed as issues
  #132-#139.
- `SIM_REWRITE_FINDINGS_2026-06-19.md` — simulator findings; each is either addressed in
  the merged simulator or tracked as a GitHub issue (#84, #194-#198).
- `PROJECT_STATE_REVIEW_2026-07-08.md` — meta-review of the whole software state:
  it audits the accuracy of the five reviews above, the completeness of the plan
  set as it stood, and the documentation, against freshly run quality gates. Its
  six recommendations were largely executed by PR #200 the same day it was written
  and the rest have since been folded into the live plans; its enduring value is
  the verification record (37 of 38 sampled code findings independently confirmed)
  and its verdict that the engineering quality outruns the evidentiary basis until
  the validation program runs.
- `SIM_REALISM_CRITIQUE_2026-07-18.md` — independent assessment of the executed
  simulator realism and de-circularization work
  (`plans/archive/SIM_REALISM_PLAN_2026-07-18.md`), written against the
  `rf_sim_realism` branch as it merged. It is the assessment
  `docs/dev_guide/dev_guide_simulator.rst` and
  `util/calibration/CAMPAIGN_20260718.md` cite for the capability envelope's
  phrasing. Its open findings are the simulator-fidelity issues that now feed
  the realism-anchored calibration campaign (#309): #325, #329-#333, #341-#345,
  #290, #377, plus the boundary and mirror-parity guards #310 and #311.
- `TITAN_NAV_CONCEPT_2026-07-25.md` — the method analysis that selected haze
  solar-symmetry navigation (the French method) over the alternatives for
  Titan, frozen as the record of why the shipped approach won.
- `TITAN_NAV_PLAN_CRITIQUE_2026-07-25.md`, `..._R2.md`, `..._R3.md`,
  `..._R4_OPUS.md`, `..._R5_OPUS.md`, `..._R6_OPUS.md` — the six adversarial
  review rounds over `plans/archive/TITAN_NAV_PLAN_2026-07-25.md` before
  implementation; each round's findings were folded into the plan revision that
  followed it, so the plan as archived (revision 12) is what they produced.
- `TITAN_NAV_COLLATERAL_SWEEP_2026-07-25.md` — the seventh round: a sweep for
  what the Titan work would touch outside its own subsystem, run before
  implementation began.
- `CK_KERNEL_BRANCH_CRITIQUE_2026-08-07.md` — review of the corrected-pointing
  C-kernel branch (`plans/archive/CK_KERNEL_PLAN_2026-08-04.md`) before it
  merged. Its findings that were not fixed on the branch are filed as issues
  (#433, #437, #440, #443, #444, #446, #448, #455, #459); the kernel
  classification defects it raised (#449, #452) were fixed and closed.
- `CMATRIX_READERS_PLAN_CRITIQUE_2026-08-07.md` — adversarial review of
  `plans/archive/CMATRIX_READERS_PLAN_2026-08-09.md` *before* implementation,
  verdict CONDITIONAL with the mechanism verified correct by execution. Every
  finding was folded into the plan before coding, which its section 0
  enumerates; this document is the record of what the review caught that the
  plan as first written would have shipped wrong.
- `NAV_MEMORY_SWEEP_2026-09-05.md` — the measurement that asked what per-task
  memory limit navigation needs: 1,613 clean peak-RSS measurements over 1,800
  images across the four instruments, each navigated in its own process. Its
  finding is that peak memory follows extended field-of-view area, so the four
  instruments span a factor of six, and that a scene rather than a volume or
  an encounter is what costs the memory. Its final section is a record of the
  ways the measurement itself went wrong, including a probe that inflated the
  process it measured by 2.6 GB.
- `NAV_MEMORY_AFTER_THE_FIXES_2026-09-05.md` — the same method over the 244
  heaviest frames the sweep found, measured after the strip-level release, the
  correlation spectra reordering and the frame-covering-body decline (PRs
  #581, #582 and #583). 8 GB holds for 238 of the 244 and for every Cassini,
  Galileo and New Horizons frame; the six that exceed it are all Voyager.
- `NAV_MEMORY_WHAT_HELD_IT_2026-09-06.md` — the third round, over those six
  Voyager frames. It overturns its predecessor's fragmentation floor: the
  residue a ring render leaves is held by the observation's own backplane
  caches, and dropping it brings all six under the limit at about half the
  runtime, with every offset and status unchanged. That change shipped as PR
  #586 and closed #573. Its two open questions are filed as #584 (holding the
  model stage's backplane cache to one model's worth) and #585 (the
  correlation quality metric's background).
