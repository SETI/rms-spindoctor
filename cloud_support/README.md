# Cloud support

Operator tooling for running SpinDoctor on GCP through
[cloud_tasks](https://github.com/SETI/rms-cloud-tasks). Nothing here is part of
the `rms-spindoctor` distribution; these files live in the repository and are
run from a checkout.

- `scripts/make_coiss_tasks.py` -- Cassini ISS task files, grouped into volume
  sets of roughly 50,000 images
- `scripts/make_vgiss_tasks.py` -- Voyager ISS task files, one per planetary
  encounter
- `scripts/make_gossi_tasks.py` -- Galileo SSI task file, the whole mission as
  one batch
- `scripts/make_nhlorri_tasks.py` -- New Horizons LORRI task file, the whole
  mission as one batch
- `scripts/task_gen_common.py` -- What the four generators share
- `tests/test_task_gen_common.py` -- Unit tests for the volume grouping; run
  with `pytest cloud_support/tests`
- `startup_script.sh` -- Runs on each compute instance: mounts the data disk,
  installs SpinDoctor, runs a worker
- `job_config_template.yml` -- Starting point for a `cloud_tasks run`
  configuration

## Generating task files

The generators do not enumerate images themselves. Each one runs `sd_offset
--output-cloud-tasks-file`, which is the program that defines the task JSON, and
then copies or divides what it wrote. A task file made here is the same document
`sd_offset` writes directly.

Run them from a checkout with the project virtualenv active, on a machine
configured to read a PDS3 holdings tree the ordinary way -- `PDS3_HOLDINGS_DIR`
or `environment.pds3_holdings_root` in the configuration.

```bash
python cloud_support/scripts/make_nhlorri_tasks.py \
    --holdings-root gs://my-bucket/holdings \
    --output-dir ~/tasks
```

`--output-dir` may be a cloud URL as readily as a path, since `cloud_tasks run
--task-file` loads a task file from either.

`--holdings-root` is required and has no default on purpose, and it governs the
task file alone. **It does not redirect the enumeration**: `sd_offset` reads the
holdings this machine is already configured for, which is usually a local mount
and is the fastest thing to enumerate from. What the flag names is where the
*workers* will read, and every image and label URL is rewritten under it before
the file is written.

The rewrite is a matter of exchanging one holdings root for another: everything
from the volumes directory onward (`volumes/` for most instruments,
`calibrated/` for Cassini) identifies the file, and everything before it is the
root. So an enumeration that produced

```text
/local/pds/holdings/calibrated/COISS_2xxx/COISS_2001/data/.../N..._CALIB.IMG
```

is written into the task file as

```text
gs://my-bucket/holdings/calibrated/COISS_2xxx/COISS_2001/data/.../N..._CALIB.IMG
```

and each run prints the root it read and the root it wrote. The local root is
read back from the URLs rather than assumed, and every URL in a file has to
agree on it; a file whose URLs come from two roots, or one whose URLs lie under
no volumes directory at all, stops the run rather than being half rewritten.

Anything after a bare `--` is passed through to `sd_offset`, so the usual
selection options are available. To re-run only what has not been navigated yet:

```bash
python cloud_support/scripts/make_vgiss_tasks.py \
    --holdings-root gs://my-bucket/holdings --output-dir ~/tasks \
    -- --has-no-offset-file --nav-results-root gs://my-bucket/nav-offset-results
```

### How each mission is divided

**New Horizons LORRI** and **Galileo SSI** are each one batch, written as
`nhlorri_tasks.json` and `gossi_tasks.json`.

**Voyager ISS** is split by encounter, because the volume sets already are:
VGISS_5xxx is Jupiter, 6xxx Saturn, 7xxx Uranus, 8xxx Neptune, each holding both
spacecraft. Four files are written, `vgiss_tasks_jupiter.json` through
`vgiss_tasks_neptune.json`; `--planets` restricts which.

**Cassini ISS** is divided into consecutive groups of whole volumes holding
roughly `--group-size` images each (50,000 by default). A volume is never split,
so a group is only approximately that size. Two things keep the groups near each
other in size rather than merely near the target: the images are spread evenly
over as many groups as the target implies, so the last group is not left holding
whatever remains, and a volume that would overshoot a group is held back for the
next one whenever stopping short lands closer to the target than going over. A
remainder under a fifth of a group joins the group before it. Each file is named
for the group number and the volumes it spans, for example
`coiss_tasks_03_COISS_2015_COISS_2028.json`.

No volume's image count is recorded anywhere cheap to read, so the script
enumerates each volume in turn, counts what came back, and forms the groups from
those counts. That is the slow part: about two seconds a volume, so a few
minutes for the whole mission. It prints the running total as it goes, and
per-volume files are kept if you pass `--work-dir`.

Task ids are renumbered as groups are merged, so no two tasks in anything one
run writes share an id.

## Running a job

The startup script is what each instance runs, as root, when `cloud_tasks`
creates it. Its first line must be exactly `#!/bin/bash`: cloud_tasks inserts
its own `RMS_CLOUD_TASKS_*` exports directly below the shebang and rejects any
shebang that does not end in `/bash`, which `#!/usr/bin/env bash` does not.

It expects a **data disk** carrying the SPICE kernels and star catalogs, laid
out as

```text
SPICE/                  -> SPICE_PATH, and the mount point itself is OOPS_RESOURCES
star-catalogs/UCAC4     -> UCAC4_PATH
star-catalogs/YBSC      -> YBSC_PATH
```

All three directories are checked once the disk is mounted, so a disk missing
one of them stops the instance rather than reaching a worker that fails an image
at a time.

cloud_tasks creates instances with a boot disk and nothing else, so the script
attaches the data disk itself with `gcloud compute instances attach-disk`,
running as the instance service account, and then mounts it `ro,noload`. The
instance name and zone come from the metadata server rather than being assumed,
and the Google Cloud CLI is installed first if the boot image does not carry it.
Three things have to be true for that to work:

- The disk and the instances are in the same zone. cloud_tasks spreads instances
  across a region unless `zone:` is pinned in the job configuration, so pin it.
- The instance service account may `compute.instances.attachDisk` and
  `compute.disks.use`, and may write the results bucket. cloud_tasks gives
  instances the `cloud-platform` scope, so IAM is what decides.
- No instance holds the disk read-write. A read-only attachment is refused while
  one does, and GCP caps how many instances may hold one disk read-only at once,
  which is worth checking against the pool size before scaling up. The pd-ssd's
  throughput is shared out among all of its readers.

Everything an operator changes is in the settings block at the top of the
script: the git URL and ref to install, which worker to run
(`sd_offset_cloud_tasks` by default, but any of the `*_cloud_tasks` programs),
the disk name and mount point, the results root, and the holdings root. Each is
`${VAR:-default}`, so a wrapper may export them instead of editing. A results
root left at the template's placeholder stops the instance before it installs
anything.

The ref may name a branch, a tag or a commit, and the checkout is detached
either way. Only a 40-character commit SHA actually pins a pool: a pool is not
created all at once, since cloud_tasks replaces preempted instances for as long
as the queue has work, so a branch or tag that moves mid-job gives later
instances a different commit than earlier ones. A ref that is not a SHA is
reported as moving in the startup log, together with the commit it resolved to,
and whichever form is used the resolved commit is logged before installation, so
what an instance ran is recoverable from its log.

The script also pins `OMP_NUM_THREADS` and friends to 1. cloud_tasks already
runs `RMS_CLOUD_TASKS_NUM_TASKS_PER_INSTANCE` tasks at once, one per vCPU;
unpinned, each task's BLAS opens a thread per core on top of that and the
instance spends its time in contention.

A startup failure is visible only on the serial console and in
`/var/log/spindoctor-startup.log` on the instance, since an instance that fails
to install never reaches the queue. The script logs both places and stops at the
first step that fails.

Then, with a configuration copied from `job_config_template.yml`:

```bash
cloud_tasks run --config my_job.yml --task-file ~/tasks/vgiss_tasks_uranus.json
```

Start with one small task file (Uranus is 6,460 images), read the per-task time
and memory out of the event log, and set `max_runtime`, `min_memory_per_task`
and the boot disk sizes from what you measure before starting a Cassini group.
