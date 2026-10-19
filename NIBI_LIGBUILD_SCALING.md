# DOCK3 on Nibi: `ligbuild` does not scale past ~48 concurrent calls per node

Status as of 2026-10-05: diagnosed, not fixed. Use at most 48 workers per node
until the fix in [Proposed fix](#proposed-fix) is in place.

## Symptom

Docking 192 molecules on one 192-core node (`scripts/dock3_192_check.py`,
job 23082420 / worker 23082428, 2026-10-01) returned 0/192 successful docks.
Every molecule failed with `ligbuild_subprocess_timeout`.

The results CSV (`nibi_slurm_dock3_192_check_results.csv`) says
`slurm_submission_failed` instead. That is a second, separate bug; see
[Issue 2](#issue-2-coordinator-reports-slurm_submission_failed-but-the-job-runs).

## Issue 1: the `cxcalc` launcher stalls at high concurrency

### Cause

`ligbuild` calls `cxcalc` from the module-loaded freechem:

    /project/def-mailhoto/soft/software/freechem-19.15.r4/bin/cxcalc

This is a 479-line install4j `/bin/sh` launcher, not a binary. Before it starts
Java it loops over the 290 files in `lib/` and forks a subshell plus an `expr`
process for each jar to build the classpath. One `ligbuild` call does about
1,150 forks in total.

With many calls running at once on the same node, the kernel cost of creating
and tearing down those processes grows much faster than the number of calls.
At 192 concurrent calls the node spends about 163 of its 192 cores in system
time and almost every call is still inside the `cxcalc` launcher when it times
out. Very few calls ever reach AMSOL or the later stages.

The exact kernel lock was not identified: the `strace` summaries for the
192-way phase came out empty, and kernel profiling with `perf` is not available
to regular users (`perf_event_paranoid` is 2 on the login node).

### Measurements

All on one exclusive 192-core node, stock `ligbuild`, launched the same way
`Dock3Oracle` does. Per-call timeout 300 s.

| Concurrent calls | Succeeded | Slowest / median call | Node system CPU | System CPU per call (median) | Job |
|---|---|---|---|---|---|
| 1 (ethanol) | 1/1 | ~15 s | not measured | not measured | 23078029 |
| 24 | 24/24 | 100 s / 52 s | 401 s | not measured | 23150177 |
| 48 | 48/48 | 123 s / 78 s | 1,720 s | 31 s | 23261666 |
| 96 | 94/96 | 300 s / 162 s | 10,436 s | 104 s | 23261666 |
| 192 | 3/192 | 300 s / 300 s | 48,913 s | 226 s | 23261666 |

Supporting observations from job 23261666:

- User CPU per call stays at 14–18 s and page faults per call stay at about
  730k across 48, 96 and 192. The work per call is constant; only its kernel
  cost grows.
- At 192-way, 167–186 of the running processes were `cxcalc` at every 30 s
  sample from 60 s until the timeout.
- I/O wait was 0 in every multi-call phase.
- Even at 48, a call spends about 31 s in the kernel against 14 s of user
  work, so the launcher is wasteful at every scale.

### Ruled out

- **JVM thread pools.** Every 192-way run already had
  `JAVA_TOOL_OPTIONS=-XX:ActiveProcessorCount=1`. The uncommitted change to
  `SINGLE_THREAD_ENV` in `_parallel.py` is harmless but does not fix this.
- **`ligbuild`'s own process pools.** Forcing `multiprocessing.cpu_count()` to 1
  gave 7/192 instead of 2/192 (job 23150177, phases C and D).
- **Environment loading.** 192 concurrent `source dockenv_nibi.sh` calls with no
  `ligbuild` finished in 30 s with 1,006 s of system CPU (job 23150177, phase A).
- **Disk.** No I/O wait.

### Current guidance

- 48 workers per node is the largest size tested clean.
- 64 is untested. Expect it to work but run slower per molecule.
- 96 mostly finishes but calls take about twice as long as at 48, and some will
  exceed the oracle's 150 s `ligbuild_timeout`.
- All tests had the node to themselves. If several 48-core array tasks land on
  the same node they may contend with each other. Whether the contention is per
  job or node-wide is not known.

### Proposed fix

Bypass the launcher. Put a small `cxcalc` wrapper ahead of the freechem one on
`PATH` that starts `java` directly with a wildcard classpath (`lib/*`) and the
same flags the launcher passes. The freechem tree belongs to another group
member, so the wrapper should live in our own space, for example next to
`~/dock/dockenv_nibi.sh`.

Checks before relying on it:

1. On a 1-core `salloc`, confirm the wrapper gives identical `cxcalc` output to
   the stock launcher on a few molecules.
2. Rerun `ligbuild_stage_diag.sh` and confirm the 192-way phase succeeds.

`CxcalcOracle` uses a separately patched freechem copy
(`/project/def-mailhoto/share/soft/software/freechem-19.15.r4`). Whether that
copy has the same launcher loop, and so the same scaling limit, has not been
checked.

## Issue 2: coordinator reports `slurm_submission_failed` but the job runs

In job 23082420 the coordinator logged `Slurm DOCK3 submission failed:` with an
empty message and marked all 192 molecules `slurm_submission_failed`. The array
job had in fact been submitted (23082428, submitted 17:22:50, coordinator failed
17:23:05) and later ran for 6 minutes on 192 cores with nobody waiting for it.

Likely cause, not confirmed: `_SCHEDULER_TIMEOUT` is 30 s in
`slurm_dock3_oracle.py`, and `sbatch` took longer than that. The timeout raises
`_SubmissionError` with no message, which matches the empty log line.

Suggested fix: raise the timeout, and when `sbatch` times out, look the job up
by name and either adopt it or cancel it, so no orphan array runs.

## Incident: `dock_env` module failed to load (2026-10-05)

Job 23260677 failed in 35 s because loading `dock_env/0.0.1` runs
`pip install chemstep==0.4.1` and that wheel was missing from
`/project/def-mailhoto/soft/pkgstore/python/wheelhouse/2023`. The directory had
been modified at 12:00 that day. The wheels (`chemstep`, `ligbuild`,
`ga_decoys`, `omltk`) were restored at 12:49 and the next job loaded the module
normally. If a job fails in its first minute with a pip "No matching
distribution" error, check that directory first.

## Where the evidence is

Scripts and logs are in `/scratch/ethankrz/activelearning/diag/`. Scratch is
purged periodically, so copy anything worth keeping.

| File | What it is |
|---|---|
| `ligbuild_diag.sh`, `ligbuild_diag.log` | Single `ligbuild` call on ethanol with process trees |
| `ligbuild_scale_diag.{sh,py}`, `ligbuild_scale_diag_23150177.log` | Env-only x192, x24, x192 with and without `cpu_count=1` |
| `ligbuild_stage_diag.{sh,py}`, `ligbuild_stage_diag_23261666.log` | x1, x48, x96, x192 with per-program CPU breakdown |
| `strace_23261666_solo_0.txt` | Syscall summary for one call running alone |

The failed 192-molecule run's artifacts are in
`/scratch/ethankrz/activelearning/dock3_queries/b7e4ea3b3c6a4776945d23e653a170c0/`.
