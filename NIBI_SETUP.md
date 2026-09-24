# Running `activelearning` on Nibi: handoff for a Claude session on Nibi

You are a Claude Code session on a **Nibi** login node (Digital Research Alliance of Canada).
Your job is to make this repository run on Nibi, including its DOCK3 and cxcalc molecular
oracles, **exactly as it runs on Narval, without editing any tracked code.** The user is
Ethan Kreuzer (`ethankrz`). This file was written on Narval on 2026-09-24, against commit
`59671a8` on branch `docking_oracles`.

Read the whole file before doing anything. Items marked **VERIFY** are assumptions nobody has
checked on Nibi yet.

---

## 0. ⛔ Hard rule: never run tests or compute on the login node

The same rule as in this repo's `CLAUDE.md`, and it applies on Nibi too. The user's account has
already been flagged **twice** for login-node compute; a third time can get it banned, and the
account is shared across all Alliance clusters.

- No `pytest` of any size (not one test, not `-k`, not "a tiny CPU test"), no `make test`, no
  training, no `activelearning` runs, no smoke tests, no `ligbuild`, `dock64` or `cxcalc` calls,
  and no `python -c` that builds an oracle, **unless you are inside a Slurm allocation the user
  started with `salloc`**.
- **Never start `salloc` or `sbatch` yourself.** Give the user the exact commands and wait.
- Allowed on the login node: `git`, reading and editing files, `ls`/`md5sum`/`stat`/`cat`,
  `module avail`/`module spider`/`module show` (read-only module queries), `sshare`, `squeue`,
  `sinfo`, `uv sync`, and downloads (for example pre-fetching HuggingFace models).

If you are unsure whether a command counts as compute, treat it as compute and ask.

---

## 1. The key fact: every cluster-specific path is a config field

On Narval the oracle defaults are hardcoded constants:

| Constant | Narval value | Source |
|---|---|---|
| `DEFAULT_DOCKENV_SH` | `/project/rrg-mailhoto/share/dockingpackages/dockenv.sh` | `src/activelearning/applications/molecules/dock3_oracle.py:49` |
| `DEFAULT_DOCK64_EXE` | `/project/rrg-mailhoto/share/dock64` | `dock3_oracle.py:50` |
| `DEFAULT_CXCALC_EXE` | `/project/rrg-mailhoto/share/software/freechem-19.15.r4/bin/cxcalc` | `cxcalc_oracle.py:44` |

**`/project/rrg-mailhoto` does not exist on Nibi**, so all three defaults are dead there. But
they are only *defaults*: the pydantic configs in `src/activelearning/oracle/config.py` expose
each one as an optional field, and `null` falls back to the constant:

- `Dock3OracleConfig` / `SlurmDock3OracleConfig`: `dockenv_sh`, `dock64_exe`, `indock_template`,
  `dockfiles_dir`, `tmp_dir`, and for Slurm also `shared_work_dir`, `sbatch_args`,
  `num_workers`, `num_array_tasks`, `max_parallel_tasks`.
- `CxcalcOracleConfig`: `cxcalc_exe`, `env_setup` (default `"module load java"`).

**So Nibi needs no code change, only config overrides.** Do not edit `src/`, `tests/` or the
tracked YAML under `config/`. Put Nibi values in **new, untracked** overlay files (§6) or in
command-line dotlist overrides.

Config mechanics (`src/activelearning/utils/config_loader.py`): YAML files merge left to right
with `OmegaConf.merge`, then `key=value` dotlist overrides are applied. **Lists are replaced
wholesale on merge, not merged by element.** That matters for the multi-fidelity config, whose
oracle is `CompositeOracle` with a `sub_oracles:` list. A Nibi overlay has to restate the
**entire** `oracle:` block. Dotlist overrides like `oracle.sub_oracles.1.dock64_exe=...` are not
a safe way to do this.

---

## 2. What the software admin (Albert) installed on Nibi

From his message, the layout differs from Narval in three ways:

1. **The group is `def-mailhoto`, not `rrg-mailhoto`.** Albert has also moved the Narval paths
   to `def-mailhoto`.
2. **There is no `dockenv.sh` any more.** The DOCK environment is now a module:
   ```sh
   module use /project/def-mailhoto/soft/modulefiles
   module load dock_env/0.0.1
   ```
   Note that this path has **no `share/`** component, unlike Narval's
   `/project/rrg-mailhoto/share/soft/modulefiles`. **VERIFY.**
3. **The dock64 that the oracle must run** (the 2022 build) is at
   `/project/def-mailhoto/soft/software/dock-3.8.0-2022/bin/dock64`.

The receptor files are in `/project/def-mailhoto/share/ampc_chemstep_dockfiles` (this one does
have `share/`). On Narval the grids are in a `dockfiles/` subdirectory of that folder, so expect
`.../ampc_chemstep_dockfiles/dockfiles/INDOCK`. **VERIFY** with `ls`.

He did not say where **cxcalc** (patched ChemAxon freechem 19.15) is. Find it (§3.4).

### 2.1 Why there are two dock64 binaries (do not "fix" this)

Both are DOCK 3.8.0; only the compile dates differ.
- **2022 build → the one the oracle runs.** On Narval it is `$SHARE/dock64`, md5
  `11c439078b58f861385412a82a28ddb8`. It was used for the AmpC 1.7-billion screen and matches the
  AmpC grids. The user verified that 998 of 1,000 molecules re-dock to their reference scores to
  two decimals with this build and these grids.
- **2025 build → the one the `dock/3.8.0` module puts on `PATH`.** On Narval its md5 is
  `f660b687e8fa47deda7bf5271b92c0ec`. The oracle never runs it, but the ligand-building module
  depends on it. It fixes a Fortran overflow that printed `**********` in OUTDOCK's `Total`
  column. The oracle's parser (`_parse_outdock_score`, `dock3_oracle.py:455`) already works around
  that overflow, so **do not switch `dock64_exe` to the 2025 build**.

---

## 3. Verify the install (read-only, on the login node)

Report what you find to the user before writing any config. Everything in this section is
`ls`/`stat`/`md5sum`/`cat`/`module show`, so it is allowed on the login node.

### 3.1 Paths and checksums
```sh
ls -la /project/def-mailhoto /project/def-mailhoto/share /project/def-mailhoto/soft
md5sum /project/def-mailhoto/soft/software/dock-3.8.0-2022/bin/dock64   # expect 11c43907...28ddb8
ls -la /project/def-mailhoto/share/ampc_chemstep_dockfiles
```
The receptor directory must contain exactly these 12 files, with these byte sizes:
```
INDOCK 3306 | ligand.desolv.heavy 882551 | ligand.desolv.hydrogen 882551
matching_spheres.sph 3376 | matching_spheres_3.sph 3376 | matching_spheres_3_2R9W.sph 3592
spheres.pdb 2655 | spheres_3.pdb 2655 | trim.electrostatics.phi 1431806
vdw.bmp 2291351 | vdw.parms.amb.mindock 1653 (mode 0755) | vdw.vdw 18104016
```
`INDOCK` refers to its grids as `../dockfiles/<file>`. The oracle copies whatever
`dockfiles_dir` points at into `<dock_root>/dockfiles/`, so the source directory's name does not
matter, but its contents do.

### 3.2 The `dock_env` module
```sh
module show dock_env/0.0.1     # after: module use /project/def-mailhoto/soft/modulefiles
```
Once the module is loaded, the oracle needs all of the following. Check that the module
provides each one (directly or through its dependencies):
- `ligbuild` on `PATH`;
- a `python` on `PATH` that can `import openeye, Pyro4` (this is what the oracle's warmup probe
  checks);
- `OE_LICENSE`, `CHEMAXON_LICENSE_URL` and `B3DPY_LIBRARIES` set, pointing at readable files or
  directories;
- the `corina/5.0.0`, `amsol/7.1` (with its `lib/libg2c.so.0` on `LD_LIBRARY_PATH`),
  `jchem/24.3.2` and `dock/3.8.0` modules, plus the central modules `StdEnv/2023`,
  `python/3.11.5`, `scipy-stack/2023b`, `ambertools/25`, `rdkit/2024.09.6` and
  `openbabel-omp/3.1.1`. For any central module that is missing on Nibi, **report the available
  version to the user. Do not quietly substitute one.**
- `java`: Narval's `dockenv.sh` also ran `module load java`. Check whether `dock_env` does.

On a fresh node, the `build_3d_dock_py` part may pip-install itself into `$SLURM_TMPDIR`
the first time it is loaded. That is expected. The oracle's `warmup: true` probe runs that install
once, single-threaded, so leave `warmup` on.

### 3.3 `module` must work inside plain `bash -c`
The oracles run `bash -c 'source <dockenv_sh> && ligbuild ...'`, deliberately **without** `-l`.
That only works if lmod exports the `module` shell function:
```sh
bash -c 'type module'    # must say "module is a function"
```
This is cheap and read-only, so it is fine on the login node. Still, confirm it again inside
`salloc`, since compute nodes can differ.

### 3.4 cxcalc
```sh
find /project/def-mailhoto -maxdepth 5 -path '*freechem-19.15.r4/bin/cxcalc' 2>/dev/null
cat <freechem>/bin/cxcalc.vmoptions      # expect -Xmx512m and -XX:ActiveProcessorCount=1
grep -n 'db_home' <freechem>/bin/cxcalc  # expect db_home=/tmp
module spider java                        # what bare `module load java` resolves to (Narval: java/17.0.6)
```
Both patches matter. `CxcalcOracle` does not set `OMP_NUM_THREADS`; it relies on
`ActiveProcessorCount=1` to keep each JVM on one core. If cxcalc is missing or unpatched, **stop
and tell the user**. Do not install ChemAxon from the vendor.

**Unresolved licence question:** on Narval, `cxcalc msdistr` works with no
`CHEMAXON_LICENSE_URL` set. If that is not true on Nibi, a licence failure shows up only as
**all-NaN cxcalc scores plus one warmup warning**. The fix is config only: set
`env_setup: "module load java && export CHEMAXON_LICENSE_URL=<path>"`.

---

## 4. Create the `dockenv_sh` shim (the one non-repo file)

The oracle `source`s whatever file `dockenv_sh` names. Nibi has a module where Narval had a
script, so create a small script **outside the repository** that reproduces Narval's
`dockenv.sh` using the new module. For example, at `$HOME/dock/dockenv_nibi.sh`:
```sh
module use /project/def-mailhoto/soft/modulefiles
module load dock_env/0.0.1
module load java    # only if dock_env does not already load it (§3.2)
```
Narval's version, for reference:
```sh
export CHEMAXON_LICENSE_URL=/project/rrg-mailhoto/share/licenses/license.cxl
module use /project/rrg-mailhoto/share/soft/modulefiles
module load build_3d_dock_py/0.0.0
module load java
source /project/rrg-mailhoto/share/dock_env/bin/activate
```
Keep it minimal. Add lines only for the gaps found in §3.2. The file must be on a filesystem that
compute nodes can see (`$HOME` and `/project` are both fine), because Slurm array tasks source it
too.

---

## 5. Repository, environment and data assets

### 5.1 Code
```sh
git clone https://github.com/ethankreuzer/activelearning.git   # origin; upstream is milaforscience/activelearning
cd activelearning && git checkout docking_oracles               # at or after 59671a8
```
On Narval the repo lives at `/home/ethankrz/activelearning`. Put it at the same path on Nibi if
you can. The existing job scripts `cd` there, and `run_writer.output_dir` and the asset paths are
relative to it.

### 5.2 Python environment (login node, `uv sync` is allowed)
- Do **not** copy `.venv` from Narval. Its interpreter is a symlink into Narval's
  `/cvmfs/.../x86-64-v3/` subtree (Python 3.11.4).
- Install uv 0.11.x into `~/.local/bin` if it is missing, then run `uv sync --all-extras` on the
  login node. Use Python 3.11, to match Narval.
- Every job then runs `uv run --no-sync ...` so the compute node never resolves dependencies.

### 5.3 Untracked assets to copy from Narval (the user does this, or you give them the commands)
None of these are in git. Copy them to the same paths relative to the repo root:

| Path | Size | Notes |
|---|---|---|
| `data/10M_unif_random_subset.csv`, `data/10k_unif_random_subset.csv` (plus `50k`, `ampc_subset_331k`) | 163 MB total | initial datasets |
| `cache/ampc/s3gfn_minimol_ampc_fingerprints.npy` + `.npy.json` | 20 GB | MiniMol feature cache. **Tied to the exact row order of the 10M CSV: copy both byte-identical, never one without the other.** Do not copy the `.lock` file. |
| `minimol_ampc_encoder/` | 34 MB | AmpC MiniMol encoder package and `model/final.pt`. Run its `verify_install.py` inside `salloc`. |
| `~/.cache/huggingface/hub/models--ibm-research--GP-MoLFormer-Uniq` and `models--ibm-research--MoLFormer-XL-both-10pct` | ~360 MB | Either rsync them, or pre-fetch them on the Nibi login node. |

`ampc_hitrate_fits/` **is** tracked (`fitted_params.json`, `data/full_scores.df`), so it arrives
with the clone. Globus or `rsync` between login nodes are both fine for the 20 GB cache.

---

## 6. Nibi config overlays (new untracked files, not edits)

Suggested locations: `config/ampc/overrides/nibi_single_fidelity.yaml` and
`config/ampc/overrides/nibi_multi_fidelity.yaml`. Leave them untracked unless the user says to
commit them.

### 6.1 Single fidelity (`s3gfn_minimol_ampc_variational_single_fidelity.yaml`)
Its `oracle:` is a plain `Dock3Oracle` mapping, not a list, so an overlay merges by key. That
config currently points at `/home/ethankrz/dock_smiles/ampc_dockfiles`, which exists only on
Narval:
```yaml
oracle:
  indock_template: /project/def-mailhoto/share/ampc_chemstep_dockfiles/dockfiles/INDOCK   # VERIFY path
  dockfiles_dir: /project/def-mailhoto/share/ampc_chemstep_dockfiles/dockfiles/
  dockenv_sh: /home/ethankrz/dock/dockenv_nibi.sh
  dock64_exe: /project/def-mailhoto/soft/software/dock-3.8.0-2022/bin/dock64
```
The equivalent dotlist also works, because nothing here is inside a list:
`oracle.dockenv_sh=... oracle.dock64_exe=... oracle.indock_template=... oracle.dockfiles_dir=...`.

### 6.2 Multi fidelity (`s3gfn_minimol_ampc_variational_multi_fidelity_narval.yaml`)
Its oracle is a `CompositeOracle`, and lists are replaced on merge, so **copy the whole `oracle:`
block from the Narval config into the overlay** and change only these fields:
- `CxcalcOracle`: `cxcalc_exe: <path found in §3.4>`, plus `env_setup` only if §3.4 required it.
  Keep `fidelity_confidences: {0: 0.3}`; the comment in the base config explains why it must stay
  explicit.
- `SlurmDock3Oracle`: `dockenv_sh`, `dock64_exe`, `indock_template` and `dockfiles_dir` as in
  §6.1, and then:
  - `sbatch_args`: Nibi's CPU account instead of `--account=rrg-mailhoto_cpu`. Get the
    account names from `sshare -U -u ethankrz`; Narval uses names like `def-yvesbrun_cpu` and
    `rrg-mailhoto_cpu`, and the suffix convention may differ on Nibi. **VERIFY.**
  - `num_workers`: the number of physical cores in one Nibi CPU node (Narval: 64), because each
    array task takes a whole node with `--exclusive --mem=0`. Get it from
    `sinfo -o '%P %c %m %G' | sort -u`. Then resize `num_array_tasks` / `max_parallel_tasks`
    (Narval: 16 × 64 cores ≈ 1,000 cores).
  - `shared_work_dir`: `/scratch/ethankrz/activelearning/dock3_queries` (Nibi `/scratch`
    **VERIFY**). The parent job and every array task must see it at the same path, and it must
    support atomic rename.

Also restate `logger:` and set a separate `run_writer.output_dir` if a Narval run already uses the
base one (starting a run truncates `round_history.jsonl`).

`SlurmDock3Oracle` writes a `run_worker.sh` that `exec`s `sys.executable`, which is the repo's
`.venv/bin/python`. So the repo and `.venv` must be on a filesystem that compute nodes can see.
Nested `sbatch` from inside a job, and `squeue`/`scancel` from compute nodes, must be allowed.
**VERIFY**, which in practice means the smoke test in §7.

---

## 7. Checks the user runs inside `salloc` (you prepare the commands, you do not run them)

Ask the user to start an allocation, for example:
```sh
salloc --account=<nibi cpu account> --time=1:00:00 --cpus-per-task=4 --mem=16G
```
Inside it, in order (each step isolates one piece):

1. `bash -c 'type module'`, and `echo $SLURM_TMPDIR; touch /tmp/x.$$ && rm /tmp/x.$$`.
   The oracle aliases `/tmp/d.$SLURM_JOB_ID` → `$SLURM_TMPDIR` because AMSOL's Fortran path
   buffer overflows past about 80 characters. If `/tmp` is not writable, AMSOL fails silently and
   those molecules come back as NaN.
2. The warmup probe, in exactly the form the oracle runs it:
   ```sh
   bash -c 'source "$HOME/dock/dockenv_nibi.sh" && command -v ligbuild && python -c "import openeye, Pyro4" && env | grep -E "OE_LICENSE|CHEMAXON_LICENSE_URL|B3DPY_LIBRARIES"'
   ```
3. cxcalc, in exactly the form the oracle runs it (licence test):
   ```sh
   cd $SLURM_TMPDIR && echo "CCO m1" > in.smi
   bash -c 'module load java && "<cxcalc>" -o out.sdf msdistr -H 7.4 in.smi' ; grep -c 'DISTR\[pH=7.4\]' out.sdf
   ```
   Judge success by a non-empty `out.sdf`, not the return code.
4. End-to-end through the real oracle classes: build a `Dock3Oracle` and a `CxcalcOracle` in
   `uv run --no-sync python` with the §6 values, and query 3–5 SMILES (for example `CCO`,
   `c1ccccc1C(=O)O`, `CC(=O)Nc1ccc(O)cc1`). Expect finite docking scores, and a cxcalc
   probability of `0.0` or `0.01` per molecule. NaNs mean a broken toolchain; the oracle logs a
   failure stage and reason.
5. The repo's unit tests. These mock the binaries, so they prove wiring, not the install:
   ```sh
   uv run --no-sync --all-extras pytest tests/applications/molecules/test_dock3_oracle.py \
     tests/applications/molecules/test_cxcalc_oracle.py tests/applications/molecules/test_slurm_dock3_oracle.py
   ```
6. The Slurm smoke test (1,000 molecules, nested `sbatch --array`). Make a Nibi copy of
   `scripts/run_dock3_smoke.sh`, keeping it untracked, with the Nibi account and repo path, and
   pass the Nibi overlay after the base config (the script accepts dotlist overrides). The user
   submits it with `sbatch`.

`scripts/verify_dock3_cxcalc.sh`, which `CLUSTER_REPLICATION_DOCK3_CXCALC.md` §7 refers to,
**does not exist in the repo**. Steps 1–4 above replace it.

---

## 8. Job scripts

The Narval job scripts (`job.sh`, `job_*.sh`, `scripts/run_dock3_smoke.sh`) are templates. Make
Nibi copies, again untracked, and change only the cluster-specific lines:
- `#SBATCH --account=...`: the Nibi accounts from `sshare`.
- `#SBATCH --gres=gpu:a100:1`: Nibi's GPU type (`sinfo -o '%G' | sort -u`; **VERIFY**, as
  Nibi's GPU nodes are reported to be H100s). Also update `--cpus-per-task` and `--mem`. Narval's
  `--mem=510000M` is specific to Narval's A100 nodes. The 10M AmpC run needs about 478 GB peak
  RAM, so it needs a large-memory or whole node.
- `cd /home/ethankrz/activelearning`, `source .venv/bin/activate`, and
  `uv run --no-sync python scripts/run_with_sigterm_flush.py <base.yaml> <nibi overlay.yaml> [dotlist...]`.

Keep `WANDB_MODE=offline` and `HF_HUB_OFFLINE=1` unless you confirm that Nibi compute nodes have
internet access (**VERIFY**). They are harmless either way.

---

## 9. Constraints, restated

- Do not modify tracked files (`src/`, `tests/`, tracked `config/` YAML, `pyproject.toml`,
  `uv.lock`). If something seems to *need* a code change, stop and explain why to the user
  first. Every path involved is already configurable (§1).
- Do not install, upgrade or reinstall anything under `/project/def-mailhoto`. That belongs to
  the software admin; report gaps to the user.
- Do not swap `dock64_exe` to the 2025 build, disable `warmup`, or add `-l` to any `bash`
  invocation.
- Do not commit or push unless the user asks.
