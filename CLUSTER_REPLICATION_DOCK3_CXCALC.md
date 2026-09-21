# Replicating DOCK3 and cxcalc on Rorqual and Nibi

**Audience:** whoever installs the shared software (not the `activelearning` maintainer).
**Goal:** run this repository's `Dock3Oracle` / `SlurmDock3Oracle` (high fidelity) and
`CxcalcOracle` (low fidelity) on **Rorqual** and **Nibi** with **no change to the codebase**.
**Reference cluster:** Narval, where everything below currently lives.

---

## 0. The one hard requirement

Every external path these two oracles use is a **hardcoded default in the Python source**, not a
config value:

| Constant | Value | Source |
|---|---|---|
| `DEFAULT_DOCKENV_SH` | `/project/rrg-mailhoto/share/dockingpackages/dockenv.sh` | `src/activelearning/applications/molecules/dock3_oracle.py:49` |
| `DEFAULT_DOCK64_EXE` | `/project/rrg-mailhoto/share/dock64` | `src/activelearning/applications/molecules/dock3_oracle.py:50` |
| `DEFAULT_CXCALC_EXE` | `/project/rrg-mailhoto/share/software/freechem-19.15.r4/bin/cxcalc` | `src/activelearning/applications/molecules/cxcalc_oracle.py:44` |

No config file in the repo overrides any of them. The receptor inputs are the only external paths
set in YAML, in five files:

```
config/ampc/s3gfn_minimol_ampc_variational_multi_fidelity_narval.yaml:108-109
config/ampc/s3gfn_minimol_ampc_variational_multi_fidelity.yaml:97-98
config/ampc/s3gfn_minimol_ampc_variational_single_fidelity.yaml:79-80
config/ampc/sanity_check_s3gfn_minimol_ampc_variational_multi_fidelity.yaml:92-93
config/molecules/s3gfn_minimol_dock3.yaml:67-68
config/molecules/s3gfn_minimol_cxcalc_dock3.yaml:102-103
```

> **Therefore: reproduce the tree at byte-identical absolute paths under
> `/project/rrg-mailhoto/share/`.** If the install lands anywhere else, the three constants above
> and those YAML keys all have to be edited per cluster, which is exactly what we are trying to
> avoid. Same paths ⇒ zero code change, and the same config files run on all three clusters.

---

## 1. Prerequisite I need answered first

**Does the `rrg-mailhoto` group hold a `/project` allocation on Rorqual and on Nibi, with my
account (`ethankrz`) as a member?** Everything below assumes `/project/rrg-mailhoto/share/` is
creatable and group-readable on both clusters. If not, nothing else in this document is possible
and we need a different path decision (see §0).

Note that `/project/rrg-mailhoto` is the *canonical symlink*. Its realpath on Narval is
`/lustre06/project/6102177`, and it **will be different on Rorqual and Nibi**. That difference is
harmless for everything the code touches, because the code only ever uses the `/project/...`
spelling — but it is the reason §3 exists.

---

## 2. Copy verbatim (rsync-safe)

All paths are relative to `SHARE=/project/rrg-mailhoto/share`. Sizes are from Narval today.
Preserve permissions and the executable bit (`rsync -a`).

| # | Path under `$SHARE` | What it is | Size | Who reads it |
|---|---|---|---|---|
| 1 | `dockingpackages/dockenv.sh` | 6-line environment bootstrap; the whole env contract | 244 B | sourced by `dock3_oracle.py:218` and `:301` |
| 2 | `dock64` | **The DOCK 3.8 binary the oracle actually executes** | 3.6 MB | `dock3_oracle.py:441` |
| 3 | `soft/software/dock-3.8.0/bin/dock64` | second, **different** DOCK build exposed by the `dock/3.8.0` module | 6.7 MB | `PATH` inside `dockenv.sh` |
| 4 | `soft/software/corina-5.0.0/` | CORINA 5.0 3D conformer generator (licensed, expiring — see §5) | ~3 MB | ligbuild |
| 5 | `soft/software/amsol-7.1/` | AMSOL 7.1 solvation, **including `lib/libg2c.so.0`** | ~4 MB | ligbuild |
| 6 | `soft/software/jchem-24.3.2/` | ChemAxon JChem 24.3.2 **plus its bundled `jdk-17.0.2/`** | 216 MB | ligbuild (protonation) |
| 7 | `soft/software/build_3d_dock_py-0.0.0-048dd15/` | pinned `build_3d_py` venv (commit `048dd15`) | ~520 MB | optional pinned variant of the module |
| 8 | `soft/modulefiles/` | all of `dock/`, `corina/`, `amsol/`, `jchem/`, `build_3d_dock_py/`, `pydock3/` | < 1 MB | `module use` in `dockenv.sh` |
| 9 | `soft/pkgstore/` | wheelhouse, pinned reqs, `libraries/build_3d_dock_py/`, licences, install recipes | 42 MB | first-source pip install; `B3DPY_LIBRARIES` |
| 10 | `software/freechem-19.15.r4/` | **patched** ChemAxon JChem 19.15 — provides `bin/cxcalc` | 337 MB | `cxcalc_oracle.py:44` |
| 11 | `licenses/license.cxl` | ChemAxon licence set by `dockenv.sh` | 1.4 KB | `CHEMAXON_LICENSE_URL` |
| 12 | `soft/pkgstore/licenses/jchem-license.cxl` | ChemAxon licence set by the `jchem/24.3.2` module (this one wins inside ligbuild) | 1.4 KB | `CHEMAXON_LICENSE_URL` |
| 13 | `soft/pkgstore/licenses/oe_license.txt` | **OpenEye — commercial, must be node-readable** | 7.6 KB | `OE_LICENSE` |
| 14 | `dockingpackages/ligbuild/` | ligbuild source tree (installed `-e` into the venv in §3) | 14 MB | needed to rebuild `dock_env` |
| 15 | `ampc_chemstep_dockfiles/dockfiles/` | AmpC β-lactamase receptor grids (12 files) | 20 MB | `indock_template`, `dockfiles_dir` |

**Total to copy: ≈ 1.2 GB.** Item 7 is the largest and is optional (see §3, note 2).

### 2.1 Do not confuse the two `dock64` binaries

They are genuinely different builds:

```
11c439078b58f861385412a82a28ddb8  $SHARE/dock64                              <- the oracle runs THIS one
11c439078b58f861385412a82a28ddb8  $SHARE/soft/pkgstore/software/dock3/dock64 <- identical copy
f660b687e8fa47deda7bf5271b92c0ec  $SHARE/soft/software/dock-3.8.0/bin/dock64 <- different, module's copy
```

Both must exist at both paths. Mode on `$SHARE/dock64` is `-rwxr-x--x`.

### 2.2 `ampc_chemstep_dockfiles/dockfiles/` must contain exactly these 12 files

```
INDOCK                        3306 B   (mode 0644)
ligand.desolv.heavy         882551 B
ligand.desolv.hydrogen      882551 B
matching_spheres.sph          3376 B
matching_spheres_3.sph        3376 B
matching_spheres_3_2R9W.sph   3592 B
spheres.pdb                   2655 B
spheres_3.pdb                 2655 B
trim.electrostatics.phi    1431806 B
vdw.bmp                    2291351 B
vdw.parms.amb.mindock         1653 B   (mode 0755 — keep the executable bit)
vdw.vdw                   18104016 B
```

`INDOCK` references its grids as `../dockfiles/<file>`, so the directory **name** `dockfiles` is
load-bearing: the oracle copies this directory to `<dock_root>/dockfiles/` and runs `dock64` from a
sibling directory `<dock_root>/r*/`. Do not rename it.

(`/home/ethankrz/dock_smiles/ampc_dockfiles/` on Narval is byte-identical to this directory —
`diff -rq` is clean — so only the `$SHARE` copy needs replicating. I will repoint the four configs
that still use my home copy.)

### 2.3 Not needed

`$SHARE/software/{amsol7.1, corina, bak_corina, jchem-24.3.2, jdk-17.0.2, extralibs}` is a legacy
flat tree superseded by `$SHARE/soft/software/`. Only `freechem-19.15.r4` from `$SHARE/software/`
is required (item 10). `$SHARE/soft/software/build_3d_dock_py-0.0.0` (no commit suffix) does not
exist on Narval and must not be created — the module builds it at runtime (§3, note 2).

---

## 3. Must be rebuilt per cluster — **these three cannot be copied**

### 3.1 `$SHARE/dock_env` (748 MB) — rebuild, do not rsync

This venv provides the `ligbuild` executable. Two things inside it are hard-wired to Narval:

```
$SHARE/dock_env/bin/ligbuild   shebang: #!/lustre06/project/6102177/share/dock_env/bin/python
$SHARE/dock_env/bin/python  ->  /cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/lib/python-exec/python3.11/python
```

The shebang is Narval's numeric Lustre realpath, and the interpreter symlink points into the
**`x86-64-v3`** CVMFS subtree. On a cluster with a different microarchitecture subtree, both are
dead links and `ligbuild` fails with `bad interpreter`. Recreate it instead:

```sh
module load StdEnv/2023 python/3.11.5
python -m venv /project/rrg-mailhoto/share/dock_env
source /project/rrg-mailhoto/share/dock_env/bin/activate
pip install --no-index --upgrade pip
# ligbuild's own declared deps (setup.py)
pip install --no-index 'numpy<2' pandas scipy ConfigArgParse pathos tqdm
pip install --no-index -e /project/rrg-mailhoto/share/dockingpackages/ligbuild
```

That is all this repository needs. For parity with Narval, the venv there also carries three other
editable packages that `activelearning` never calls — install them only if other lab code will use
this venv:

```sh
pip install --no-index -e /project/rrg-mailhoto/share/dockingpackages/CHEMSTEP      # chemstep 0.4.0
pip install --no-index -e /project/rrg-mailhoto/share/dockingpackages/ga_decoys     # ga_decoys 0.1.0
pip install --no-index -e /project/rrg-mailhoto/share/dockingpackages/omlab_toolkit # omltk 0.0.2
```

Narval's `dock_env` resolves to `numpy 1.26.4`, `pandas 2.3.3`, `matplotlib 3.10.1`,
`numba 0.62.1`, `pathos 0.3.4`, `dill 0.4.0`, `multiprocess 0.70.18` (all `+computecanada`
wheels). Exact pins are not required; `numpy<2` **is**.

### 3.2 `build_3d_dock_py` — nothing to build, but verify the wheelhouse resolves

`module load build_3d_dock_py/0.0.0` resolves to `0.0.0.lua`, which `source_sh`s `0.0.0.sh`:

```sh
ROOT=${SLURM_TMPDIR-$SCRATCH}/soft
/project/rrg-mailhoto/share/soft/pkgstore/modules/build_3d_dock_py-0.0.0.sh $ROOT
export EBPYTHONPREFIXES=$ROOT/software/build_3d_dock_py-0.0.0${EBPYTHONPREFIXES:+":$EBPYTHONPREFIXES"}
export OE_LICENSE=/project/rrg-mailhoto/share/soft/pkgstore/licenses/oe_license.txt
export B3DPY_LIBRARIES=/project/rrg-mailhoto/share/soft/pkgstore/libraries/build_3d_dock_py
```

So on **first `source dockenv.sh` on each fresh compute node** it pip-installs itself into
`$SLURM_TMPDIR/soft`, offline:

```sh
pip install --no-index --find-links $SHARE/soft/pkgstore/python/wheelhouse/2023 \
            --requirement $SHARE/soft/pkgstore/python/envs/build_3d_dock_py-0.0.0.reqs.txt
```

The local wheelhouse holds exactly one wheel — `build_3d_dock_py-0.0.0-py3-none-any.whl`
(pure Python, portable). **The other 12 pins come from that cluster's own DRAC wheelhouse**, so
please confirm they all resolve with `--no-index` on Rorqual and Nibi:

```
build_3d_dock_py==0.0.0        ConfigArgParse==1.7      dill==0.4.0
multiprocess==0.70.18          networkx==3.5            OpenEye_toolkits==2021.2.0
OpenEye_toolkits_python3_linux_x64==2021.2.0            pathos==0.3.3
pox==0.3.5                     ppft==1.7.6.9            Pyro4==4.77
serpent==1.41                  tqdm==4.67.1
```

`OpenEye_toolkits==2021.2.0` and `Pyro4==4.77` are the two most likely to be missing. If a pin is
unavailable, please tell me the nearest available version rather than editing the reqs file — the
oracle's warmup probe asserts `import openeye, Pyro4` and I would rather pin deliberately.

`$SHARE/soft/pkgstore/python/recipes/build_3d_dock_py-0.0.0.sh` is the recipe that produced the
wheel (clones `git@github.com:gregorpatof/build_3d_py.git` at commit `048dd15`), in case it has to
be rebuilt from source.

### 3.3 `$SHARE/scratch`-side module variant

`$SHARE/soft/modulefiles/build_3d_dock_py/0.0.0.sh` writes into `${SLURM_TMPDIR-$SCRATCH}/soft`.
On a login node with no `$SLURM_TMPDIR`, that is `$SCRATCH/soft` — **user-owned scratch, which DRAC
purges**. Nothing to install, but be aware the first job on a fresh node pays a one-off pip install
(~1 min). The oracle deliberately serialises it via its warmup probe
(`dock3_oracle.py:188-196`) to avoid a concurrent-pip race, so please leave `warmup: true` alone.

---

## 4. Environment contract the code depends on

Each item names the code that breaks if it is not true on the target cluster.

1. **`module` must be an exported shell function visible to a non-login, non-interactive
   `bash -c`.** The oracles run `bash -c 'source dockenv.sh && …'` — deliberately **not**
   `bash -lc`, because a login shell re-reads `/etc/profile` and clobbers the caller's environment
   (`dock3_oracle.py:200-206`; the argv is pinned by
   `tests/applications/molecules/test_dock3_oracle.py:990-991`). On DRAC this works because lmod
   exports `BASH_FUNC_module%%`. **This is the single most cluster-specific assumption in the whole
   stack — please verify it explicitly** (`bash -c 'type module'` must report a function).

2. **These module names must resolve with these exact version strings** (from
   `soft/modulefiles/build_3d_dock_py/0.0.0.lua`):
   `StdEnv/2023`, `python/3.11.5`, `scipy-stack/2023b`, `ambertools/25`, `rdkit/2024.09.6`,
   `openbabel-omp/3.1.1`, plus the four local ones `corina/5.0.0`, `amsol/7.1`, `jchem/24.3.2`,
   `dock/3.8.0`. If a central module version is absent on Rorqual/Nibi, tell me the available one
   and I will not silently float it.

3. **`module load java`** (bare, no version) must work — that is the entire env setup
   `CxcalcOracle` performs (`cxcalc_oracle.py:48`). On Narval it resolves to `java/17.0.6`. Note
   `cxcalc` here is ChemAxon JChem **19.15** (2019) running under JDK 17; if bare `java` resolves
   to something incompatible on the new clusters, report the version and I will pin
   `env_setup: "module load java/<x>"` in config.

4. **`$SLURM_TMPDIR` must exist, be node-local and writable**, and **`/tmp` must be writable**.
   The oracle creates a symlink `/tmp/d.$SLURM_JOB_ID → $SLURM_TMPDIR` because AMSOL's Fortran
   path buffer overflows past ~80 characters and `$SLURM_TMPDIR`
   (`/localscratch/<user>.<jobid>.0/`) already eats ~30 of them
   (`dock3_oracle.py:65`, `:111-163`). If `/tmp` is unwritable the oracle falls back to a long
   path and AMSOL failures show up as silent NaNs.

5. **Nested `sbatch` from inside a running job must be permitted**, and `squeue` / `scancel` must
   be callable from compute nodes. `SlurmDock3Oracle` submits and polls its own job array
   (`slurm_dock3_oracle.py:632-676`).

6. **A shared filesystem with atomic POSIX rename**, visible at the same path from the parent job
   and every array task, for `shared_work_dir` (result publication uses
   `NamedTemporaryFile` + `os.fsync` + `os.replace`, `slurm_dock3_oracle.py:89-111`). I will set
   this per cluster; it just has to exist.

7. **Binary portability is fine — no rebuilds needed.** All four native binaries are plain
   dynamically-linked x86-64 ELF against old glibc:

   | Binary | glibc floor | Extra shared libs |
   |---|---|---|
   | `$SHARE/dock64` | 2.3.4 | `libz`, `libnuma`, `libpthread`, `librt`, `libm` |
   | `soft/software/dock-3.8.0/bin/dock64` | 2.3.4 | `libz`, `libdl`, `libpthread`, `librt`, `libgcc_s`, `libm` |
   | `corina-5.0.0/bin/corina` | 2.14 | `libm` only |
   | `amsol-7.1/bin/amsol7.1` | 2.7 | **`libg2c.so.0`** |

   `libnuma` must be present on the nodes for `dock64`. `libg2c.so.0` is shipped inside
   `amsol-7.1/lib/` and put on `LD_LIBRARY_PATH` by the `amsol/7.1` modulefile — that `lib/`
   directory must be copied along with `bin/`.

8. **Compute nodes have no outbound internet** on Narval and we assume the same elsewhere. Every
   install step above is `--no-index`; please keep it that way.

---

## 5. Two things about the cxcalc install specifically

**Copy the patched tree; do not reinstall ChemAxon from the vendor.**
`$SHARE/software/freechem-19.15.r4/bin/cxcalc.README` (dated 2023-05-31) documents two local
patches to the install4j launcher that this repo's cost model depends on:

- `db_home=/tmp` — moves the install4j cache off NFS `$HOME`. Without it, every one of the 64
  concurrent cxcalc processes hammers `$HOME` and the oracle stalls.
- `-XX:ActiveProcessorCount=1` (also in `bin/cxcalc.vmoptions`, alongside `-Xmx512m`) — pins the
  JVM to one core. **`CxcalcOracle` is the only oracle in this repo that does not set
  `OMP_NUM_THREADS=1` on its subprocess** (contrast `dock3_oracle.py:970`), precisely because the
  install does it. A stock ChemAxon install will thread per core and oversubscribe the node.

**Open item I need confirmed, not assumed: the cxcalc licence.**
`CxcalcOracle` runs `bash -c 'module load java && <cxcalc> -o out.sdf msdistr -H 7.4 in.smi'`
(`cxcalc_oracle.py:268-274`). That is the *entire* environment — it never sets
`CHEMAXON_LICENSE_URL`, and `~/.chemaxon` does not exist on my account. So on Narval today,
`msdistr` works with no licence variable set. (`freechem-19.15.r4/env.sh` exists but is never
sourced, and it points at the original UCSF paths `/nfs/soft/jchem/...`, which do not exist here.)
Please confirm the same holds after the copy — the verification script in §7 tests exactly this
path, and a licence failure surfaces only as **all-NaN scores plus one warmup warning**, which is
easy to miss. If the new clusters do need `CHEMAXON_LICENSE_URL`, tell me and I will set
`env_setup` in config rather than relying on the caller's shell.

Note also that `dockenv.sh` sets `CHEMAXON_LICENSE_URL=$SHARE/licenses/license.cxl`, but
`module load build_3d_dock_py/0.0.0` → `jchem/24.3.2` then overrides it with
`$SHARE/soft/pkgstore/licenses/jchem-license.cxl`. Both files must be copied; the second is the
effective one inside ligbuild.

`corina-5.0.0/bin-expired-nov2025/` on Narval shows the CORINA licence has expired once already
and the binary was swapped. Please copy the **current** `bin/corina` (dated 2025-11-24), and note
who renews it, since a lapse silently breaks ligbuild on all three clusters at once.

---

## 6. What stays on my side

So the boundary is clear — after §2 and §3 land, I handle all of this myself, with no further
install work:

- per-cluster Slurm account and partition in `job.sh` and in the `sbatch_args` list of
  `config/ampc/s3gfn_minimol_ampc_variational_multi_fidelity_narval.yaml:120-126` (currently
  `--account=rrg-mailhoto_cpu`, `--exclusive`, `--mem=0`, `--time=03:00:00`, sized to Narval's
  64-core nodes);
- `shared_work_dir` (currently `/scratch/ethankrz/activelearning/dock3_queries`);
- repo checkout path and `uv sync --all-extras` on each login node;
- the project's own untracked data assets (10M-molecule CSV, MiniMol fingerprint cache, the AmpC
  MiniMol encoder, pre-fetched GP-MoLFormer weights) — **not your problem**;
- repointing the four configs that still reference `/home/ethankrz/dock_smiles/ampc_dockfiles` at
  the `$SHARE` copy.

---

## 7. Acceptance test

`scripts/verify_dock3_cxcalc.sh` in this repository is a self-contained check. It touches nothing
outside `$SLURM_TMPDIR` / `/tmp` and reads `$SHARE` read-only.

> **It must be run inside a Slurm allocation, never on a login node.** The script refuses to run
> without `$SLURM_JOB_ID`. This repo's `CLAUDE.md` forbids login-node compute, and my account has
> already been flagged twice.

```sh
salloc --account=<your_account> --time=1:00:00 --cpus-per-task=4 --mem=16G
bash scripts/verify_dock3_cxcalc.sh
```

It prints one `PASS` / `FAIL` / `WARN` line per item and exits non-zero on any `FAIL`:

1. every path in §2 exists, with the right mode and the `dock64` md5s;
2. `source dockenv.sh` succeeds in a bare `bash -c`, and afterwards `command -v ligbuild` resolves,
   `python -c "import openeye, Pyro4"` works, and `$OE_LICENSE` / `$CHEMAXON_LICENSE_URL` /
   `$B3DPY_LIBRARIES` point at readable files;
3. `module` is an exported function in a non-login shell, and each of the ten required modules
   loads;
4. `cxcalc -o out.sdf msdistr -H 7.4` on `CCO` produces an SDF containing `DISTR[pH=7.4]`, run
   through **exactly** the `bash -c "module load java && …"` form the oracle uses — this is the
   licence test from §5;
5. `ligbuild` builds one real molecule to a `.tgz` containing a `.db2`;
6. `dock64 INDOCK_run` docks that `.db2` against the copied AmpC grids in the oracle's exact
   directory layout and yields a parseable `OUTDOCK` score.

Steps 5–6 are the real end-to-end proof; 1–4 exist so a failure tells you *which* piece is missing.

Once it passes, I run the repo-side checks myself (also inside `salloc`):

```sh
uv run --no-sync --all-extras pytest \
  tests/applications/molecules/test_dock3_oracle.py \
  tests/applications/molecules/test_cxcalc_oracle.py \
  tests/applications/molecules/test_slurm_dock3_oracle.py
```

(these mock the binaries, so they prove wiring, not the install), then the real end-to-end
`scripts/smoke_test_slurm_dock3.py`, which generates 1,000 molecules and drives a nested
`sbatch --array` DOCK3 batch.

---

## Appendix A — `$SHARE/dockingpackages/dockenv.sh`, verbatim

```sh
export CHEMAXON_LICENSE_URL=/project/rrg-mailhoto/share/licenses/license.cxl

module use /project/rrg-mailhoto/share/soft/modulefiles
module load build_3d_dock_py/0.0.0
module load java

source /project/rrg-mailhoto/share/dock_env/bin/activate
```

## Appendix B — modulefiles, verbatim

`$SHARE/soft/modulefiles/build_3d_dock_py/0.0.0.lua`:

```lua
depends_on("StdEnv/2023")
depends_on("python/3.11.5")
depends_on("scipy-stack/2023b")
depends_on("ambertools/25")
depends_on("rdkit/2024.09.6")
depends_on("openbabel-omp/3.1.1")

depends_on("corina/5.0.0")
depends_on("amsol/7.1")
depends_on("jchem/24.3.2")
depends_on("dock/3.8.0")

source_sh("bash", "/project/rrg-mailhoto/share/soft/modulefiles/build_3d_dock_py/0.0.0.sh")
```

`$SHARE/soft/modulefiles/build_3d_dock_py/0.0.0.sh`:

```sh
#!/usr/bin/env bash

ROOT=${SLURM_TMPDIR-$SCRATCH}/soft

/project/rrg-mailhoto/share/soft/pkgstore/modules/build_3d_dock_py-0.0.0.sh $ROOT

export EBPYTHONPREFIXES=$ROOT/software/build_3d_dock_py-0.0.0${EBPYTHONPREFIXES:+":$EBPYTHONPREFIXES"}
export OE_LICENSE=/project/rrg-mailhoto/share/soft/pkgstore/licenses/oe_license.txt
export B3DPY_LIBRARIES=/project/rrg-mailhoto/share/soft/pkgstore/libraries/build_3d_dock_py
```

`$SHARE/soft/modulefiles/dock/3.8.0` — `prepend-path PATH $root/bin`, with
`root = $SHARE/soft/software/dock-3.8.0`.
`corina/5.0.0` — same shape.
`amsol/7.1` — same, plus `prepend-path LD_LIBRARY_PATH $root/lib`.
`jchem/24.3.2`:

```tcl
set             root                /project/rrg-mailhoto/share/soft/software/jchem-24.3.2
setenv          INSTALL4J_JAVA_HOME  $root/jdk-17.0.2
setenv          CHEMAXON_LICENSE_URL /project/rrg-mailhoto/share/soft/pkgstore/licenses/jchem-license.cxl
prepend-path    PATH                 $root/bin
```

## Appendix C — how the code invokes each binary

For reference when debugging a failed acceptance test. `_run_subprocess` is
`src/activelearning/applications/molecules/_subprocess.py`; it uses `start_new_session=True` and
SIGKILLs the process group on timeout.

```python
# cxcalc_oracle.py:268-274  — one call per chunk of <=12,500 molecules, cwd = the chunk tempdir
bash -c 'module load java && "<cxcalc>" "-o" "<out.sdf>" "msdistr" "-H" "7.4" "<input.smi>"'
#   input.smi lines are "<smiles> <id>"; success = non-empty out.sdf, NOT the return code

# dock3_oracle.py:217-223  — warmup probe, once per process
bash -c 'source "<dockenv.sh>" && command -v "ligbuild" >/dev/null && python -c "import openeye, Pyro4" 2>&1'

# dock3_oracle.py:295-306  — per molecule, cwd = <work>/, timeout 300 s
bash -c 'source "<dockenv.sh>" && ligbuild "<work>/lig.smi" "<work>/ligbuild_out" "<work>/custom_parms.json"'
#   custom_parms.json = {"verbose": 1, "timeout": 150}; return code ignored (ligbuild rc=1 on a
#   benign cleanup race); success = a *.tgz appears, containing a *.db2

# dock3_oracle.py:441-443  — per molecule, cwd = <dock_root>/r*/
<dock_root>/dock64 INDOCK_run
#   return code ignored (dock64 raises ieee_inexact); success = OUTDOCK exists and parses
```

Directory layout `dock64` requires — `run_dir` must be a **direct child** of the dock root so that
`INDOCK`'s `../dockfiles/` references resolve:

```
<dock_root>/                 # mkdtemp(prefix="k"), created once per oracle
├── dockfiles/               # real copy of §2.2, never a symlink
├── dock64                   # copy of $SHARE/dock64, chmod +x
└── r<random>/               # mkdtemp(prefix="r"), one per molecule
    ├── INDOCK_run           # INDOCK with "DOCK 3.7 parameter" -> "DOCK 3.8 parameter"
    │                        # and "split_database_index" -> <relative path to the .db2>
    ├── <bundle>/lig.db2
    └── OUTDOCK, test.*      # dock64 output; only OUTDOCK is read
```
