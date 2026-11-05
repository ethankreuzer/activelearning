"""Entry point for the active learning loop.

Usage
-----
    uv run activelearning <config.yaml> [<config2.yaml> ...] [key=value ...]

Positional arguments
--------------------
args
    One or more YAML configuration file paths, optionally followed by
    OmegaConf dotlist overrides. Arguments containing ``=`` are treated as
    overrides; all other arguments are treated as config file paths.
    Multiple config files are merged left to right (later files take
    precedence for shared keys). Overrides are applied last.

Examples
--------
    # Single config with overrides
    uv run activelearning config/base.yaml budget.available_budget=10

    # Merge two configs, then apply an override
    uv run activelearning config/base.yaml config/mf.yaml sampler.num_samples=500
"""

import argparse
import hashlib
import json
import logging
import os
import shlex
import subprocess
import sys
from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig, OmegaConf

from activelearning.logger.config import bootstrap_logger_backend_imports
from activelearning.utils.config_loader import load_config

if TYPE_CHECKING:
    from activelearning.config import ActiveLearningConfig


#: Environment variable overriding how much of the library's own logging is
#: shown, by name (``DEBUG``, ``INFO``, ``WARNING``, ...).
_LOG_LEVEL_VARIABLE = "ACTIVELEARNING_LOG_LEVEL"

#: Marks the handler this module installed, so configuring twice is harmless.
_HANDLER_NAME = "activelearning-stdout"

_module_logger = logging.getLogger(__name__)


def configure_logging(level: int | str | None = None) -> None:
    """Send this library's logging to stdout, third-party logging to warnings.

    Nothing configured the ``logging`` module before this, so the root logger
    had no handler and Python's last-resort fallback applied: WARNING and above,
    to stderr, and every ``logger.info`` in the library discarded. That is why
    job 24882282 docked 2,000 molecules without printing one progress line.

    Only the ``activelearning`` logger is lowered to INFO. Doing it at the root
    would turn on every dependency's INFO logging -- botorch, matplotlib, wandb
    and graphium among them -- and bury the run's own output.

    Records go to **stdout** so a job's log is in one file, the one not already
    full of RDKit's parse complaints, which it writes to stderr from C++ and
    which no Python logging setting affects.

    Parameters
    ----------
    level : int or str, optional
        Level for the ``activelearning`` logger. Defaults to the
        ``ACTIVELEARNING_LOG_LEVEL`` environment variable, else INFO.
    """
    if level is None:
        level = os.environ.get(_LOG_LEVEL_VARIABLE) or logging.INFO
    root = logging.getLogger()
    if not any(handler.name == _HANDLER_NAME for handler in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.name = _HANDLER_NAME
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(handler)
    # The root stays at WARNING; only this library is verbose.
    if root.level == logging.NOTSET or root.level > logging.WARNING:
        root.setLevel(logging.WARNING)
    logging.getLogger("activelearning").setLevel(level)


def _parse_args(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        prog="activelearning",
        description=(
            "Run the active learning loop from one or more YAML config files. "
            "Arguments containing '=' are treated as OmegaConf dotlist overrides; "
            "all other arguments are treated as config file paths merged left to right."
        ),
    )
    parser.add_argument(
        "args",
        nargs="+",
        metavar="config_or_override",
        help=(
            "Config file path(s) and/or key=value overrides. "
            "Example: config/base.yaml config/mf.yaml budget.available_budget=5"
        ),
    )
    return parser.parse_known_args(argv)


def process_arguments(
    argv: Sequence[str] | None = None,
) -> tuple[DictConfig, "ActiveLearningConfig", list[Path], list[str]]:
    """Parse CLI arguments, load configs, and validate the merged config."""
    args, unknown = _parse_args(argv)

    configs = [arg for arg in args.args if "=" not in arg]
    overrides = [arg for arg in args.args if "=" in arg]

    if unknown:
        import warnings

        warnings.warn(
            f"Unrecognised arguments ignored: {unknown}. "
            "Pass OmegaConf overrides as positional key=value tokens, "
            "e.g. budget.available_budget=10.",
            stacklevel=2,
        )

    if not configs:
        raise ValueError(
            "At least one config file path must be provided. "
            "Arguments containing '=' are interpreted as overrides."
        )

    config_paths = [Path(config).resolve() for config in configs]
    raw_cfg = load_config(
        path=configs if len(configs) > 1 else configs[0],
        overrides=overrides or None,
    )
    bootstrap_logger_backend_imports(OmegaConf.to_container(raw_cfg, resolve=False))

    from activelearning.config import ActiveLearningConfig
    from activelearning.utils.config_loader import parse_config

    cfg = parse_config(raw_cfg, ActiveLearningConfig)
    return raw_cfg, cfg, config_paths, overrides


def main(argv: Sequence[str] | None = None) -> None:
    """Load config, build components, run the active learning loop."""
    configure_logging()
    raw_cfg, cfg, config_paths, overrides = process_arguments(argv)

    from activelearning.active_learning import active_learning
    from activelearning.runtime import bind_runtime_context
    from activelearning.utils.seeding import set_global_seed

    # Seed before anything is built. runtime.seed was reaching only the
    # components that read it explicitly -- the S3-GFN sampler reseeds per round
    # and the subset selectors carry their own generators -- while NumPy's
    # global RNG and the cuDNN determinism flags were never set at all, so a
    # config advertising seed 42 did not actually govern the whole run.
    set_global_seed(cfg.runtime.seed)

    # Built once and shared: the run writer persists it to disk and the logger
    # sends it to the tracking backend, so the two cannot disagree about what
    # produced the run.
    run_metadata = _build_run_metadata(
        config_paths=config_paths,
        overrides=overrides,
        raw_cfg=raw_cfg,
        cfg=cfg,
    )
    _warn_on_dirty_working_tree(run_metadata)
    _save_uncommitted_diff(run_metadata, cfg=cfg)

    # Build instances of active learning components from pydantic config models
    dataset = cfg.dataset.build()
    surrogate = cfg.surrogate.build()
    acquisition = cfg.acquisition.build()
    sampler = cfg.sampler.build()
    selector = cfg.selector.build()
    oracle = cfg.oracle.build()
    budget = cfg.budget.build()
    logger = cfg.logger.build() if cfg.logger is not None else None
    run_writer = (
        cfg.run_writer.build(
            metadata=run_metadata,
            # The sampler's reward shape, so each round's scored-samples CSV
            # can carry the acquisition on the scale the policy was trained
            # against beside the raw value. Read off the sampler config here
            # rather than configured on the writer: one source of truth, so
            # the saved column cannot describe a reward the run did not use.
            # A sampler without these attributes simply omits the column.
            reward_transform=getattr(cfg.sampler, "reward_transform", None),
            log_score_floor=getattr(cfg.sampler, "log_score_floor", None),
        )
        if cfg.run_writer is not None
        else None
    )
    runtime_context = cfg.runtime.build(logger=logger)

    bind_runtime_context(
        [dataset, surrogate, acquisition, sampler, selector, oracle],
        runtime_context,
    )

    if logger is not None:
        logger.log_config(OmegaConf.to_container(raw_cfg, resolve=True))
        # The VALIDATED config as well, under `effective`. The raw config above
        # is the YAML plus overrides, so anything left to a schema default is
        # absent from it -- with `sampler.loss` unset a run cannot be told from
        # a vargrad one, and the GP's kernel, output scale and pinned mean are
        # invisible unless written out. This dump carries every field, default
        # or not, so a run in the table always says exactly what it ran.
        logger.log_config({"effective": cfg.model_dump(mode="json")})
        # Provenance too, which until now reached only run_manifest.json on
        # disk. A tracking-backend run that cannot say which commit, which
        # archive file and which encoder checkpoint produced it is not a record
        # of an experiment, and the on-disk manifest is no help once the output
        # directory is moved or deleted. `config` and `effective_config` are
        # omitted here because the two calls above already carry them.
        logger.log_config(
            {
                "cli": run_metadata["cli"],
                "provenance": run_metadata["provenance"],
            }
        )

    # Run active learning campaign
    dataset, total_cost, num_rounds = active_learning(
        dataset=dataset,
        surrogate=surrogate,
        acquisition=acquisition,
        sampler=sampler,
        selector=selector,
        oracle=oracle,
        budget=budget,
        runtime_context=runtime_context,
        run_writer=run_writer,
        diagnostics_config=cfg.diagnostics,
    )

    print(f"Done. Rounds: {num_rounds} | Total cost: {total_cost:.4f}")


def _build_run_metadata(
    *,
    config_paths: Sequence[Path],
    overrides: Sequence[str],
    raw_cfg: Any,
    cfg: Any = None,
) -> dict[str, Any]:
    """Build compact reproducibility metadata from the CLI inputs."""
    resolved_config = OmegaConf.to_container(raw_cfg, resolve=True)
    return {
        "config": resolved_config,
        # Every field including the defaults the raw config leaves out, so the
        # on-disk record identifies the run as completely as the W&B one.
        "effective_config": (
            None if cfg is None else cfg.model_dump(mode="json")
        ),
        "cli": {
            "command": _format_activelearning_command(config_paths, overrides),
            "config_files": [_display_path(path) for path in config_paths],
            "config_overrides": list(overrides),
        },
        "provenance": {
            "git_commit_sha": _git_commit_sha(),
            # A SHA describes the code only if the tree matched it.
            **_git_working_tree_state(),
            "resolved_config_md5": _stable_md5(resolved_config),
            "input_files": _collect_input_files(config_paths, resolved_config),
        },
    }


#: Filename of the saved working-tree patch, inside the run's output directory.
_UNCOMMITTED_PATCH_NAME = "uncommitted.patch"


def _warn_on_dirty_working_tree(run_metadata: dict[str, Any]) -> None:
    """Say loudly, at startup, that the recorded commit is not what is running."""
    provenance = run_metadata.get("provenance", {})
    if not provenance.get("dirty"):
        return
    paths = provenance.get("dirty_paths") or []
    _module_logger.warning(
        "Working tree differs from HEAD (%s): %d tracked file(s) modified -- %s. "
        "The recorded commit %s does NOT describe the code being run. The diff "
        "is saved beside the run outputs as %s and its md5 is in the run's "
        "provenance, but committing before a long run is the only way to make "
        "the run reproducible from the commit alone.",
        provenance.get("git_commit_sha"),
        len(paths),
        ", ".join(paths[:10]) + (", ..." if len(paths) > 10 else ""),
        provenance.get("git_commit_sha"),
        _UNCOMMITTED_PATCH_NAME,
    )


def _save_uncommitted_diff(run_metadata: dict[str, Any], *, cfg: Any) -> None:
    """Write the working-tree patch beside the run's outputs, when it is dirty.

    The digest in the provenance distinguishes two dirty runs; the patch itself
    is what lets one be reconstructed. Nothing here is allowed to fail a run:
    a run that cannot record its diff is worse off, not broken.
    """
    provenance = run_metadata.get("provenance", {})
    if not provenance.get("dirty"):
        return
    output_dir = getattr(getattr(cfg, "run_writer", None), "output_dir", None)
    if output_dir is None:
        _module_logger.warning(
            "No run_writer output directory, so the working-tree patch cannot be "
            "saved. Only its md5 is recorded."
        )
        return
    diff = _git_uncommitted_diff()
    if not diff:
        return
    try:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        patch_path = destination / _UNCOMMITTED_PATCH_NAME
        patch_path.write_text(diff, encoding="utf-8")
    except OSError as error:
        _module_logger.warning("Could not save the working-tree patch: %s", error)
        return
    run_metadata["provenance"]["uncommitted_diff_path"] = _display_path(patch_path)


def _format_activelearning_command(
    config_paths: Sequence[Path],
    overrides: Sequence[str],
) -> str:
    """Return a shell-friendly reproduction command for the current run."""
    args = [_display_path(path) for path in config_paths] + list(overrides)
    return "activelearning " + " ".join(shlex.quote(arg) for arg in args)


def _git_commit_sha() -> str | None:
    """Return the current Git commit SHA when available."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


def _git_working_tree_state() -> dict[str, Any]:
    """Return how the working tree differs from HEAD, if at all.

    A commit SHA alone does not identify the code that ran: a run launched from
    a modified tree records a SHA describing something else, and nothing in the
    run says so. This reports the difference so a run can be trusted or
    distrusted on its own evidence.

    Tracked modifications and untracked files are reported separately and only
    the former set ``dirty``. An untracked file does not change what the
    recorded commit means -- this repository habitually carries a few unfinished
    job scripts -- so counting them as dirty would make the flag meaningless.

    Returns
    -------
    dict
        ``dirty`` (None when git is unavailable), ``dirty_paths``,
        ``untracked_paths`` and ``uncommitted_diff_md5``, the digest of
        ``git diff HEAD`` that distinguishes two runs from differently modified
        trees.
    """
    unknown: dict[str, Any] = {
        "dirty": None,
        "dirty_paths": [],
        "untracked_paths": [],
        "uncommitted_diff_md5": None,
    }
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0:
        return unknown

    dirty_paths: list[str] = []
    untracked_paths: list[str] = []
    for line in status.stdout.splitlines():
        if not line.strip():
            continue
        # Porcelain v1: two status characters, a space, then the path.
        entry = line[3:].strip()
        if line.startswith("??"):
            untracked_paths.append(entry)
        else:
            dirty_paths.append(entry)

    state: dict[str, Any] = {
        "dirty": bool(dirty_paths),
        "dirty_paths": sorted(dirty_paths),
        "untracked_paths": sorted(untracked_paths),
        "uncommitted_diff_md5": None,
    }
    if dirty_paths:
        state["uncommitted_diff_md5"] = _stable_text_md5(_git_uncommitted_diff() or "")
    return state


def _git_uncommitted_diff() -> str | None:
    """Return the full diff of the working tree against HEAD, staged included."""
    result = subprocess.run(
        ["git", "diff", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def _collect_input_files(
    config_paths: Sequence[Path],
    resolved_config: Any,
) -> list[dict[str, str]]:
    """Collect explicit input files and their hashes for reproducibility."""
    seen_paths: set[Path] = set()
    recorded_files: list[dict[str, str]] = []

    for path in config_paths:
        if path.is_file() and path not in seen_paths:
            recorded_files.append(_file_record(path))
            seen_paths.add(path)

    for candidate, is_directory in _iter_config_file_candidates(resolved_config):
        resolved_candidate = _resolve_existing_path(candidate, directory=is_directory)
        if resolved_candidate is None or resolved_candidate in seen_paths:
            continue
        recorded_files.append(_file_record(resolved_candidate))
        seen_paths.add(resolved_candidate)

    return recorded_files


#: Config keys whose string values name an input FILE worth recording.
#: Matched on the key, not the value, because a config is full of strings and
#: only these are inputs. Extended past the original four generic names after
#: an audit of sf-10round-5000dock-5000steps-seed42, whose manifest recorded
#: three files and silently omitted the rest: the MiniMol checkpoint that
#: defines the GP's entire feature space, the fixed-feature store and the SMILES
#: list that fixes its row order, the GFN prior, and the DOCK3 installation.
#: A value that is not an existing file -- a Hugging Face hub id under
#: ``model_name_or_path``, say -- resolves to nothing and is skipped.
_INPUT_FILE_KEYS = frozenset(
    {
        "path",
        "config_path",
        "file",
        "file_path",
        "checkpoint_path",
        "state_path",
        "store_path",
        "smiles_path",
        "model_name_or_path",
        "tokenizer_name_or_path",
        "indock_template",
        "dock64_exe",
        "dockenv_sh",
    }
)

#: Config keys naming an input DIRECTORY. Recorded by path alone: hashing a tree
#: means walking it, and the DOCK3 dockfiles directory is shared, large and not
#: ours. Naming it still tells a reader which installation was used.
_INPUT_DIR_KEYS = frozenset({"package_path", "dockfiles_dir"})

#: Keys that name where a run WRITES. Never recorded as inputs.
_OUTPUT_KEYS = frozenset({"output_dir", "log_dir", "cache_dir", "subset_dir"})


def _iter_config_file_candidates(
    value: Any, key: str | None = None
) -> Iterator[tuple[str, bool]]:
    """Yield ``(path, is_directory_key)`` for input-like config values.

    Output destinations are skipped: recording them as inputs would hash this
    run's own artifacts and change the provenance from one run to the next.
    """
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            if child_key in _OUTPUT_KEYS:
                continue
            yield from _iter_config_file_candidates(child_value, child_key)
        return

    if isinstance(value, list):
        for child_value in value:
            yield from _iter_config_file_candidates(child_value, key)
        return

    if not isinstance(value, str):
        return
    if key in _INPUT_FILE_KEYS:
        yield value, False
    elif key in _INPUT_DIR_KEYS:
        yield value, True


def _resolve_existing_path(path_like: str, *, directory: bool = False) -> Path | None:
    """Resolve a config path against the working directory if it exists.

    Symlinks are followed, because the fixed-feature store is a link into
    project space and the target is the input that matters.
    """
    candidate = Path(path_like).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        candidate = candidate.resolve()
    except OSError:
        return None
    if directory:
        return candidate if candidate.is_dir() else None
    return candidate if candidate.is_file() else None


def _resolve_existing_file(path_like: str) -> Path | None:
    """Resolve a config path to an existing file, or ``None``."""
    return _resolve_existing_path(path_like, directory=False)


#: Files above this size are identified by size and mtime rather than a digest.
#: The fixed-feature store is 20.5 GB on a networked filesystem; hashing it at
#: every run start would read all of it for no gain, since nothing rewrites it
#: in place. The threshold is above every other input the AmpC configs name --
#: the largest are the 33 MB MiniMol checkpoint and the 23 MB held-out set --
#: so those are still hashed.
_MAX_HASHED_BYTES = 256 * 1024 * 1024


def _file_record(path: Path) -> dict[str, Any]:
    """Return a path, size and digest record for one input.

    ``md5`` is ``None`` for a directory or an oversized file, and
    ``md5_skipped`` then says why. A null digest must never read as "this input
    was unchanged".
    """
    record: dict[str, Any] = {"path": _display_path(path)}
    try:
        stat = path.stat()
    except OSError as error:
        record["md5"] = None
        record["md5_skipped"] = f"could not stat: {error.strerror or error}"
        return record

    if path.is_dir():
        record["kind"] = "directory"
        record["md5"] = None
        record["md5_skipped"] = "directory; recorded by path only"
        return record

    record["bytes"] = stat.st_size
    record["mtime"] = datetime.fromtimestamp(
        stat.st_mtime, tz=timezone.utc
    ).isoformat()
    if stat.st_size > _MAX_HASHED_BYTES:
        record["md5"] = None
        record["md5_skipped"] = (
            f"larger than {_MAX_HASHED_BYTES} bytes; identified by size and mtime"
        )
        return record
    try:
        record["md5"] = _file_md5(path)
    except OSError as error:
        record["md5"] = None
        record["md5_skipped"] = f"could not read: {error.strerror or error}"
    return record


def _display_path(path: Path) -> str:
    """Prefer repository-relative paths when possible."""
    try:
        return str(path.resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(path.resolve())


def _file_md5(path: Path) -> str:
    """Compute the MD5 digest for a file on disk."""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_md5(payload: Any) -> str:
    """Compute a deterministic MD5 for a JSON-serializable payload."""
    encoded_payload = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.md5(encoded_payload).hexdigest()


def _stable_text_md5(text: str) -> str:
    """Compute the MD5 digest of a string."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
