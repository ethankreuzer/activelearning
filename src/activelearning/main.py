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
            metadata=_build_run_metadata(
                config_paths=config_paths,
                overrides=overrides,
                raw_cfg=raw_cfg,
                cfg=cfg,
            )
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
            "resolved_config_md5": _stable_md5(resolved_config),
            "input_files": _collect_input_files(config_paths, resolved_config),
        },
    }


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

    for candidate in _iter_config_file_candidates(resolved_config):
        resolved_candidate = _resolve_existing_file(candidate)
        if resolved_candidate is None or resolved_candidate in seen_paths:
            continue
        recorded_files.append(_file_record(resolved_candidate))
        seen_paths.add(resolved_candidate)

    return recorded_files


def _iter_config_file_candidates(value: Any, key: str | None = None) -> Iterator[str]:
    """Yield file-like config values while skipping known output paths."""
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            if child_key in {"output_dir", "log_dir"}:
                continue
            yield from _iter_config_file_candidates(child_value, child_key)
        return

    if isinstance(value, list):
        for child_value in value:
            yield from _iter_config_file_candidates(child_value, key)
        return

    if isinstance(value, str) and key in {"path", "config_path", "file", "file_path"}:
        yield value


def _resolve_existing_file(path_like: str) -> Path | None:
    """Resolve a config path against the current working directory if it exists."""
    candidate = Path(path_like).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    if candidate.is_file():
        return candidate
    return None


def _file_record(path: Path) -> dict[str, str]:
    """Return a stable path/hash record for one file."""
    return {"path": _display_path(path), "md5": _file_md5(path)}


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


if __name__ == "__main__":
    main()
