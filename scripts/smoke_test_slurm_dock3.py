"""Generate 1,000 untrained S3-GFN molecules and dock them through Slurm."""

from __future__ import annotations

import argparse
import csv
import logging
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from activelearning.config import ActiveLearningConfig
from activelearning.oracle.config import (
    CompositeOracleConfig,
    SlurmDock3OracleConfig,
)
from activelearning.runtime import bind_runtime_context
from activelearning.sampler.config import S3GFNSamplerConfig
from activelearning.utils.config_loader import load_and_parse
from activelearning.utils.types import Candidate, Observation

logger = logging.getLogger(__name__)

_SAMPLE_COUNT = 1000
_FIXED_OVERRIDES = [
    f"sampler.n_samples={_SAMPLE_COUNT}",
    "sampler.n_train_steps=0",
    "sampler.performance_mode=eager",
    "sampler.compile_strategy=none",
]


def _select_slurm_oracle_config(
    oracle_config: object,
) -> SlurmDock3OracleConfig:
    """Select the sole Slurm DOCK3 config from a direct or composite oracle."""
    if isinstance(oracle_config, SlurmDock3OracleConfig):
        return oracle_config
    if isinstance(oracle_config, CompositeOracleConfig):
        slurm_configs = [
            sub_oracle
            for sub_oracle in oracle_config.sub_oracles
            if isinstance(sub_oracle, SlurmDock3OracleConfig)
        ]
        if len(slurm_configs) == 1:
            return slurm_configs[0]
        raise ValueError(
            "Smoke test requires exactly one SlurmDock3Oracle sub-oracle; "
            f"found {len(slurm_configs)}"
        )
    raise ValueError(
        "Smoke test requires oracle.type=SlurmDock3Oracle or a composite "
        "oracle containing exactly one SlurmDock3Oracle"
    )


def _set_docking_fidelity(
    candidates: Sequence[Candidate],
    fidelity: int,
) -> list[Candidate]:
    """Make every sampled candidate target the Slurm oracle's fidelity."""
    return [
        Candidate(x=candidate.x, fidelity=fidelity, metadata=candidate.metadata)
        for candidate in candidates
    ]


def _write_results(
    path: Path,
    candidates: Sequence[Candidate],
    observations: Sequence[Observation],
) -> None:
    """Write all sampled molecules and docking outcomes to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "index",
                "smiles",
                "fidelity",
                "y",
                "dock3_raw_score",
                "dock3_pprop",
                "dock3_failure_reason",
            ],
        )
        writer.writeheader()
        for index, (candidate, observation) in enumerate(zip(candidates, observations)):
            metadata = observation.metadata or {}
            writer.writerow(
                {
                    "index": index,
                    "smiles": candidate.x,
                    "fidelity": candidate.fidelity,
                    "y": observation.y,
                    "dock3_raw_score": metadata.get("dock3_raw_score"),
                    "dock3_pprop": metadata.get("dock3_pprop"),
                    "dock3_failure_reason": metadata.get("dock3_failure_reason"),
                }
            )


def run_smoke_test(
    config_path: Path,
    *,
    overrides: Sequence[str],
    output_path: Path,
) -> int:
    """Run generation and distributed docking, returning a process status."""
    config = load_and_parse(
        config_path,
        ActiveLearningConfig,
        overrides=[*overrides, *_FIXED_OVERRIDES],
    )
    if not isinstance(config.sampler, S3GFNSamplerConfig):
        raise TypeError("Smoke test requires sampler.type=S3GFNSampler")
    slurm_oracle_config = _select_slurm_oracle_config(config.oracle)
    docking_fidelity = next(iter(slurm_oracle_config.fidelity_costs))

    sampler_config = config.sampler.model_copy(
        update={"fidelities": [docking_fidelity]}
    )
    sampler = sampler_config.build()
    oracle = slurm_oracle_config.build()
    runtime = config.runtime.build()
    bind_runtime_context([sampler, oracle], runtime)

    logger.info(
        "Generating %d molecules from the fresh pretrained S3-GFN policy",
        _SAMPLE_COUNT,
    )
    candidates = _set_docking_fidelity(
        list(sampler.sample()),
        docking_fidelity,
    )
    if len(candidates) != _SAMPLE_COUNT:
        raise RuntimeError(
            f"S3-GFN returned {len(candidates)} molecules; expected {_SAMPLE_COUNT}"
        )
    if len({candidate.x for candidate in candidates}) != _SAMPLE_COUNT:
        raise RuntimeError("S3-GFN returned duplicate molecules")

    logger.info(
        "Submitting %d molecules to Slurm DOCK3 at fidelity %d",
        len(candidates),
        docking_fidelity,
    )
    observations = list(oracle.query(candidates))
    if len(observations) != len(candidates):
        raise RuntimeError(
            f"DOCK3 returned {len(observations)} observations for "
            f"{len(candidates)} molecules"
        )
    for candidate, observation in zip(candidates, observations):
        if candidate.x != observation.x or candidate.fidelity != observation.fidelity:
            raise RuntimeError("DOCK3 observations do not preserve candidate order")
    _write_results(output_path, candidates, observations)

    reasons = Counter(
        (observation.metadata or {}).get("dock3_failure_reason")
        for observation in observations
    )
    reasons.pop(None, None)
    successful = sum(
        math.isfinite(float(observation.y)) for observation in observations
    )
    infrastructure_failures = sum(
        count for reason, count in reasons.items() if str(reason).startswith("slurm_")
    )
    logger.info(
        "Smoke test complete: %d/%d successful docks; results=%s",
        successful,
        len(observations),
        output_path,
    )
    for reason, count in sorted(reasons.items()):
        logger.info("Failure %s: %d", reason, count)

    if infrastructure_failures:
        logger.error(
            "%d molecule(s) failed because of Slurm infrastructure",
            infrastructure_failures,
        )
        return 2
    if successful == 0:
        logger.error("No molecule docked successfully")
        return 3
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone smoke-test argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        type=Path,
        help="Active-learning YAML with S3GFNSampler and SlurmDock3Oracle.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional OmegaConf key=value overrides for cluster-specific paths.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("slurm_dock3_smoke_results.csv"),
        help="CSV path for all molecules and docking outcomes.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the smoke-test CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)
    raise SystemExit(
        run_smoke_test(
            args.config.resolve(),
            overrides=args.overrides,
            output_path=args.output.resolve(),
        )
    )


if __name__ == "__main__":
    main()
