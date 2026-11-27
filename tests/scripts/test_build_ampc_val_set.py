"""Tests for ``scripts/build_ampc_val_set.py`` against a miniature fake screen."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.build_ampc_val_set import (
    BIN_WIDTH,
    DEFAULT_N_SHARDS,
    OUTPUT_COLUMNS,
    PROVENANCE_COLUMNS,
    STREAM_OFFSET,
    bin_index,
    build_plan,
    load_exclusion,
    main,
    parse_targets,
    read_table,
)

# Four score bins, chosen so that each lands in a distinct pProp bin:
#   pprop 2.60 -> bin 5, 1.20 -> bin 2, 0.70 -> bin 1, 0.20 -> bin 0
# The counts match the fake screen below and sum to a library of 10.
TABLE_TEXT = """score n cumul_n prop cumul_prop pprop
-10.0 1 1 0.1 0.1 2.6
-9.0 2 3 0.2 0.3 1.2
-8.0 3 6 0.3 0.6 0.7
-7.0 4 10 0.4 1.0 0.2
"""

# Bin 0 holds G H I J, bin 1 holds D E F, bin 2 holds B C, bin 5 holds A.
SHARDS = {
    0: (["A", "B", "C", "D", "E"], [-10.0, -9.0, -9.0, -8.0, -8.0]),
    1: (["F", "G", "H", "I", "J"], [-8.0, -7.0, -7.0, -7.0, -7.0]),
}

#: Bins 2 and above are take-all, so only bins 0 and 1 need targets.
TAKE_ALL_FROM = 1.0
TARGETS = "0.0:2,0.5:2"


def _write_shard(
    data_dir: Path, shard: int, smiles: list[str], scores: list[float]
) -> None:
    """Write one fake shard: a ``.smi`` file aligned with a ``.npy`` of scores."""
    (data_dir / f"smi_{shard}.smi").write_text(
        "".join(f"{token} ZINC{index:04d}\n" for index, token in enumerate(smiles)),
        encoding="utf-8",
    )
    np.save(data_dir / f"scores_{shard}.npy", np.asarray(scores, dtype=np.float32))


@pytest.fixture
def screen(tmp_path: Path) -> Path:
    """A two-shard fake screen of ten molecules, matching ``TABLE_TEXT``."""
    data_dir = tmp_path / "screen"
    data_dir.mkdir()
    for shard, (smiles, scores) in SHARDS.items():
        _write_shard(data_dir, shard, smiles, scores)
    return data_dir


@pytest.fixture
def table(tmp_path: Path) -> Path:
    """The miniature score to pProp table."""
    path = tmp_path / "table.df"
    path.write_text(TABLE_TEXT, encoding="utf-8")
    return path


def _exclusion_csv(tmp_path: Path, smiles: list[str], column: str = "SMILE") -> Path:
    """Write an exclusion CSV in the 10M training set's column layout."""
    path = tmp_path / "exclude.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([column, "y", "fidelity"])
        for token in smiles:
            writer.writerow([token, "0.1", "1"])
    return path


def _run(
    tmp_path: Path,
    screen: Path,
    table: Path,
    *,
    exclude: list[str] | None = None,
    seed: int = 0,
    targets: str = TARGETS,
    workers: int = 1,
    out_name: str = "val.csv",
) -> tuple[int, list[dict[str, str]], dict]:
    """Run ``main`` on the fake screen and read back its outputs."""
    out = tmp_path / out_name
    out_dir = tmp_path / f"out_{out_name}"
    exclude_arg = (
        str(_exclusion_csv(tmp_path, exclude)) if exclude is not None else "none"
    )
    code = main(
        [
            "--table",
            str(table),
            "--targets",
            targets,
            "--take-all-from",
            str(TAKE_ALL_FROM),
            "--data-dir",
            str(screen),
            "--n-shards",
            str(len(SHARDS)),
            "--exclude-csv",
            exclude_arg,
            "--out",
            str(out),
            "--out-dir",
            str(out_dir),
            "--workers",
            str(workers),
            "--seed",
            str(seed),
        ]
    )
    with open(out, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    return code, rows, summary


def test_bin_index_uses_half_open_bins() -> None:
    """A pProp sits in the bin its lower edge names, and edges round down."""
    assert bin_index([0.0, 0.49, 0.5, 0.99, 3.5, 5.0]).tolist() == [0, 0, 1, 1, 7, 10]


def test_bin_index_clips_negative_pprop_into_bin_zero() -> None:
    """The rescaled tables bottom out just above zero; -0.0 must not go negative."""
    assert bin_index([-0.0, -1e-9]).tolist() == [0, 0]


def test_read_table_accepts_the_distribution_table_layout(tmp_path: Path) -> None:
    """``target score count ...`` parses as well as ``score n cumul_n ...``."""
    path = tmp_path / "alt.df"
    path.write_text(
        "target score count cumul prop cumul_prop pprop\nAmpC -10.0 1 1 0.1 0.1 2.6\n",
        encoding="utf-8",
    )
    score, count, pprop = read_table(path)
    assert score.tolist() == [-10.0]
    assert count.tolist() == [1]
    assert pprop.tolist() == [2.6]


def test_read_table_rejects_a_missing_column(tmp_path: Path) -> None:
    """A table without a pProp column is an error, not a silent NaN."""
    path = tmp_path / "bad.df"
    path.write_text("score n cumul_n\n-10.0 1 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pprop"):
        read_table(path)


@pytest.mark.parametrize(
    "spec",
    ["", "0.0", "0.0:0", "0.0:-1", "0.3:5", "0.0:5,0.0:5", "0.0:x"],
)
def test_parse_targets_rejects_invalid_specs(spec: str) -> None:
    """Malformed, non-positive, off-grid and repeated targets all raise."""
    with pytest.raises(ValueError):
        parse_targets(spec)


def test_parse_targets_maps_edges_to_bin_indices() -> None:
    """An edge is divided by the bin width to get its index."""
    assert parse_targets("0.0:2,0.5:3,3.5:4") == {0: 2, 1: 3, 7: 4}


def test_build_plan_rejects_a_target_on_a_take_all_bin(table: Path) -> None:
    """A take-all bin must not also carry a target."""
    with pytest.raises(ValueError, match="take-all"):
        build_plan(table, {5: 1}, TAKE_ALL_FROM, 1.05)


def test_build_plan_reads_exact_bin_populations(table: Path) -> None:
    """Populations come from the table's count column, summed per bin."""
    plan, _, _ = build_plan(table, parse_targets(TARGETS), TAKE_ALL_FROM, 1.05)
    assert plan.library_size == 10
    assert plan.population[0] == 4
    assert plan.population[1] == 3
    assert plan.population[2] == 2
    assert plan.population[5] == 1
    assert plan.is_take_all(2)
    assert plan.is_take_all(5)
    assert not plan.is_take_all(0)


def test_load_exclusion_requires_its_column(tmp_path: Path) -> None:
    """The 10M set names the column SMILE; a wrong name is an error."""
    path = _exclusion_csv(tmp_path, ["A"])
    with pytest.raises(ValueError, match="SMILES"):
        load_exclusion(path, "SMILES")


@pytest.mark.parametrize("workers", [1, 2])
def test_builds_the_requested_rows_per_bin(
    tmp_path: Path, screen: Path, table: Path, workers: int
) -> None:
    """Subsampled bins hit their target exactly; take-all bins keep everything."""
    code, rows, summary = _run(tmp_path, screen, table, workers=workers)
    assert code == 0, summary["checks"]
    per_bin = {entry["bin"]: entry for entry in summary["per_bin"]}
    assert per_bin[0.0]["n_retained"] == 2
    assert per_bin[0.5]["n_retained"] == 2
    assert per_bin[1.0]["n_retained"] == 2
    assert per_bin[2.5]["n_retained"] == 1
    assert len(rows) == 7


def test_ipw_times_count_recovers_the_eligible_population(
    tmp_path: Path, screen: Path, table: Path
) -> None:
    """With nothing excluded the weights must sum to the whole library."""
    _, rows, summary = _run(tmp_path, screen, table)
    assert summary["weight_sum"] == pytest.approx(10.0)
    assert summary["checks"]["weight_sum_matches_eligible_library"]
    per_bin = {entry["bin"]: entry for entry in summary["per_bin"]}
    # Bin 0 keeps 2 of 4, so each row stands for two molecules.
    assert per_bin[0.0]["ipw"] == pytest.approx(2.0)
    assert per_bin[0.5]["ipw"] == pytest.approx(1.5)
    # Take-all bins keep every molecule, so their weight is 1.
    assert per_bin[1.0]["ipw"] == pytest.approx(1.0)
    assert per_bin[2.5]["ipw"] == pytest.approx(1.0)


def test_output_rows_carry_the_bin_and_ipw_of_their_pprop(
    tmp_path: Path, screen: Path, table: Path
) -> None:
    """Each row's bin is its pProp's lower edge, and pProp matches the table."""
    _, rows, _ = _run(tmp_path, screen, table)
    assert tuple(rows[0].keys()) == OUTPUT_COLUMNS
    pprop_of_bin = {"0.0": 0.2, "0.5": 0.7, "1.0": 1.2, "2.5": 2.6}
    for row in rows:
        assert float(row["pprop"]) == pytest.approx(pprop_of_bin[row["bin"]])
        assert float(row["bin"]) == pytest.approx(
            np.floor(float(row["pprop"]) / BIN_WIDTH) * BIN_WIDTH
        )


def test_excluded_molecules_never_appear(
    tmp_path: Path, screen: Path, table: Path
) -> None:
    """Anything in the exclusion set is dropped, and the take-all bin shrinks."""
    code, rows, summary = _run(tmp_path, screen, table, exclude=["B", "G"])
    assert code == 0, summary["checks"]
    assert {row["SMILES"] for row in rows}.isdisjoint({"B", "G"})
    per_bin = {entry["bin"]: entry for entry in summary["per_bin"]}
    # Bin 1.0 is take-all and held B and C, so only C survives.
    assert per_bin[1.0]["n_retained"] == 1
    assert per_bin[1.0]["n_excluded"] == 1
    # Its one row now stands for the one remaining eligible molecule.
    assert per_bin[1.0]["ipw"] == pytest.approx(1.0)


def test_a_bin_that_cannot_reach_its_target_raises(
    tmp_path: Path, screen: Path, table: Path
) -> None:
    """Excluding a whole bin is a hard error, never a silent undershoot."""
    with pytest.raises(ValueError, match="oversample"):
        _run(tmp_path, screen, table, exclude=["G", "H", "I", "J"])


def test_duplicate_smiles_are_deduped_and_lift_the_take_all_weight(
    tmp_path: Path, table: Path
) -> None:
    """A library duplicate yields one row whose ipw covers both instances."""
    data_dir = tmp_path / "dup_screen"
    data_dir.mkdir()
    # B appears twice at the same score, as genuine library duplicates do.
    _write_shard(
        data_dir, 0, ["A", "B", "B", "D", "E"], [-10.0, -9.0, -9.0, -8.0, -8.0]
    )
    _write_shard(data_dir, 1, ["F", "G", "H", "I", "J"], [-8.0, -7.0, -7.0, -7.0, -7.0])
    code, rows, summary = _run(tmp_path, data_dir, table)
    assert code == 0, summary["checks"]
    per_bin = {entry["bin"]: entry for entry in summary["per_bin"]}
    assert per_bin[1.0]["n_retained"] == 1
    assert per_bin[1.0]["n_duplicate"] == 1
    assert per_bin[1.0]["ipw"] == pytest.approx(2.0)
    assert [row["SMILES"] for row in rows].count("B") == 1
    # Deduping must not lose the duplicate's weight.
    assert summary["weight_sum"] == pytest.approx(10.0)


def test_the_draw_is_seeded(tmp_path: Path, screen: Path, table: Path) -> None:
    """The same seed reproduces the row set; some other seed changes it."""
    _, first, _ = _run(tmp_path, screen, table, seed=7, out_name="a.csv")
    _, again, _ = _run(tmp_path, screen, table, seed=7, out_name="b.csv")
    assert [row["SMILES"] for row in first] == [row["SMILES"] for row in again]

    baseline = {row["SMILES"] for row in first}
    differs = False
    for seed in range(1, 12):
        if seed == 7:
            continue
        _, other, _ = _run(tmp_path, screen, table, seed=seed, out_name=f"s{seed}.csv")
        if {row["SMILES"] for row in other} != baseline:
            differs = True
            break
    assert differs, "no seed in 1..11 changed the sample"


def test_main_writes_the_csv_provenance_and_summary(
    tmp_path: Path, screen: Path, table: Path
) -> None:
    """All three artifacts appear, row-aligned, with no .tmp left behind."""
    code, rows, summary = _run(tmp_path, screen, table)
    assert code == 0
    out = tmp_path / "val.csv"
    provenance = tmp_path / "val_provenance.csv"
    assert out.exists()
    assert provenance.exists()
    assert not list(tmp_path.glob("*.tmp"))
    with open(provenance, newline="", encoding="utf-8") as handle:
        prov_rows = list(csv.DictReader(handle))
    assert tuple(prov_rows[0].keys()) == PROVENANCE_COLUMNS
    assert len(prov_rows) == len(rows) == summary["n_rows"]
    assert [int(row["row"]) for row in prov_rows] == list(range(len(rows)))


def test_provenance_points_at_the_recorded_smiles(
    tmp_path: Path, screen: Path, table: Path
) -> None:
    """``(shard, local_idx)`` really locates each row's molecule in the screen."""
    _, rows, _ = _run(tmp_path, screen, table)
    with open(tmp_path / "val_provenance.csv", newline="", encoding="utf-8") as handle:
        prov_rows = list(csv.DictReader(handle))
    lines = {
        shard: (screen / f"smi_{shard}.smi").read_text(encoding="utf-8").splitlines()
        for shard in SHARDS
    }
    for row, prov in zip(rows, prov_rows):
        shard = int(prov["shard"])
        local_idx = int(prov["local_idx"])
        assert lines[shard][local_idx].split()[0] == row["SMILES"]


def test_key_stream_cannot_coincide_with_the_10m_builder() -> None:
    """Our seeding must not reproduce ``build_unif_random.py``'s stream.

    That builder used ``default_rng(seed + shard)`` with seed 0 over shards
    0..293. Sharing its stream made our candidates preferentially *be* the 10M's
    picks, so exclusion ran ~30x the uniform rate.
    """
    builder_seeds = set(range(DEFAULT_N_SHARDS))
    our_seeds = {shard + STREAM_OFFSET for shard in range(DEFAULT_N_SHARDS)}
    assert not (our_seeds & builder_seeds)
    for shard in (0, 1, DEFAULT_N_SHARDS - 1):
        ours = np.random.default_rng(shard + STREAM_OFFSET).random(8)
        builders = np.random.default_rng(shard).random(8)
        assert not np.array_equal(ours, builders)


def test_entropy_tuple_seeding_would_not_separate_the_streams() -> None:
    """Pins why ``STREAM_OFFSET`` exists rather than ``default_rng([seed, shard])``.

    ``SeedSequence`` absorbs trailing zeros, so the entropy tuple ``[0, 0]``
    hashes to the same pool as the scalar ``0`` and shard 0 still collides with
    the 10M builder. This is the trap the offset avoids; do not "simplify" back.
    """
    assert np.array_equal(
        np.random.default_rng([0, 0]).random(4), np.random.default_rng(0).random(4)
    )


def test_a_bins_draw_is_independent_of_other_bins_targets(
    tmp_path: Path, screen: Path, table: Path
) -> None:
    """Keys depend on (seed, shard, row) only, never on the bin layout.

    Drawing keys per bin made a row's key position depend on how many rows the
    lower bins held, which is what aligned our draw with the 10M's. Drawing once
    per shard and indexing per bin removes that coupling, so changing one bin's
    target must leave every other bin's rows untouched.
    """
    _, baseline, _ = _run(
        tmp_path, screen, table, targets="0.0:2,0.5:2", out_name="a.csv"
    )
    _, changed, _ = _run(
        tmp_path, screen, table, targets="0.0:2,0.5:1", out_name="b.csv"
    )

    def rows_of_bin(rows: list[dict[str, str]], edge: str) -> list[str]:
        return sorted(row["SMILES"] for row in rows if row["bin"] == edge)

    assert rows_of_bin(baseline, "0.0") == rows_of_bin(changed, "0.0")
    assert rows_of_bin(baseline, "1.0") == rows_of_bin(changed, "1.0")
    assert rows_of_bin(baseline, "2.5") == rows_of_bin(changed, "2.5")
    # The bin whose target actually changed should shrink.
    assert len(rows_of_bin(changed, "0.5")) == 1
