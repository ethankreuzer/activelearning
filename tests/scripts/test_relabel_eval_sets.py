"""Tests for the eval-set relabelling script.

The script's contract is narrow and the tests hold it to exactly that: **same
rows, same order, corrected labels, taken verbatim from the corrected library.**
Everything else is a refusal.

Two hazards get explicit coverage because both are silent failures rather than
crashes:

* These CSVs do not share a line terminator -- the cache files are CRLF from
  ``csv.writer`` while the library is LF from ``pandas.to_csv`` -- so a writer
  that re-emits rows would rewrite the endings of half of them.
* The old label floor is stored in the tree as two different literals that parse
  to different doubles, so an equality test for it misses one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import relabel_eval_sets as rl

# The two literals the old floor is stored as. They differ in the 16th digit and
# parse to different doubles, which is why the floor check uses a tolerance.
FLOOR_17 = "0.03182123035321084"
FLOOR_16 = "0.0318212303532108"


def _write(path: Path, header: str, rows: list[str], terminator: str = "\r\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = terminator.join([header, *rows]) + terminator
    path.write_bytes(body.encode("utf-8"))


def _write_library(path: Path, rows: list[tuple[str, str]]) -> None:
    """Write a library stand-in, LF-terminated like the real one."""
    lines = ["SMILE,y,fidelity"] + [f"{s},{y},1" for s, y in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def _library_rows(n: int = 12) -> list[tuple[str, str]]:
    """Rows whose first y is the literal the library guard checks for."""
    rows = [("C" * 1, rl.LIBRARY_ROW1_Y.decode())]
    rows += [(f"C{'O' * i}", repr(0.001 * (i + 1))) for i in range(1, n)]
    return rows


@pytest.fixture
def relaxed_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the library guard at a small stand-in instead of the real 10M file.

    The real expectations (10M rows, 116 hits, 3,653 enriched) cannot be met by a
    fixture, so the counts are rebound. The *mechanism* of the guard is tested
    separately in :class:`TestLibraryGuard`.
    """
    monkeypatch.setattr(rl, "LIBRARY_EXPECTED_ROWS", 12)
    monkeypatch.setattr(rl, "LIBRARY_EXPECTED_HITS", 0)
    monkeypatch.setattr(rl, "LIBRARY_EXPECTED_ENRICHED", 0)
    monkeypatch.setattr(rl, "LIBRARY_EXPECTED_MEAN", 0.0055)
    monkeypatch.setattr(rl, "LIBRARY_EXPECTED_MEAN_TOLERANCE", 1.0)
    # The first row carries LIBRARY_ROW1_Y, which is larger than the rest.
    monkeypatch.setattr(rl, "LIBRARY_EXPECTED_MAX", float(rl.LIBRARY_ROW1_Y))
    monkeypatch.setattr(rl, "LIBRARY_EXPECTED_DISTINCT_Y", 12)


@pytest.fixture
def spec(tmp_path: Path) -> rl.RelabelSpec:
    return rl.RelabelSpec(
        name="toy",
        path=tmp_path / "toy.csv",
        smiles_column="SMILES",
        label_column="y",
        extra_columns=(),
        status=rl.SetStatus.RELABEL,
    )


class TestFloorLiterals:
    """The premise behind using a tolerance for the old-floor check."""

    def test_the_two_stored_literals_are_different_doubles(self) -> None:
        assert float(FLOOR_17) != float(FLOOR_16)
        assert float(FLOOR_17) == rl.OLD_FLOOR_Y

    @pytest.mark.parametrize("literal", [FLOOR_17, FLOOR_16])
    def test_both_are_caught_by_the_tolerance(self, literal: str) -> None:
        assert abs(float(literal) - rl.OLD_FLOOR_Y) <= rl.OLD_FLOOR_TOLERANCE


class TestByteLevelCsv:
    """Reading and splicing must not reformat anything but the label."""

    @pytest.mark.parametrize("terminator", ["\r\n", "\n"])
    def test_terminator_is_preserved(
        self, spec: rl.RelabelSpec, terminator: str
    ) -> None:
        _write(spec.path, "SMILES,y", ["CC,0.5"], terminator=terminator)
        csv_file = rl.read_csv_bytes(spec.path, spec)
        spliced = rl.splice_field(csv_file.lines[0], 1, b"0.25")
        assert spliced == f"CC,0.25{terminator}".encode()

    def test_splice_touches_only_the_named_field(self, spec: rl.RelabelSpec) -> None:
        line = b"CC,0.5,1.0,keep\r\n"
        assert rl.splice_field(line, 1, b"9") == b"CC,9,1.0,keep\r\n"

    def test_undeclared_column_aborts(self, spec: rl.RelabelSpec) -> None:
        _write(spec.path, "SMILES,y,extra", ["CC,0.5,1"])
        with pytest.raises(SystemExit, match="expected"):
            rl.read_csv_bytes(spec.path, spec)

    def test_missing_column_aborts(self, spec: rl.RelabelSpec) -> None:
        _write(spec.path, "SMILES", ["CC"])
        with pytest.raises(SystemExit, match="expected"):
            rl.read_csv_bytes(spec.path, spec)

    def test_unexpected_comma_count_aborts(self, spec: rl.RelabelSpec) -> None:
        _write(spec.path, "SMILES,y", ["CC,0.5", "C,C,0.5"])
        with pytest.raises(SystemExit, match="line 3 has 2 commas"):
            rl.read_csv_bytes(spec.path, spec)

    def test_missing_file_aborts(self, spec: rl.RelabelSpec) -> None:
        with pytest.raises(SystemExit, match="file not found"):
            rl.read_csv_bytes(spec.path, spec)

    def test_unparseable_label_becomes_nan(self) -> None:
        assert rl.parse_label(b"nan") != rl.parse_label(b"nan")
        assert rl.parse_label(b"") != rl.parse_label(b"")
        assert rl.parse_label(b"0.25") == 0.25


class TestLibraryGuard:
    """The guard is what stops a second generation of wrong labels."""

    def test_corrected_stand_in_passes(
        self, tmp_path: Path, relaxed_guard: None
    ) -> None:
        path = tmp_path / "lib.csv"
        _write_library(path, _library_rows())
        _, stats = rl.scan_library(path, set())
        rl.check_library(stats)
        assert stats.rows == 12

    def test_old_library_is_rejected_on_the_first_row(self, tmp_path: Path) -> None:
        path = tmp_path / "lib.csv"
        rows = _library_rows()
        rows[0] = (rows[0][0], "0.04817457063353419")
        _write_library(path, rows)
        with pytest.raises(SystemExit, match="OLD, WRONG library"):
            rl.scan_library(path, set())

    def test_wrong_header_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "lib.csv"
        path.write_bytes(b"SMILES,y\nCC,0.5\n")
        with pytest.raises(SystemExit, match="expected"):
            rl.scan_library(path, set())

    def test_hit_count_off_by_one_aborts(
        self, tmp_path: Path, relaxed_guard: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(rl, "LIBRARY_EXPECTED_HITS", 1)
        path = tmp_path / "lib.csv"
        _write_library(path, _library_rows())
        _, stats = rl.scan_library(path, set())
        with pytest.raises(SystemExit, match="hits 0 != 1"):
            rl.check_library(stats)

    def test_a_library_label_at_the_old_floor_aborts(
        self, tmp_path: Path, relaxed_guard: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(rl, "LIBRARY_EXPECTED_DISTINCT_Y", 12)
        path = tmp_path / "lib.csv"
        rows = _library_rows()
        rows[3] = (rows[3][0], FLOOR_16)
        _write_library(path, rows)
        _, stats = rl.scan_library(path, set())
        with pytest.raises(SystemExit, match="old floor"):
            rl.check_library(stats)

    def test_mean_is_compared_with_a_tolerance(
        self, tmp_path: Path, relaxed_guard: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "lib.csv"
        _write_library(path, _library_rows())
        _, stats = rl.scan_library(path, set())
        monkeypatch.setattr(rl, "LIBRARY_EXPECTED_MEAN", stats.mean_y + 1e-18)
        monkeypatch.setattr(rl, "LIBRARY_EXPECTED_MEAN_TOLERANCE", 1e-15)
        rl.check_library(stats)
        monkeypatch.setattr(rl, "LIBRARY_EXPECTED_MEAN", stats.mean_y + 1e-9)
        with pytest.raises(SystemExit, match="mean y"):
            rl.check_library(stats)

    def test_duplicate_molecules_keep_the_larger_label(self, tmp_path: Path) -> None:
        """The convention build_svgp_scaling_sets already uses for this.

        896 SMILES occur twice in the real library and 196 of those disagree, so
        the join needs a rule. Taking the better observation matches an objective
        about finding the best molecules.
        """
        path = tmp_path / "lib.csv"
        rows = _library_rows()
        rows.append(("CCO", "0.002"))
        rows.append(("CCO", "0.009"))
        rows.append(("CCN", "0.009"))
        rows.append(("CCN", "0.002"))
        _write_library(path, rows)
        labels, stats = rl.scan_library(path, {b"CCO", b"CCN"})
        assert labels[b"CCO"] == b"0.009"
        assert labels[b"CCN"] == b"0.009"
        assert stats.n_duplicate_needed == 2
        assert stats.n_conflicting_needed == 2

    def test_duplicates_with_identical_labels_are_not_conflicts(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "lib.csv"
        rows = _library_rows() + [("CCO", "0.002"), ("CCO", "0.002")]
        _write_library(path, rows)
        _, stats = rl.scan_library(path, {b"CCO"})
        assert stats.n_duplicate_needed == 1
        assert stats.n_conflicting_needed == 0

    def test_only_needed_smiles_are_retained(self, tmp_path: Path) -> None:
        path = tmp_path / "lib.csv"
        _write_library(path, _library_rows())
        labels, stats = rl.scan_library(path, {b"CO"})
        assert set(labels) == {b"CO"}
        assert stats.rows == 12


class TestJoin:
    """The join is exact-string on raw bytes, by design."""

    def _plan(
        self, tmp_path: Path, rows: list[str], library: dict[bytes, bytes]
    ) -> rl.SetPlan:
        spec = rl.RelabelSpec(
            name="toy",
            path=tmp_path / "toy.csv",
            smiles_column="SMILES",
            label_column="y",
            extra_columns=(),
            status=rl.SetStatus.RELABEL,
        )
        _write(spec.path, "SMILES,y", rows)
        return rl.plan_set(spec, rl.read_csv_bytes(spec.path, spec), library)

    def test_stereochemistry_is_not_canonicalized_away(self, tmp_path: Path) -> None:
        """A relabel that canonicalized would attach the wrong molecule's label.

        The feature cache's ``input_sha256`` and the feature store both verify the
        raw strings, and this repo's RDKit calls strip stereochemistry, so the two
        must never be mixed.
        """
        plan = self._plan(
            tmp_path,
            ["C[C@H](N)O,0.5"],
            {b"C[C@@H](N)O": b"0.001"},
        )
        assert plan.matched == 0
        assert plan.unmatched == 1

    def test_matched_rows_take_the_library_bytes_verbatim(self, tmp_path: Path) -> None:
        plan = self._plan(tmp_path, ["CC,0.5"], {b"CC": b"1.2345678901234567e-05"})
        assert rl.field_bytes(plan.new_lines[0], 1) == b"1.2345678901234567e-05"

    def test_counts_are_reported(self, tmp_path: Path) -> None:
        plan = self._plan(tmp_path, ["CC,0.5", "CO,0.5"], {b"CC": b"0.001"})
        assert (plan.matched, plan.unmatched) == (1, 1)
        assert plan.match_rate == 0.5


class TestChecks:
    """Everything is checked before anything is written."""

    def _plan_and_check(
        self,
        tmp_path: Path,
        rows: list[str],
        library: dict[bytes, bytes],
        **spec_kwargs: object,
    ) -> list[str]:
        spec = rl.RelabelSpec(
            name="toy",
            path=tmp_path / "toy.csv",
            smiles_column="SMILES",
            label_column="y",
            extra_columns=(),
            status=rl.SetStatus.RELABEL,
            **spec_kwargs,  # type: ignore[arg-type]
        )
        _write(spec.path, "SMILES,y", rows)
        csv_file = rl.read_csv_bytes(spec.path, spec)
        plan = rl.plan_set(spec, csv_file, library)
        return rl.check_plan(plan, set(library.values()), tmp_path)

    def test_a_clean_relabel_has_no_problems(self, tmp_path: Path) -> None:
        assert self._plan_and_check(tmp_path, ["CC,0.5"], {b"CC": b"0.001"}) == []

    def test_unchanged_label_aborts(self, tmp_path: Path) -> None:
        """A relabel is a correction; a no-op means the join or the spec is wrong.

        This is the active safety net behind the ``SKIP_ALREADY_CORRECT`` status:
        pointing the script at an already-correct file trips here.
        """
        problems = self._plan_and_check(tmp_path, ["CC,0.001"], {b"CC": b"0.001"})
        assert any("kept the same label" in p for p in problems)

    def test_forbid_policy_rejects_any_miss(self, tmp_path: Path) -> None:
        problems = self._plan_and_check(
            tmp_path, ["CC,0.5", "CO,0.5"], {b"CC": b"0.001"}
        )
        assert any("not in the library" in p for p in problems)

    def test_nan_policy_permits_exactly_the_misses(self, tmp_path: Path) -> None:
        problems = self._plan_and_check(
            tmp_path,
            ["CC,0.5", "CO,0.5"],
            {b"CC": b"0.001"},
            unmatched=rl.UnmatchedPolicy.NAN,
            min_match_rate=0.5,
        )
        assert problems == []

    def test_match_rate_below_the_floor_aborts(self, tmp_path: Path) -> None:
        problems = self._plan_and_check(
            tmp_path,
            ["CC,0.5", "CO,0.5"],
            {b"CC": b"0.001"},
            unmatched=rl.UnmatchedPolicy.NAN,
            min_match_rate=0.9,
        )
        assert any("match rate" in p for p in problems)

    def test_expected_row_count_mismatch_aborts(self, tmp_path: Path) -> None:
        problems = self._plan_and_check(
            tmp_path, ["CC,0.5"], {b"CC": b"0.001"}, expected_rows=2
        )
        assert any("row count 1 != expected 2" in p for p in problems)

    @pytest.mark.parametrize("literal", [FLOOR_17, FLOOR_16])
    def test_an_output_label_at_the_old_floor_aborts(
        self, tmp_path: Path, literal: str
    ) -> None:
        """Both stored literals must be caught, which is why the test is paired."""
        problems = self._plan_and_check(tmp_path, ["CC,0.5"], {b"CC": literal.encode()})
        assert any("old floor" in p for p in problems)

    def test_a_label_absent_from_the_library_aborts(self, tmp_path: Path) -> None:
        """Proves the labels came from the library file and nowhere else."""
        spec = rl.RelabelSpec(
            name="toy",
            path=tmp_path / "toy.csv",
            smiles_column="SMILES",
            label_column="y",
            extra_columns=(),
            status=rl.SetStatus.RELABEL,
        )
        _write(spec.path, "SMILES,y", ["CC,0.5"])
        csv_file = rl.read_csv_bytes(spec.path, spec)
        plan = rl.plan_set(spec, csv_file, {b"CC": b"0.001"})
        problems = rl.check_plan(plan, {b"0.002"}, tmp_path)
        assert any("not values present in the library" in p for p in problems)

    def test_feature_cache_hash_is_verified(self, tmp_path: Path) -> None:
        """Row order is what keeps the .npy valid, so the hash is re-derived."""
        spec = rl.RelabelSpec(
            name="toy",
            path=tmp_path / "toy.csv",
            smiles_column="SMILES",
            label_column="y",
            extra_columns=(),
            status=rl.SetStatus.RELABEL,
            cache_stem="toy",
        )
        _write(spec.path, "SMILES,y", ["CC,0.5", "CO,0.5"])
        (tmp_path / "toy.npy.json").write_text(
            json.dumps({"row_count": 2, "input_sha256": "deadbeef"})
        )
        csv_file = rl.read_csv_bytes(spec.path, spec)
        plan = rl.plan_set(spec, csv_file, {b"CC": b"0.001", b"CO": b"0.002"})
        problems = rl.check_plan(plan, {b"0.001", b"0.002"}, tmp_path)
        assert any("input_sha256" in p for p in problems)

    def test_matching_cache_hash_passes(self, tmp_path: Path) -> None:
        from activelearning.applications.molecules.minimol_encoder import (
            _hash_ordered_strings,
        )

        spec = rl.RelabelSpec(
            name="toy",
            path=tmp_path / "toy.csv",
            smiles_column="SMILES",
            label_column="y",
            extra_columns=(),
            status=rl.SetStatus.RELABEL,
            cache_stem="toy",
        )
        _write(spec.path, "SMILES,y", ["CC,0.5", "CO,0.5"])
        (tmp_path / "toy.npy.json").write_text(
            json.dumps(
                {
                    "row_count": 2,
                    "input_sha256": _hash_ordered_strings(["CC", "CO"]),
                }
            )
        )
        csv_file = rl.read_csv_bytes(spec.path, spec)
        plan = rl.plan_set(spec, csv_file, {b"CC": b"0.001", b"CO": b"0.002"})
        assert rl.check_plan(plan, {b"0.001", b"0.002"}, tmp_path) == []

    def test_cache_row_count_mismatch_aborts(self, tmp_path: Path) -> None:
        spec = rl.RelabelSpec(
            name="toy",
            path=tmp_path / "toy.csv",
            smiles_column="SMILES",
            label_column="y",
            extra_columns=(),
            status=rl.SetStatus.RELABEL,
            cache_stem="toy",
        )
        _write(spec.path, "SMILES,y", ["CC,0.5"])
        (tmp_path / "toy.npy.json").write_text(json.dumps({"row_count": 7}))
        csv_file = rl.read_csv_bytes(spec.path, spec)
        plan = rl.plan_set(spec, csv_file, {b"CC": b"0.001"})
        problems = rl.check_plan(plan, {b"0.001"}, tmp_path)
        assert any("row_count 7" in p for p in problems)


class TestVerifySkipped:
    """A declared reason that is never tested is a comment, not a check."""

    def _spec(self, tmp_path: Path, status: rl.SetStatus) -> rl.RelabelSpec:
        return rl.RelabelSpec(
            name="toy",
            path=tmp_path / "toy.csv",
            smiles_column="SMILES",
            label_column="y",
            extra_columns=(),
            status=status,
        )

    def test_unlabelled_set_with_all_nan_passes(self, tmp_path: Path) -> None:
        spec = self._spec(tmp_path, rl.SetStatus.SKIP_UNLABELLED)
        _write(spec.path, "SMILES,y", ["CC,nan", "CO,nan"])
        assert rl.verify_skipped(spec, rl.read_csv_bytes(spec.path, spec)) == []

    def test_unlabelled_set_with_a_real_label_aborts(self, tmp_path: Path) -> None:
        spec = self._spec(tmp_path, rl.SetStatus.SKIP_UNLABELLED)
        _write(spec.path, "SMILES,y", ["CC,nan", "CO,0.5"])
        problems = rl.verify_skipped(spec, rl.read_csv_bytes(spec.path, spec))
        assert any("declared unlabelled" in p for p in problems)

    def test_already_correct_set_passes(self, tmp_path: Path) -> None:
        spec = self._spec(tmp_path, rl.SetStatus.SKIP_ALREADY_CORRECT)
        _write(spec.path, "SMILES,y", ["CC,0.5"])
        assert rl.verify_skipped(spec, rl.read_csv_bytes(spec.path, spec)) == []

    @pytest.mark.parametrize("literal", [FLOOR_17, FLOOR_16])
    def test_already_correct_set_at_the_old_floor_aborts(
        self, tmp_path: Path, literal: str
    ) -> None:
        spec = self._spec(tmp_path, rl.SetStatus.SKIP_ALREADY_CORRECT)
        _write(spec.path, "SMILES,y", [f"CC,{literal}"])
        problems = rl.verify_skipped(spec, rl.read_csv_bytes(spec.path, spec))
        assert any("old floor" in p for p in problems)


class TestWriting:
    def _plan(self, tmp_path: Path) -> rl.SetPlan:
        spec = rl.RelabelSpec(
            name="toy",
            path=tmp_path / "toy.csv",
            smiles_column="SMILES",
            label_column="y",
            extra_columns=("weight",),
            status=rl.SetStatus.RELABEL,
        )
        _write(spec.path, "SMILES,y,weight", ["CC,0.5,1.0", "CO,0.5,2"])
        csv_file = rl.read_csv_bytes(spec.path, spec)
        return rl.plan_set(spec, csv_file, {b"CC": b"0.001", b"CO": b"0.002"})

    def test_only_the_label_bytes_change(self, tmp_path: Path) -> None:
        plan = self._plan(tmp_path)
        original = plan.csv_file.path.read_bytes()
        rl.write_set(plan, overwrite=False)
        written = plan.csv_file.path.read_bytes()
        assert written == original.replace(b",0.5,", b",0.001,", 1).replace(
            b",0.5,", b",0.002,", 1
        )

    def test_extras_are_carried_byte_for_byte(self, tmp_path: Path) -> None:
        """``weight`` is 1/P(selected) for the rule actually applied.

        Reparsing it would silently turn ``1.0`` into ``1.0`` and ``2`` into
        ``2.0``; an IPW estimator depends on it being left exactly as written.
        """
        plan = self._plan(tmp_path)
        rl.write_set(plan, overwrite=False)
        lines = plan.csv_file.path.read_bytes().splitlines()
        assert lines[1].endswith(b",1.0")
        assert lines[2].endswith(b",2")

    def test_the_old_file_is_archived(self, tmp_path: Path) -> None:
        plan = self._plan(tmp_path)
        original = plan.csv_file.path.read_bytes()
        archive = rl.write_set(plan, overwrite=False)
        assert archive.name == f"toy{rl.BACKUP_SUFFIX}.csv"
        assert archive.read_bytes() == original

    def test_a_second_run_refuses_to_clobber_the_archive(self, tmp_path: Path) -> None:
        plan = self._plan(tmp_path)
        rl.write_set(plan, overwrite=False)
        with pytest.raises(SystemExit, match="already exists"):
            rl.write_set(plan, overwrite=False)

    def test_overwrite_permits_it(self, tmp_path: Path) -> None:
        plan = self._plan(tmp_path)
        rl.write_set(plan, overwrite=False)
        rl.write_set(plan, overwrite=True)

    def test_no_temporary_file_survives(self, tmp_path: Path) -> None:
        plan = self._plan(tmp_path)
        rl.write_set(plan, overwrite=False)
        assert list(tmp_path.glob("*.tmp")) == []


class TestCli:
    def test_list_sets_exits_cleanly(self, capsys: pytest.CaptureFixture[str]) -> None:
        rl.main(["--list-sets"])
        out = capsys.readouterr().out
        assert "train_random" in out
        assert "skip_no_scores" in out

    def test_requesting_a_skipped_set_is_refused(self) -> None:
        with pytest.raises(SystemExit, match="not re-labellable"):
            rl.resolve_specs(
                ["gp_molformer_set"],
                cache_dir=Path("cache"),
                data_dir=Path("data"),
            )

    def test_unknown_set_is_refused(self) -> None:
        with pytest.raises(SystemExit, match="unknown set"):
            rl.resolve_specs(["nope"], cache_dir=Path("cache"), data_dir=Path("data"))

    def test_default_selection_is_only_relabellable_sets(self) -> None:
        selected, skipped = rl.resolve_specs(
            None, cache_dir=Path("cache"), data_dir=Path("data")
        )
        assert [s.name for s in selected] == ["train_random"]
        assert all(s.status is not rl.SetStatus.RELABEL for s in skipped)

    def test_paths_are_rebased_onto_the_requested_directories(
        self, tmp_path: Path
    ) -> None:
        selected, skipped = rl.resolve_specs(
            None, cache_dir=tmp_path / "c", data_dir=tmp_path / "d"
        )
        assert selected[0].path == tmp_path / "c" / "train_random.csv"
        by_name = {s.name: s for s in skipped}
        assert (
            by_name["data_ampc_top_2000"].path == tmp_path / "d" / "ampc_top_2000.csv"
        )

    def test_every_declared_set_has_a_reason_unless_relabelled(self) -> None:
        for spec in rl.RELABEL_SETS:
            if spec.status is rl.SetStatus.RELABEL:
                continue
            assert spec.skip_reason, f"{spec.name} is skipped without a reason"


class TestEndToEnd:
    """A full run over a stand-in library, dry then applied."""

    def _setup(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        cache = tmp_path / "cache"
        library = tmp_path / "lib.csv"
        rows = _library_rows()
        _write_library(library, rows)
        # One re-labellable set holding the library's own molecules.
        _write(
            cache / "train_random.csv",
            "SMILES,y",
            [f"{s},0.5" for s, _ in rows[:4]],
        )
        monkeypatch.setattr(
            rl,
            "RELABEL_SETS",
            (
                rl.RelabelSpec(
                    name="train_random",
                    path=rl.DEFAULT_CACHE_DIR / "train_random.csv",
                    smiles_column="SMILES",
                    label_column="y",
                    extra_columns=(),
                    status=rl.SetStatus.RELABEL,
                    expected_rows=4,
                ),
            ),
        )
        return library

    def test_dry_run_writes_nothing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        relaxed_guard: None,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        library = self._setup(tmp_path, monkeypatch)
        target = tmp_path / "cache" / "train_random.csv"
        before = target.read_bytes()
        rl.main(
            [
                "--library-csv",
                str(library),
                "--cache-dir",
                str(tmp_path / "cache"),
                "--data-dir",
                str(tmp_path / "data"),
            ]
        )
        assert target.read_bytes() == before
        assert not rl.backup_path(target).exists()
        assert not (tmp_path / "cache" / rl.MANIFEST_FILE).exists()
        manifest = json.loads(capsys.readouterr().out)
        assert manifest["dry_run"] is True
        assert manifest["sets"]["train_random"]["match_rate"] == 1.0

    def test_apply_writes_and_records(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        relaxed_guard: None,
    ) -> None:
        library = self._setup(tmp_path, monkeypatch)
        target = tmp_path / "cache" / "train_random.csv"
        before = target.read_bytes()
        rl.main(
            [
                "--library-csv",
                str(library),
                "--cache-dir",
                str(tmp_path / "cache"),
                "--data-dir",
                str(tmp_path / "data"),
                "--apply",
            ]
        )
        assert target.read_bytes() != before
        assert rl.backup_path(target).read_bytes() == before
        manifest = json.loads((tmp_path / "cache" / rl.MANIFEST_FILE).read_text())
        entry = manifest["sets"]["train_random"]
        assert entry["status"] == "relabelled"
        assert entry["rows"] == 4
        assert entry["unmatched"] == 0
        assert entry["line_terminator"] == "crlf"
        assert manifest["library"]["duplicate_rule"] == "keep the larger label"
        assert manifest["dry_run"] is False

    def test_nothing_is_written_when_a_check_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        relaxed_guard: None,
    ) -> None:
        library = self._setup(tmp_path, monkeypatch)
        target = tmp_path / "cache" / "train_random.csv"
        # Add a row the library does not contain, under the FORBID policy.
        with target.open("ab") as handle:
            handle.write(b"NOTINLIB,0.5\r\n")
        before = target.read_bytes()
        with pytest.raises(SystemExit, match="self-checks failed"):
            rl.main(
                [
                    "--library-csv",
                    str(library),
                    "--cache-dir",
                    str(tmp_path / "cache"),
                    "--data-dir",
                    str(tmp_path / "data"),
                    "--apply",
                ]
            )
        assert target.read_bytes() == before
        assert not rl.backup_path(target).exists()


class TestRealData:
    """Pinned against the files on disk; skipped where they are absent."""

    def test_the_real_library_passes_its_guard(self) -> None:
        library = Path(rl.DEFAULT_LIBRARY_CSV)
        if not library.is_file():
            pytest.skip(f"{library} is not present")
        _, stats = rl.scan_library(library, set())
        rl.check_library(stats)
        assert stats.rows == 10_000_000
        assert stats.n_hits == 116
        assert stats.n_enriched == 3_653
        assert stats.n_distinct_y == 9_871
        assert stats.n_at_old_floor == 0
        assert stats.mean_y == pytest.approx(0.0042864189360438737, abs=1e-15)
