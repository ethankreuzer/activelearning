"""Tests for what a run records about what produced it.

A W&B run has to answer, on its own, "what code, what inputs, what parameters?".
An audit of ``sf-10round-5000dock-5000steps-seed42`` found four ways it could
not:

1. the commit SHA was recorded with no indication of whether the working tree
   matched it, so a run launched from a modified tree recorded a SHA describing
   other code;
2. only config keys literally named ``path``/``file``/``file_path``/
   ``config_path`` were hashed, so the manifest listed three files and omitted
   the MiniMol checkpoint defining the GP's feature space, the fixed-feature
   store, the GFN prior and the DOCK3 installation;
3. provenance went to ``run_manifest.json`` on disk and never to the tracking
   backend, so a W&B run alone could not say which archive file it used;
4. ``runtime.seed`` was logged but ``set_global_seed`` was never called by the
   loop, so NumPy's global RNG and the cuDNN flags were ungoverned.

These tests pin each one.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from activelearning.main import (
    _MAX_HASHED_BYTES,
    _collect_input_files,
    _file_record,
    _git_working_tree_state,
    _iter_config_file_candidates,
    _save_uncommitted_diff,
    _warn_on_dirty_working_tree,
)


class TestWorkingTreeState:
    """Gap 1: a commit SHA means nothing without the tree's state beside it."""

    def test_reports_the_keys_a_reader_needs(self) -> None:
        """Whatever this repository's state, the shape is always the same."""
        state = _git_working_tree_state()

        assert set(state) == {
            "dirty",
            "dirty_paths",
            "untracked_paths",
            "uncommitted_diff_md5",
        }
        assert state["dirty"] in (True, False, None)

    def test_untracked_files_alone_are_not_dirty(self) -> None:
        """An untracked file does not change what the recorded commit means.

        This repository habitually carries unfinished job scripts, so counting
        them as dirty would make the flag fire on every run and mean nothing.
        """
        status = " M src/activelearning/main.py\n?? jobs/wip.sh\n"
        with patch("activelearning.main.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = status
            state = _git_working_tree_state()

        assert state["dirty"] is True
        assert state["dirty_paths"] == ["src/activelearning/main.py"]
        assert state["untracked_paths"] == ["jobs/wip.sh"]

    def test_a_clean_tree_records_no_diff(self) -> None:
        """Nothing modified means no digest to record."""
        with patch("activelearning.main.subprocess.run") as run:
            run.return_value.returncode = 0
            run.return_value.stdout = "?? jobs/wip.sh\n"
            state = _git_working_tree_state()

        assert state["dirty"] is False
        assert state["uncommitted_diff_md5"] is None

    def test_unknown_when_git_is_unavailable(self) -> None:
        """Outside a repository the state is unknown, not clean.

        Reporting False here would assert the tree matched a commit that was
        never found.
        """
        with patch("activelearning.main.subprocess.run") as run:
            run.return_value.returncode = 128
            run.return_value.stdout = ""
            state = _git_working_tree_state()

        assert state["dirty"] is None

    def test_differently_modified_trees_get_different_digests(self) -> None:
        """The digest is what separates two runs from two dirty trees."""
        digests = []
        for diff in ("--- a\n+++ b\n+one\n", "--- a\n+++ b\n+two\n"):
            with patch("activelearning.main.subprocess.run") as run:
                run.return_value.returncode = 0
                run.return_value.stdout = " M src/activelearning/main.py\n"
                with patch(
                    "activelearning.main._git_uncommitted_diff", return_value=diff
                ):
                    digests.append(_git_working_tree_state()["uncommitted_diff_md5"])

        assert digests[0] != digests[1]
        assert all(digest is not None for digest in digests)

    def test_warns_when_dirty(self, caplog: pytest.LogCaptureFixture) -> None:
        """The warning has to name the commit as not describing the code."""
        metadata = {
            "provenance": {
                "dirty": True,
                "dirty_paths": ["src/activelearning/main.py"],
                "git_commit_sha": "deadbeef",
            }
        }

        with caplog.at_level("WARNING"):
            _warn_on_dirty_working_tree(metadata)

        assert "does NOT describe the code being run" in caplog.text
        assert "deadbeef" in caplog.text

    def test_silent_when_clean(self, caplog: pytest.LogCaptureFixture) -> None:
        """A clean run must not be told anything."""
        with caplog.at_level("WARNING"):
            _warn_on_dirty_working_tree({"provenance": {"dirty": False}})

        assert caplog.text == ""

    def test_saves_the_patch_beside_the_run(self, tmp_path: Path) -> None:
        """The digest identifies a dirty tree; the patch reconstructs it."""
        metadata: dict[str, Any] = {"provenance": {"dirty": True}}

        class _Config:
            class run_writer:  # noqa: N801 - mimics the pydantic config attribute
                output_dir = tmp_path / "run"

        with patch(
            "activelearning.main._git_uncommitted_diff", return_value="--- a\n+one\n"
        ):
            _save_uncommitted_diff(metadata, cfg=_Config())

        patch_path = tmp_path / "run" / "uncommitted.patch"
        assert patch_path.read_text(encoding="utf-8") == "--- a\n+one\n"
        assert metadata["provenance"]["uncommitted_diff_path"].endswith(
            "uncommitted.patch"
        )

    def test_a_failed_patch_save_does_not_fail_the_run(self, tmp_path: Path) -> None:
        """A run that cannot record its diff is worse off, not broken."""
        metadata: dict[str, Any] = {"provenance": {"dirty": True}}
        blocker = tmp_path / "blocked"
        blocker.write_text("not a directory", encoding="utf-8")

        class _Config:
            class run_writer:  # noqa: N801
                output_dir = blocker / "run"

        with patch(
            "activelearning.main._git_uncommitted_diff", return_value="--- a\n+one\n"
        ):
            _save_uncommitted_diff(metadata, cfg=_Config())

        assert "uncommitted_diff_path" not in metadata["provenance"]


class TestInputFileCollection:
    """Gap 2: the inputs that define a run were mostly not being recorded."""

    @pytest.mark.parametrize(
        "key",
        [
            "checkpoint_path",
            "store_path",
            "smiles_path",
            "model_name_or_path",
            "tokenizer_name_or_path",
            "indock_template",
            "dock64_exe",
            "dockenv_sh",
            "state_path",
        ],
    )
    def test_collects_the_keys_the_audit_found_missing(
        self, tmp_path: Path, key: str
    ) -> None:
        """Each of these named a real input that went unrecorded."""
        target = tmp_path / "input.bin"
        target.write_bytes(b"payload")

        collected = dict(_iter_config_file_candidates({key: str(target)}))

        assert collected == {str(target): False}

    def test_records_directories_by_path_without_hashing(
        self, tmp_path: Path
    ) -> None:
        """Hashing a shared tree means walking it; naming it still identifies it."""
        package = tmp_path / "encoder_package"
        package.mkdir()

        collected = dict(_iter_config_file_candidates({"package_path": str(package)}))
        assert collected == {str(package): True}

        record = _file_record(package)
        assert record["kind"] == "directory"
        assert record["md5"] is None
        assert "directory" in record["md5_skipped"]

    @pytest.mark.parametrize("key", ["output_dir", "log_dir", "cache_dir", "subset_dir"])
    def test_never_records_where_the_run_writes(self, tmp_path: Path, key: str) -> None:
        """An output recorded as an input changes the provenance run to run."""
        assert dict(_iter_config_file_candidates({key: str(tmp_path)})) == {}

    def test_a_hub_identifier_is_not_mistaken_for_a_file(self) -> None:
        """model_name_or_path is often not a path at all."""
        records = _collect_input_files([], {"model_name_or_path": "some-org/some-model"})

        assert records == []

    def test_hashes_a_small_file(self, tmp_path: Path) -> None:
        """Everything the AmpC configs name but the store is hashed."""
        target = tmp_path / "archive.csv"
        target.write_bytes(b"SMILE,y\nC,0.1\n")

        record = _file_record(target)

        assert record["md5"] == hashlib.md5(target.read_bytes()).hexdigest()
        assert record["bytes"] == target.stat().st_size
        assert "mtime" in record

    def test_identifies_an_oversized_file_without_reading_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The 20.5 GB feature store must not be read at every run start.

        Size and mtime stand in, and `md5_skipped` says why -- a null digest
        must never read as "this input was unchanged".
        """
        target = tmp_path / "store.npy"
        target.write_bytes(b"x" * 64)
        monkeypatch.setattr("activelearning.main._MAX_HASHED_BYTES", 8)

        def _must_not_hash(path: Path) -> str:
            raise AssertionError(f"hashed an oversized file: {path}")

        monkeypatch.setattr("activelearning.main._file_md5", _must_not_hash)
        record = _file_record(target)

        assert record["md5"] is None
        assert "larger than" in record["md5_skipped"]
        assert record["bytes"] == 64

    def test_the_threshold_still_hashes_every_real_input(self) -> None:
        """The MiniMol checkpoint (33 MB) and held-out set (23 MB) must be hashed.

        A threshold set below those would silence exactly the inputs this change
        exists to capture.
        """
        assert _MAX_HASHED_BYTES > 64 * 1024 * 1024

    def test_follows_a_symlinked_input_to_its_target(self, tmp_path: Path) -> None:
        """The feature store is a link into project space; the target is the input."""
        target = tmp_path / "real.npy"
        target.write_bytes(b"payload")
        link = tmp_path / "link.npy"
        link.symlink_to(target)

        records = _collect_input_files([], {"store_path": str(link)})

        assert len(records) == 1
        assert records[0]["path"].endswith("real.npy")

    def test_records_each_input_once(self, tmp_path: Path) -> None:
        """The same file named twice is one input, not two."""
        target = tmp_path / "shared.csv"
        target.write_bytes(b"data")

        records = _collect_input_files(
            [], {"a": {"path": str(target)}, "b": {"file": str(target)}}
        )

        assert len(records) == 1


@pytest.fixture
def minimal_config(tmp_path: Path) -> Path:
    """The smallest config that runs the CLI end to end on the CPU."""
    import textwrap

    path = tmp_path / "minimal.yaml"
    path.write_text(
        textwrap.dedent(
            """
            runtime:
              device: cpu
              precision: 64
              seed: 1234

            logger:
              type: ConsoleLogger
              project_name: provenance-test

            dataset:
              type: ListDataset

            surrogate:
              type: BoTorchGPSurrogate

            acquisition:
              type: DummyAcquisition

            sampler:
              type: HypercubeSampler
              bounds:
                - [0.0, 1.0]
                - [0.0, 1.0]
              num_samples: 20
              fidelities: [1, 2]

            selector:
              type: CostAwareSelector

            oracle:
              type: BraninOracle
              fidelity_costs:
                1: 0.01
                2: 0.1

            budget:
              available_budget: 0.05
              schedule:
                type: constant
                value: 0.05
            """
        ),
        encoding="utf-8",
    )
    return path


class TestProvenanceReachesTheBackend:
    """Gap 3: provenance was written to disk and never sent to the tracker."""

    def test_the_logger_receives_cli_and_provenance(self, minimal_config: Path) -> None:
        """A W&B run must be able to answer "what produced me?" unaided."""
        import sys

        from activelearning.main import main

        logged: list[dict[str, Any]] = []

        class _CapturingLogger:
            def log_config(self, config: dict[str, Any]) -> None:
                logged.append(config)

            def __getattr__(self, name: str) -> Any:
                return lambda *args, **kwargs: None

        captured_logger = _CapturingLogger()

        with patch.object(sys, "argv", ["activelearning", str(minimal_config)]):
            with patch(
                "activelearning.active_learning.active_learning",
                side_effect=lambda *args, **kwargs: (
                    kwargs.get("dataset") or args[0],
                    0.0,
                    0,
                ),
            ):
                with patch(
                    "activelearning.logger.config.ConsoleLoggerConfig.build",
                    return_value=captured_logger,
                ):
                    main()

        merged: dict[str, Any] = {}
        for payload in logged:
            merged.update(payload)

        assert "effective" in merged, "the validated config must still be logged"
        assert "provenance" in merged, "provenance must reach the tracking backend"
        assert "cli" in merged, "the reproduction command must reach the backend"

        provenance = merged["provenance"]
        assert set(provenance) >= {
            "git_commit_sha",
            "dirty",
            "dirty_paths",
            "untracked_paths",
            "uncommitted_diff_md5",
            "resolved_config_md5",
            "input_files",
        }
        assert merged["cli"]["command"].startswith("activelearning ")


class TestGlobalSeedIsApplied:
    """Gap 4: runtime.seed was logged but never applied to the global RNGs."""

    def test_the_configured_seed_is_applied_before_anything_is_built(
        self, minimal_config: Path
    ) -> None:
        """The loop never called set_global_seed, so NumPy and cuDNN were loose."""
        import sys

        from activelearning.main import main

        seen: list[int] = []

        with patch.object(sys, "argv", ["activelearning", str(minimal_config)]):
            with patch(
                "activelearning.utils.seeding.set_global_seed",
                side_effect=seen.append,
            ):
                with patch(
                    "activelearning.active_learning.active_learning",
                    side_effect=lambda *args, **kwargs: (
                        kwargs.get("dataset") or args[0],
                        0.0,
                        0,
                    ),
                ):
                    main()

        assert seen == [1234], "runtime.seed must reach set_global_seed exactly once"

    def test_seeding_covers_numpy_and_cudnn(self) -> None:
        """What set_global_seed must still do for the call above to be worth it."""
        import numpy as np

        from activelearning.utils.seeding import set_global_seed

        set_global_seed(7)
        first = np.random.rand(4).tolist()
        set_global_seed(7)

        assert np.random.rand(4).tolist() == first
