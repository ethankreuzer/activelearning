import pytest

from activelearning.oracle.config import ConstantOracleConfig
from activelearning.oracle.constant_oracle import ConstantOracle
from activelearning.utils.types import Candidate


class TestConstantOracleQuery:
    """ConstantOracle labels every candidate identically and preserves order."""

    def test_query_returns_the_configured_value_for_every_candidate(self):
        oracle = ConstantOracle(value=0.25)
        candidates = [
            Candidate(x="CCO", fidelity=1),
            Candidate(x="c1ccccc1", fidelity=1),
        ]

        observations = oracle.query(candidates)

        assert [obs.y for obs in observations] == [0.25, 0.25]

    def test_query_preserves_x_order_and_metadata(self):
        oracle = ConstantOracle()
        candidates = [
            Candidate(x="CCO", fidelity=1, metadata={"source": "prior"}),
            Candidate(x="CCN", fidelity=1, metadata=None),
        ]

        observations = oracle.query(candidates)

        assert [obs.x for obs in observations] == ["CCO", "CCN"]
        assert observations[0].metadata == {"source": "prior"}
        assert observations[1].metadata is None

    def test_query_rejects_an_unsupported_fidelity(self):
        oracle = ConstantOracle(fidelity_costs={1: 1.0})

        with pytest.raises(ValueError, match="Unsupported fidelity 2"):
            oracle.query([Candidate(x="CCO", fidelity=2)])

    def test_query_of_nothing_returns_nothing(self):
        assert list(ConstantOracle().query([])) == []


class TestConstantOracleCosts:
    """The cost is fixed per query, which is what the budget schedule validates."""

    def test_get_costs_returns_one_cost_per_candidate(self):
        oracle = ConstantOracle(fidelity_costs={1: 2.5})

        assert oracle.get_costs([Candidate(x="CCO", fidelity=1)] * 3) == [2.5, 2.5, 2.5]

    def test_min_query_cost_matches_the_per_candidate_cost(self):
        assert ConstantOracle(fidelity_costs={1: 2.5}).get_min_query_cost() == 2.5

    def test_get_costs_rejects_an_unsupported_fidelity(self):
        with pytest.raises(ValueError, match="Unsupported fidelity 3"):
            ConstantOracle().get_costs([Candidate(x="CCO", fidelity=3)])


class TestConstantOracleFidelities:
    """A single declared level, at full confidence."""

    def test_declares_one_fidelity_at_full_confidence(self):
        oracle = ConstantOracle(fidelity_costs={1: 1.0})

        assert oracle.get_fidelity_confidences() == {1: 1.0}
        assert oracle.get_supported_fidelities() == [1]

    def test_default_fidelity_is_one_not_candidates_default_of_zero(self):
        """The default matches the pipeline, which is NOT ``DEFAULT_FIDELITY``.

        ``sampler.fidelities: [1]`` in the reward-shape configs, so every
        candidate the S3-GFN sampler emits carries fidelity 1, matching
        ``Dock3OracleConfig``'s single declared level. ``Candidate`` itself
        defaults to ``DEFAULT_FIDELITY`` (0), so a hand-built candidate with no
        fidelity is rejected. Asserted so the mismatch is deliberate.
        """
        assert ConstantOracle().get_supported_fidelities() == [1]
        with pytest.raises(ValueError, match="Unsupported fidelity 0"):
            ConstantOracle().query([Candidate(x="CCO")])


class TestConstantOracleConfig:
    """The config builds the oracle and is reachable through the discriminator."""

    def test_build_passes_the_configured_fields_through(self):
        oracle = ConstantOracleConfig(value=0.5, fidelity_costs={1: 2.0}).build()

        assert isinstance(oracle, ConstantOracle)
        assert oracle.get_min_query_cost() == 2.0
        assert oracle.query([Candidate(x="CCO", fidelity=1)])[0].y == 0.5

    def test_defaults_cost_one_and_value_zero(self):
        oracle = ConstantOracleConfig().build()

        assert oracle.get_min_query_cost() == 1.0
        assert oracle.query([Candidate(x="CCO", fidelity=1)])[0].y == 0.0

    def test_rejects_more_than_one_fidelity_level(self):
        with pytest.raises(ValueError):
            ConstantOracleConfig(fidelity_costs={1: 1.0, 2: 2.0})

    def test_top_level_config_can_extract_its_fidelity_metadata(self):
        """Regression: ActiveLearningConfig reads fidelities off the config.

        ``_extract_oracle_fidelity_confidences`` requires an oracle config to
        expose ``fidelity_costs`` or ``sub_oracles``, and raises otherwise, so
        a stub carrying its own ``cost``/``fidelity`` pair builds fine on its
        own and then fails every full-config load. Reaching into the private
        helper is deliberate: it is the exact check that rejected the first
        version of this config.
        """
        from activelearning.config import _extract_oracle_fidelity_confidences

        confidences = _extract_oracle_fidelity_confidences(ConstantOracleConfig())

        assert confidences == {1: 1.0}

    def test_presents_the_same_surface_to_validators_as_dock3(self):
        """The stub must look like the oracle it replaces, to every validator.

        ``ActiveLearningConfig._resolve_fidelities`` and
        ``_validate_component_compatibility`` read fidelity metadata and
        ``mol_repr`` straight off the oracle config, so a stub that differs on
        either is rejected only when a full config is loaded -- which no unit
        test of the oracle alone exercises. Asserting the surfaces match is
        what makes this stub a drop-in for the single-fidelity DOCK3 arms.
        """
        from activelearning.config import _extract_oracle_fidelity_confidences
        from activelearning.oracle.config import Dock3OracleConfig

        # Paths are never touched: Dock3OracleConfig validates fields only,
        # and the FileNotFoundError lives in Dock3Oracle.__init__, which
        # build() would reach and this test does not call.
        dock3 = Dock3OracleConfig(
            indock_template="/nonexistent/INDOCK",
            dockfiles_dir="/nonexistent",
            fidelity_costs={1: 1.0},
            fidelity_confidences={1: 1.0},
            hitrate_params="/nonexistent/fitted_params.json",
            score_pprop_table="/nonexistent/full_scores.df",
            pki_threshold=6.5,
        )
        stub = ConstantOracleConfig()

        assert _extract_oracle_fidelity_confidences(
            stub
        ) == _extract_oracle_fidelity_confidences(dock3)
        assert stub.mol_repr == dock3.mol_repr
