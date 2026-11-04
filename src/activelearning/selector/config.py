"""Pydantic models of selectors.

Changes in the interface of existing selectors should be reflected in this
configuration. New selectors should define their corresponding pydantic model here and
be added to ``SelectorConfig``.
"""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from activelearning.selector.cost_aware_selector import CostAwareSelector
from activelearning.selector.score_selector import TopKAcquisitionSelector
from activelearning.selector.selector import Selector


#: Never spend an oracle query on a candidate already queried at its own
#: fidelity. That covers two cases: one the dataset already holds a label for,
#: where the repeat would be added a second time and weighted twice by anything
#: fitted on it; and one whose query failed, which leaves no row in the dataset
#: and so would otherwise be retried at full price for the same failure. Set it
#: ``False`` only to reproduce a run made before this check existed.
_SKIP_OBSERVED = Field(default=True)


class TopKAcquisitionSelectorConfig(BaseModel):
    type: Literal["TopKAcquisitionSelector"] = "TopKAcquisitionSelector"
    num_samples: int = Field(gt=0)
    skip_observed: bool = _SKIP_OBSERVED

    def build(self) -> Selector:
        return TopKAcquisitionSelector(
            num_samples=self.num_samples,
            skip_observed=self.skip_observed,
        )


class CostAwareSelectorConfig(BaseModel):
    type: Literal["CostAwareSelector"] = "CostAwareSelector"
    skip_observed: bool = _SKIP_OBSERVED

    def build(self) -> Selector:
        return CostAwareSelector(skip_observed=self.skip_observed)


SelectorConfig = Annotated[
    Union[TopKAcquisitionSelectorConfig, CostAwareSelectorConfig],
    Field(discriminator="type"),
]
