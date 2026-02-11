""" Actor Policy implementation for use with my present packing environment. """
from tensordict.nn import (
    ProbabilisticTensorDictSequential,
    ProbabilisticTensorDictModule,
    TensorDictModule,
    OneHotCategorical,
    CompositeDistribution
)
import torch
from torch import nn
from torch import distributions as d

from model.actor.selection_modules.position_selection import PresentPositionActor
from model.actor.selection_modules.present_selection import PresentSelectionActor


class PresentActorSeq(ProbabilisticTensorDictSequential):
    """ Policy nn for PresentEnv with spatial awareness """

    def __init__(self, presents: torch.Tensor, device=torch.device("cpu")):

        self.flatten = nn.Flatten()
        self.present_selection = PresentSelectionActor(presents, device)
        self.position_selection = PresentPositionActor(presents, device)

        present_select_prob = TensorDictModule(
            self.present_selection,
            in_keys=["observation"],
            out_keys=["orient_data"]
        )

        present_select = ProbabilisticTensorDictModule(
            in_keys=["orient_data"],
            out_keys=["orient_mask"],
            distribution_class=OneHotCategorical,
            return_log_prob=True
        )

        present_pos_prob = TensorDictModule(
            self.position_selection,
            in_keys=["orient_data", "orient_mask"],
            out_keys=["action", "pos_probs"],
        )

        present_pos = ProbabilisticTensorDictModule(
            in_keys=["pos_probs"],
            distribution_class=CompositeDistribution,
            distribution_kwargs={
                "distribution_map": {
                    "x": d.Normal,
                    "y": d.Normal
                },
                "name_map": {
                    "x": ("action", "x"),
                    "y": ("action", "y")
                }
            },
            return_log_prob=True
        )

        super().__init__(
            [
                present_select_prob,
                present_select,
                present_pos_prob,
                present_pos
            ]
        )
