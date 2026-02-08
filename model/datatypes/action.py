""" Datatype to hold action tensors """
from dataclasses import dataclass

from tensordict import TensorDict
import torch


@dataclass
class Action:
    """ Holds current action """
    present_idx: int
    present: torch.Tensor
    x: int
    y: int


def from_tensordict(tensordict: TensorDict) -> list[Action]:
    """ Create a list of actions from a tensordict """
    present_idx = tensordict.get(("action", "present_idx"))
    present = tensordict.get(("action", "present"))
    x = tensordict.get(("action", "x"))
    y = tensordict.get(("action", "y"))

    actions = []

    # Get batch size
    batch_size = tensordict.batch_size[0] if tensordict.batch_size else 1

    if batch_size == 1:
        return [Action(present_idx, present, x, y)]

    for b in range(batch_size):
        actions.append(
            Action(
                present_idx[b],
                present[b],
                x[b],
                y[b]
            )
        )
    return actions
