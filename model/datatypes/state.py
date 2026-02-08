""" Datatype to hold state tensors """
from dataclasses import dataclass

from tensordict import TensorDict
import torch


@dataclass
class State:
    """ Holds current env state """
    grid: torch.Tensor
    present_count: torch.Tensor


def from_tensordict(tensordict: TensorDict) -> list[State]:
    """ Create a list of states from a tensordict """
    # Get current state
    grid = tensordict.get(("observation", "grid")).clone()
    present_count = tensordict.get(
        ("observation", "present_count")).clone()

    # add batch dims
    if grid.ndim == 2:
        grid = grid.unsqueeze(0)

    if present_count.ndim == 1:
        present_count = present_count.unsqueeze(0)

    states = []

    # Get batch size
    batch_size = tensordict.batch_size[0] if tensordict.batch_size else 1

    for b in range(batch_size):
        states.append(
            State(
                grid[b],
                present_count[b]
            )
        )
    return states
