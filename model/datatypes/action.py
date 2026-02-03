""" Datatype to hold action tensors """
from dataclasses import dataclass

from tensordict import TensorDict


@dataclass
class Action:
    """ Holds current action """
    present_idx: int
    rot: int
    flip: tuple[int, int]
    x: int
    y: int


def from_tensordict(tensordict: TensorDict) -> list[Action]:
    """ Create a list of actions from a tensordict """
    present_idx = tensordict.get(("action", "present_idx"))
    rot = tensordict.get(("action", "rot"))
    flip = tensordict.get(("action", "flip"))
    x = tensordict.get(("action", "x"))
    y = tensordict.get(("action", "y"))

    actions = []

    # Get batch size
    batch_size = tensordict.batch_size[0] if tensordict.batch_size else 1

    if batch_size == 1:
        return [Action(present_idx, rot, flip, x, y)]

    for b in range(batch_size):
        actions.append(
            Action(
                present_idx[b],
                rot[b],
                flip[b],
                x[b],
                y[b]
            )
        )
    return actions
