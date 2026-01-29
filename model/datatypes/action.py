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
    batched_present_idx = tensordict.get(("action", "present_idx"))
    batched_rot = tensordict.get(("action", "rot"))
    batched_flip = tensordict.get(("action", "flip")).tolist()
    batched_x = tensordict.get(("action", "x"))
    batched_y = tensordict.get(("action", "y"))

    actions = []

    # Get batch size
    batch_size = tensordict.batch_size[0] if tensordict.batch_size else 1

    for b in range(batch_size):
        actions.append(
            Action(
                batched_present_idx[b],
                batched_rot[b],
                batched_flip[b],
                batched_x[b],
                batched_y[b]
            )
        )
    return actions
