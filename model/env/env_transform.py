""" Adds and removes dims to make them consistent when selecting / extracting features """

from tensordict import TensorDict, TensorDictBase
from torch import Tensor
from torchrl.envs.transforms import Transform


class PresentEnvTransform(Transform):
    """
    A transform that changes the dims of grid and present
    count so they both have 1 batch dimension and grid has
    one channel dimension.

    Its inverse only reverses combining batch dims if there
    was more than one before the transform.
    """

    def __init__(self):
        super().__init__(in_keys=["observation"], out_keys=[
            "observation"], in_keys_inv=["action"], out_keys_inv=["action"])

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        return next_tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        grid = tensordict_reset.get(("observation", "grid"))
        present_count = tensordict_reset.get(("observation", "present_count"))

        # Add channel
        grid = grid.unsqueeze(0)

        tensordict_reset.set("observation", {
            "grid": grid,
            "present_count": present_count
        })

        return tensordict_reset

    def forward(self, tensordict: TensorDict) -> TensorDict:
        return tensordict

    def _apply_transform(self, obs: Tensor) -> Tensor:
        raise NotImplementedError
