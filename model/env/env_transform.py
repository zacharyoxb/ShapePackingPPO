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

        self.workers = None
        self.batches = None

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        grid = next_tensordict.get(("observation", "grid"))
        present_count = next_tensordict.get(("observation", "present_count"))

        # Add channel
        grid = grid.unsqueeze(0)

        next_tensordict.set("observation", {
            "grid": grid,
            "present_count": present_count
        })

        return next_tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        self.workers = None
        self.batches = None
        return self._call(tensordict_reset)

    def forward(self, tensordict: TensorDict) -> TensorDict:
        return self._call(tensordict)

    def inv(self, tensordict: TensorDict) -> TensorDict:

        if not self.workers or not self.batches:
            return tensordict

        present_idx = tensordict.get("present_idx")
        present = tensordict.get("present")
        x = tensordict.get("x")
        y = tensordict.get("y")

        present_idx = present_idx.view(
            self.workers, self.batches, -1)
        present = present.view(
            self.workers, self.batches, -1, -1
        )
        x = x.view(
            self.workers, self.batches, -1
        )
        y = y.view(
            self.workers, self.batches, -1
        )

        tensordict.set("action", {
            "present_idx": present_idx,
            "present": present,
            "x": x,
            "y": y
        })

        return tensordict

    def _apply_transform(self, obs: Tensor) -> Tensor:
        raise NotImplementedError
