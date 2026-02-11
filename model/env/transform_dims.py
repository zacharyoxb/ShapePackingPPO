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
            "observation"], in_keys_inv=["action", "batch_dims"], out_keys_inv=["action"])

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        grid = next_tensordict.get("grid")
        present_count = next_tensordict.get("present_count")

        # If multithreaded, combine into total_batch
        workers, batches = None, None
        # If inputs are unbatched
        if grid.dim() == 2:
            # Add batch/channel
            grid = grid.unsqueeze(0).unsqueeze(0)
            present_count = present_count.unsqueeze(0)
        # If inputs are single batched
        elif grid.dim() == 3:
            # Add channel
            grid = grid.unsqueeze(1)
        # If inputs are double batched
        elif grid.dim() == 4:
            # Add channel
            grid = grid.unsqueeze(2)
            # combine workers and batches
            workers, batches = grid.shape[0], grid.shape[1]
            grid = grid.view(workers * batches, *grid.shape[2:])
            present_count = present_count.view(
                workers * batches, *present_count.shape[2:])

        next_tensordict.set("observation", {
            "grid": grid,
            "present_count": present_count
        })

        next_tensordict.set("batch_dims", {
            "workers": workers,
            "batches": batches
        })

        return next_tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._call(tensordict_reset)

    def forward(self, tensordict: TensorDict) -> TensorDict:
        return self._call(tensordict)

    def inv(self, tensordict: TensorDict) -> TensorDict:
        workers = tensordict.get("workers")
        batches = tensordict.get("batches")

        if not workers or not batches:
            return tensordict

        present_idx = tensordict.get("present_idx")
        present = tensordict.get("present")
        x = tensordict.get("x")
        y = tensordict.get("y")

        present_idx = present_idx.view(
            workers, batches, -1)
        present = present.view(
            workers, batches, -1, -1
        )
        x = x.view(
            workers, batches, -1
        )
        y = y.view(
            workers, batches, -1
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
