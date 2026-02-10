from torchrl.envs.transforms import Transform


class ObsDimsTransform(Transform):
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

    def _apply_transform(self, obs):
        grid = obs.get("grid")
        present_count = obs.get("present_count")

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
