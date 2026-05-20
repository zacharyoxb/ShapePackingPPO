""" Neural network for packing presents with orientation masks """
import torch
from tensordict import TensorDict

from torchrl.data import (Bounded, Composite, Unbounded,
                          Categorical)
from torchrl.envs import EnvBase, ParallelEnv, TransformedEnv

from model.datatypes import action, state
from model.env.env_transform import PresentEnvTransform

MAX_PRESENT_IDX = 5
MAX_ROT = 3
MAX_FLIP = 1


class PresentEnv(EnvBase):
    """ RL environment for present placement """

    def __init__(
            self,
            start_state: TensorDict,
            seed=None,
            device=None,
    ):
        super().__init__(device=device)
        self.start_state = start_state
        self.rng = None

        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()

        self.set_seed(int(seed))
        self._make_spec()

    @classmethod
    def get_action_spec(cls):
        """ Gets just the action spec for policy initialisation. """
        # Action spec: what the agent can do
        return Composite({
            "action": {
                "present_idx": Bounded(low=0, high=MAX_PRESENT_IDX, shape=torch.Size([]),
                                       dtype=torch.uint8),
                "present": Bounded(low=0, high=1, shape=torch.Size([]), dtype=torch.float32),
                "x": Unbounded(shape=torch.Size([]), dtype=torch.int64),
                "y": Unbounded(shape=torch.Size([]), dtype=torch.int64)
            }
        })

    def _make_spec(self):
        # extract x y bounds from grid shape
        grid = self.start_state.get("grid")
        h, w = grid.shape

        # Observation spec: what the agent sees
        self.observation_spec = Composite({
            "observation": {
                "grid": Bounded(low=0, high=1, dtype=torch.float32,
                                shape=torch.Size((-1, -1, h, w)), device=self.device),
                "present_count": Unbounded(shape=torch.Size([-1, 6]), dtype=torch.float32,
                                           device=self.device),
            }
        })

        # Action spec: what the agent can do
        self.action_spec = Composite({
            "action": {
                "present_idx": Bounded(low=0, high=MAX_PRESENT_IDX, shape=torch.Size([]),
                                       dtype=torch.uint8),
                "present": Bounded(low=0, high=1, shape=torch.Size([]), dtype=torch.float32),
                "x": Bounded(low=1, high=w-2, shape=torch.Size([]), dtype=torch.int64),
                "y": Bounded(low=1, high=h-2, shape=torch.Size([]), dtype=torch.int64)
            }
        })

        # Reward and done specs
        self.reward_spec = Unbounded(shape=torch.Size(
            [1]), dtype=torch.float32, device=self.device)
        self.done_spec = Categorical(
            n=2, shape=torch.Size([1]), dtype=torch.bool, device=self.device)  # 0/1 for False/True

    def _set_seed(self, seed: int | None = None):
        """
        Set random seeds for reproducibility.

        Args:
            seed: Integer seed value
        """
        self.rng = torch.manual_seed(seed)

    def _reset(self, tensordict, **kwargs) -> TensorDict:
        """ Initialize new episode - returns FIRST observation """

        grid = self.start_state.get("grid").clone()
        present_count = self.start_state.get("present_count").clone()

        # Return as TensorDict with observation keys
        return TensorDict({
            "observation": {
                "grid": grid,
                "present_count": present_count,
            },
        }, device=self.device)

    def _handle_batch(
            self,
            batch_state: state.State,
            batch_action: action.Action
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        w, h = batch_state.grid.shape[-2], batch_state.grid.shape[-1]

        # Always unsqueeze actions if there's no batches
        if not self.batch_size:
            batch_action.present = batch_action.present.unsqueeze(0)
            batch_action.present_idx = batch_action.present_idx.unsqueeze(0)
            batch_action.x = batch_action.x.unsqueeze(0)
            batch_action.y = batch_action.y.unsqueeze(0)
        # We only need to unsqueeze state once
        if batch_state.grid.dim() < 3:
            batch_state.grid = batch_state.grid.unsqueeze(0)
            batch_state.present_count = batch_state.present_count.unsqueeze(0)

        # Round coords
        x_coords = torch.round(batch_action.x)
        y_coords = torch.round(batch_action.y)

        # In bounds mask
        in_bounds = torch.where(
            (x_coords >= 0) &
            (y_coords >= 0) &
            (x_coords + 2 < w) &
            (y_coords + 2 < h),
            True,
            False
        )

        # Init reward and done values
        rewards = torch.full(
            self.batch_size or torch.Size([1, 1]), -40, dtype=torch.float32,
            device=batch_state.grid.device
        )
        dones = torch.ones(self.batch_size or torch.Size([1, 1]), dtype=torch.bool,
                           device=batch_state.grid.device)

        # If they are all out of bounds, return
        if not in_bounds.any():
            return batch_state.grid, batch_state.present_count, rewards, dones

        # Initialise collisions
        collisions = torch.zeros(
            self.batch_size or torch.Size([1, 1]), dtype=torch.bool, device=batch_state.grid.device)

        # Check collisions for every in-bounds action
        for batch_idx in torch.where(in_bounds):
            x, y = int(x_coords[batch_idx, :]), int(y_coords[batch_idx, :])
            grid_region = batch_state.grid[batch_idx, y:y+3, x:x+3]
            present = batch_action.present[batch_idx, :, :]
            collisions[batch_idx] = torch.any((present * grid_region) > 0)

        # If there are any collisions, set reward to -20
        rewards[collisions] = torch.tensor(-20, dtype=torch.float32)

        # If there are no valid placements, exit early
        if not torch.any(in_bounds & ~collisions):
            return batch_state.grid, batch_state.present_count, rewards, dones

        # Otherwise update state using action
        for batch_idx in torch.where(in_bounds & ~collisions):
            present_idx = int(batch_action.present_idx[batch_idx, :])
            batch_state.present_count[batch_idx, present_idx] -= 1
            batch_state.grid[batch_idx, y:y+3, x:x +
                             3] = torch.maximum(grid_region, present)

        # For all valid placements give reward of 10
        rewards[in_bounds & ~collisions] = torch.tensor(
            10, dtype=torch.float32)

        # If all shapes are placed in any batch, add reward.
        for batch_idx in torch.where(in_bounds & ~collisions):
            if torch.sum(batch_state.present_count[batch_idx]) == 0:
                rewards[batch_idx] += 200
            else:
                dones[batch_idx] = torch.tensor(False, dtype=torch.bool)

        return batch_state.grid, batch_state.present_count, rewards, dones

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """ Execute one action - returns NEXT observation + reward + done """
        # Process everything in one loop
        results = [
            self._handle_batch(b_state, b_action)
            for b_state, b_action in zip(
                state.from_tensordict(tensordict),
                action.from_tensordict(tensordict)
            )
        ]

        # Unzip results
        grids, present_counts, rewards, dones = zip(
            *results)

        return TensorDict({
            "observation": {
                "grid": grids,
                "present_count": present_counts
            },
            "reward": rewards,
            "done": dones
        })

    def rollout(self, max_steps=1000, policy=None, callback=None, **_kwargs):
        """ Executes environment rollout with given policy using TensorDict operations. """
        # preallocate:
        data = TensorDict({}, [max_steps])

        # Reset environment
        _data = self.reset()

        # While present_count more than 0 and steps not exceeded
        for i in range(max_steps):
            # Compute an action given a policy
            if policy:
                _data.update(policy(_data))
            else:
                _data.update(self.action_spec.rand())

            # execute step, collect data
            _data = self.step(_data)
            data[i] = _data

            # mdp step
            _data = self.step_mdp(_data)

            # check if count is 0, if so, break
            present_count = _data.get(("observation", "present_count"))
            if torch.sum(present_count) == 0:
                break

        return data

    @classmethod
    def make_transformed_env(
        cls,
        start_state: TensorDict,
        seed: int | float | None = None,
        device: torch.device | None = None,
    ) -> TransformedEnv:
        """ 
        Creates a TransformedEnv with a PresentEnv inside
        for easy batch and dimension handling.
        """
        env_start_state = start_state.clone()
        env_start_state = env_start_state.to(device)

        env = PresentEnv(
            start_state=env_start_state,
            seed=seed,
            device=device
        )

        return TransformedEnv(env, PresentEnvTransform())

    @classmethod
    def make_parallel_env(
        cls,
        start_state: TensorDict,
        num_workers: int,
        device: torch.device | None = None,
    ) -> ParallelEnv:
        """
        Creates a ParallelEnv with multiple PresentEnv instances.

        Args:
            start_state: Initial state (will be cloned for each worker)
            num_workers: Number of parallel environments 
            device: Device to run environments on
        """

        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

        base_seed = torch.empty((), dtype=torch.int64).random_().item()
        seeds = [base_seed + i for i in range(num_workers)]

        def create_env(worker_id: int = 0):
            worker_start_state = start_state.clone()
            worker_start_state = worker_start_state.to(device)

            worker_seed = seeds[worker_id] if worker_id < len(
                seeds) else seeds[0]

            return cls.make_transformed_env(
                start_state=worker_start_state,
                seed=worker_seed,
                device=device
            )

        return ParallelEnv(
            num_workers=num_workers,
            create_env_fn=create_env,
            device=device
        )

    def forward(self, *args, **kwargs):
        """ Unimplemented in environment only class """
        raise NotImplementedError("This is an env, not a nn.")
