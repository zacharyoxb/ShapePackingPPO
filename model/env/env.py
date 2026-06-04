""" Neural network for packing presents with orientation masks """
import torch
from tensordict import TensorDict

from torchrl.data import (Bounded, Composite, Unbounded,
                          Categorical)
from torchrl.envs import EnvBase, ParallelEnv, TransformedEnv

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
                "present": Bounded(low=0, high=1, shape=torch.Size([3, 3]), dtype=torch.float32),
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
                                shape=torch.Size((1, h, w)), device=self.device),
                "present_count": Unbounded(shape=torch.Size([6]), dtype=torch.float32,
                                           device=self.device),
            }
        })

        # Action spec: what the agent can do
        self.action_spec = Composite({
            "action": {
                "present_idx": Bounded(low=0, high=MAX_PRESENT_IDX, shape=torch.Size([]),
                                       dtype=torch.uint8),
                "present": Bounded(low=0, high=1, shape=torch.Size([3, 3]),
                                   dtype=torch.float32),
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

    def _handle_step(
            self,
            tensordict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        grid = tensordict.get(("observation", "grid"))
        present_count = tensordict.get(
            ("observation", "present_count"))

        present_idx = tensordict.get(("action", "present_idx"))
        present = tensordict.get(("action", "present"))
        x_coord = tensordict.get(("action", "x"))
        y_coord = tensordict.get(("action", "y"))

        w, h = grid.shape[-2], grid.shape[-1]

        # Round coords
        x = int(torch.round(x_coord))
        y = int(torch.round(y_coord))

        # In bounds check
        in_bounds = (x_coord >= 0) & (y_coord >= 0) & (
            x_coord + 2 < w) & (y_coord + 2 < h)

        # Init reward and done values
        reward = torch.tensor(-200, dtype=torch.float32).unsqueeze(0)
        done = torch.tensor(True, dtype=torch.bool).unsqueeze(0)

        # If out of bounds, return as-is
        if not in_bounds:
            return grid, present_count, reward, done

        placement_area = grid[:, y:y+3, x:x+3]
        collision = (present * placement_area) > 0

        # If there are any collisions, set reward to -20 per collision
        if torch.any(collision):
            reward = -20 * torch.sum(collision)
            reward = reward.unsqueeze(0)
            return grid, present_count, reward, done

        # For a valid placement give base reward of 10
        reward = torch.tensor(10, dtype=torch.float32,
                              device=self.device).unsqueeze(0)

        # Give bonus for compact placement (check 5x5 area)
        placement_region = grid[
            :, max(0, y-1):min(h, y+4), max(0, x-1):min(w, x+4)
        ]

        # If there aren't 25 values, add 1s for each out of bounds area
        multiplier = torch.sum(placement_region) + \
            (25 - placement_region.numel())

        reward *= multiplier

        # Update state using action
        present_idx = int(present_idx)
        present_count[present_idx] -= 1
        grid[:, y:y+3, x:x+3] = torch.maximum(placement_area, present)

        # If all presents are placed, give an extra 1000 reward
        if torch.sum(present_count) == 0:
            reward += 1000
        else:
            done = torch.tensor(False, dtype=torch.bool).unsqueeze(0)

        return grid, present_count, reward, done

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """ Execute one action - returns NEXT observation + reward + done """

        new_grid, new_present_count, reward, done = self._handle_step(
            tensordict)

        return TensorDict({
            "observation": {
                "grid": new_grid,
                "present_count": new_present_count
            },
            "reward": reward,
            "done": done
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
