""" Neural network for packing presents with orientation masks """

import torch
from tensordict import TensorDict

from torchrl.data import (Bounded, Composite, Unbounded,
                          UnboundedContinuous, Categorical)
from torchrl.envs import EnvBase

MAX_PRESENT_IDX = 5
MAX_ROT = 3
MAX_FLIP = 1


class PresentEnv(EnvBase):
    """ RL environment for present placement """

    def __init__(
            self,
            start_state: TensorDict,
            batch_size=None,
            seed=None,
            device=None,
    ):
        if batch_size is None:
            batch_size = torch.Size([])

        self.start_state = start_state

        super().__init__(device=device, batch_size=batch_size)
        self.batch_size = batch_size
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
            "present_idx": Bounded(low=0, high=MAX_PRESENT_IDX, shape=1, dtype=torch.uint8),
            "x": Unbounded(shape=1, dtype=torch.int64),
            "y": Unbounded(shape=1, dtype=torch.int64),
            "rot": Bounded(low=0, high=MAX_ROT, shape=1,
                           dtype=torch.uint8),
            "flip": Bounded(low=0, high=MAX_FLIP, shape=torch.Size([2]), dtype=torch.uint8)
        })

    def _make_spec(self):
        # Observation spec: what the agent sees
        self.observation_spec = Composite({
            "grid": UnboundedContinuous(dtype=torch.float32, device=self.device),
            "presents": Bounded(low=0, high=1, shape=torch.Size([3, 3]),
                                dtype=torch.float32, device=self.device),
            "present_count": Unbounded(shape=torch.Size([6]), dtype=torch.float32,
                                       device=self.device),
        })

        # Action spec: what the agent can do
        self.action_spec = Composite({
            "present_idx": Bounded(low=0, high=MAX_PRESENT_IDX, shape=1, dtype=torch.uint8,
                                   device=self.device),
            "x": Unbounded(shape=1, dtype=torch.int64,
                           device=self.device),
            "y": Unbounded(shape=1, dtype=torch.int64,
                           device=self.device),
            "rot": Bounded(low=0, high=MAX_ROT, shape=1,
                           dtype=torch.uint8, device=self.device),
            "flip": Bounded(low=0, high=MAX_FLIP, shape=torch.Size([2]), dtype=torch.uint8,
                            device=self.device)
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

        grid = self.start_state.get("grid")
        presents = self.start_state.get("presents")
        present_count = self.start_state.get("present_count")

        # Return as TensorDict with observation keys
        return TensorDict({
            "grid": grid,
            "presents": presents,
            "present_count": present_count,
        }, batch_size=self.batch_size, device=self.device)

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """ Execute one action - returns NEXT observation + reward + done """
        # Get current state and action
        grid = tensordict.get("grid").clone()
        presents = tensordict.get("presents")
        present_count = tensordict.get("present_count").clone()

        present_idx = int(tensordict.get(("action", "present_idx")))
        x = int(tensordict.get(("action", "x")))
        y = int(tensordict.get(("action", "y")))
        rot = int(tensordict.get(("action", "rot")))
        flip = tensordict.get(("action", "flip")).tolist()[0]

        # Get present
        present = presents[present_idx]
        present = torch.rot90(present, rot)

        if flip[0]:
            present = torch.flip(present, (1,))
        if flip[1]:
            present = torch.flip(present, (0,))

        # If collision, exit early
        grid_region = grid[y:y+3, x:x+3]
        if torch.any(present * grid_region > 0):
            return TensorDict({
                "grid": grid,
                "presents": presents,
                "present_count": present_count,
                "reward": torch.tensor(-20, dtype=torch.float32),
                "done": torch.tensor(True)
            }, batch_size=self.batch_size, device=self.device)

        # Otherwise, update tensors
        present_count[present_idx] -= 1
        grid[y:y+3, x:x+3] = torch.maximum(grid_region, present)

        # Base reward
        reward = torch.tensor(2, dtype=torch.float32)

        # Check if all shapes are placed
        done = torch.tensor(False)
        if torch.sum(present_count) == 0:
            done = torch.tensor(True)

        return TensorDict({
            "grid": grid,
            "presents": presents,
            "present_count": present_count,
            "reward": reward,
            "done": done
        }, batch_size=self.batch_size, device=self.device)

    def rollout(self, max_steps=1000, policy=None, callback=None, **_kwargs):
        """ Executes environment rollout with given policy using TensorDict operations. """
        # preallocate:
        data = TensorDict({}, [max_steps])

        # Reset environment
        _data = self.reset()

        policy_input = _data.select("grid", "presents", "present_count")

        # While present_count is not 0 OR steps are exceeded
        for i in range(max_steps):
            # Compute an action given a policy
            if policy:
                _data["action"] = policy(policy_input)
            else:
                _data["action"] = self.action_spec.rand()

            # execute step, collect data
            _data = self.step(_data)
            data[i] = _data

            # mdp step
            _data = self.step_mdp(_data)

            # check if count is 0, if so, break
            present_count = _data["present_count"]
            if torch.sum(present_count) == 0:
                break

        return data

    def forward(self, *args, **kwargs):
        """ Unimplemented in environment only class """
        raise NotImplementedError("This is an env, not a nn.")
