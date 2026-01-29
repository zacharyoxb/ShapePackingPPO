""" Neural network for packing presents with orientation masks """

import torch
from tensordict import TensorDict

from torchrl.data import (Bounded, Composite, Unbounded,
                          Categorical)
from torchrl.envs import EnvBase

from model.datatypes import action, state

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

        self.start_state = start_state

        super().__init__(device=device)
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
                "present_idx": Bounded(low=0, high=MAX_PRESENT_IDX, shape=1, dtype=torch.uint8),
                "rot": Bounded(low=0, high=MAX_ROT, shape=1,
                               dtype=torch.uint8),
                "flip": Bounded(low=0, high=MAX_FLIP, shape=torch.Size([2]), dtype=torch.uint8),
                "x": Unbounded(shape=1, dtype=torch.int64),
                "y": Unbounded(shape=1, dtype=torch.int64)
            }
        })

    def _make_spec(self):
        # extract x y bounds from grid shape
        grid = self.start_state.get("grid")
        h, w = grid.shape

        # Observation spec: what the agent sees
        self.observation_spec = Composite({
            "observation": {
                "grid": Bounded(low=0, high=1, dtype=torch.float32, shape=grid.shape,
                                device=self.device),
                "presents": Bounded(low=0, high=1, shape=torch.Size([3, 3]),
                                    dtype=torch.float32, device=self.device),
                "present_count": Unbounded(shape=torch.Size([6]), dtype=torch.float32,
                                           device=self.device),
            }
        })

        # Action spec: what the agent can do
        self.action_spec = Composite({
            "action": {
                "present_idx": Bounded(low=0, high=MAX_PRESENT_IDX, shape=1, dtype=torch.uint8),
                "rot": Bounded(low=0, high=MAX_ROT, shape=1,
                               dtype=torch.uint8),
                "flip": Bounded(low=0, high=MAX_FLIP, shape=torch.Size([2]), dtype=torch.uint8),
                "x": Bounded(low=1, high=w-2, shape=1, dtype=torch.int64),
                "y": Bounded(low=1, high=h-2, shape=1, dtype=torch.int64)
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

        grid = self.start_state.get("grid")
        presents = self.start_state.get("presents")
        present_count = self.start_state.get("present_count")

        # Return as TensorDict with observation keys
        return TensorDict({
            "observation": {
                "grid": grid,
                "presents": presents,
                "present_count": present_count,
            },
        }, device=self.device)

    def _handle_batch(
            self,
            batch_state: state.State,
            batch_action: action.Action
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        w, h = batch_state.grid.shape

        # out of bounds check
        in_bounds = (batch_action.x >= 0) & (batch_action.y >= 0) & (
            batch_action.x + 2 < w) & (batch_action.y + 2 < h)
        if not in_bounds:
            reward = torch.tensor(-40, dtype=torch.float32)
            done = torch.tensor(True)
            return batch_state.grid, batch_state.presents, batch_state.present_count, reward, done

        present = batch_state.presents[batch_action.present_idx].clone(
        )
        present = torch.rot90(present, batch_action.rot)

        if batch_action.flip[0]:
            present = torch.flip(present, (1,))
        if batch_action.flip[1]:
            present = torch.flip(present, (0,))

        # round x y values
        x, y = round(float(batch_action.x)), round(float(batch_action.y))

        # collision check
        grid_region = batch_state.grid[y:y + 3, x:x+3]
        if torch.any(present * grid_region > 0):
            reward = torch.tensor(-20, dtype=torch.float32)
            done = torch.tensor(True)
            return batch_state.grid, batch_state.presents, batch_state.present_count, reward, done

        batch_state.present_count[batch_action.present_idx] -= 1
        batch_state.grid[y:y+3, x:x+3] = torch.maximum(grid_region, present)

        # Base reward
        reward = torch.tensor(10, dtype=torch.float32)

        # Check if all shapes are placed
        done = torch.tensor(False)
        if torch.sum(batch_state.present_count) == 0:
            done = torch.tensor(True)
            reward += 200

        return batch_state.grid, batch_state.presents, batch_state.present_count, reward, done

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """ Execute one action - returns NEXT observation + reward + done """
        states = state.from_tensordict(tensordict)
        actions = action.from_tensordict(tensordict)

        # Process everything in one loop
        results = [self._handle_batch(b_state, b_action)
                   for b_state, b_action in zip(states, actions)]

        # Unzip results
        batch_grids, batch_presents, batch_present_counts, batch_rewards, batch_dones = zip(
            *results)

        return TensorDict({
            "observation": {
                "grid": torch.stack(batch_grids),
                "presents": torch.stack(batch_presents),
                "present_count": torch.stack(batch_present_counts)
            },
            "reward": torch.stack(batch_rewards),
            "done": torch.stack(batch_dones)
        })

    def rollout(self, max_steps=1000, policy=None, callback=None, **_kwargs):
        """ Executes environment rollout with given policy using TensorDict operations. """
        # preallocate:
        data = TensorDict({}, [max_steps])

        # Reset environment
        _data = self.reset()

        policy_input = _data.get("observation").select(
            "grid", "presents", "present_count")

        # While present_count more than 0 and steps not exceeded
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
