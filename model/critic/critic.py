""" Critic Policy implementation for use with my present packing environment. """
from tensordict.nn import TensorDictModule
from torch import nn

from model.critic.feature_modules.value import PresentValue


class PresentCritic(TensorDictModule):
    """ Critic for PPO """

    def __init__(
            self,
            modulated_grid_dim,
            device,
            *,
            present_feat_dim=64*6,
    ):

        self.flatten = nn.Flatten()

        self.value_head = PresentValue(
            modulated_grid_dim, present_feat_dim, device)

        super().__init__(self.value_head, in_keys=[
            "critic_data"], out_keys=["state_value"])
