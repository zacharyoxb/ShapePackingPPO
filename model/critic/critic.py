""" Critic Policy implementation for use with my present packing environment. """
from tensordict.nn import TensorDictModule
from torch import nn

from model.critic.feature_modules.value import PresentValue


class PresentCritic(TensorDictModule):
    """ Critic for PPO """

    def __init__(
            self,
            device,
    ):
        value_head = PresentValue(
            device)

        super().__init__(value_head, in_keys=[
            "observation"], out_keys=["state_value"])
        self.flatten = nn.Flatten()
