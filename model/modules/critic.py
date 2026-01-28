""" Critic Policy implementation for use with my present packing environment. """
from tensordict import TensorDict
from torch import nn
import torch

from model.modules.feature_extractor import FeatureExtractor


class PresentCritic(nn.Module):
    """ Critic for PPO """

    def __init__(self, device=torch.device("cpu")):
        super().__init__()

        self.flatten = nn.Flatten()
        self.extractor = FeatureExtractor(device)
        self.device = device

        self.critic_head = nn.Sequential(
            nn.Linear(self.extractor.features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(device)

    def forward(self, tensordict):
        """ Forward function for running of nn """
        # get features
        all_features = self.extractor(tensordict)

        # calculate value
        value = self.critic_head(all_features)

        return TensorDict({
            "value": value
        }, batch_size=value.shape[0], device=self.device)
