""" Predicts how well we can do based on current state. """
from tensordict import TensorDict
from torch import nn
import torch


class PresentValue(nn.Module):
    """ Value for PPO """

    def __init__(
            self,
            modulated_grid_dim,
            present_feat_dim,
            device,
    ):
        super().__init__()

        self.flatten = nn.Flatten()
        self.device = device

        self.features = modulated_grid_dim + present_feat_dim

        self.value_head = nn.Sequential(
            nn.Linear(self.features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(device)

    def forward(self, tensordict):
        """ Forward function for running of nn """
        # get features of modulated grids and presents from td
        present_features = tensordict.get("present_features")
        modulated_grids = tensordict.get("modulated_grids")

        all_features = torch.cat([present_features, modulated_grids])

        # calculate value
        value = self.value_head(all_features)

        batch_size = tensordict.batch_size[0] if tensordict.batch_size else 1
        return TensorDict({
            "state_value": value
        }, batch_size=torch.Size([batch_size]), device=self.device)
