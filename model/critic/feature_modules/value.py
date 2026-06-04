""" Predicts how well we can do based on current state. """
from tensordict import TensorDict
from torch import nn
import torch


class PresentValue(nn.Module):
    """ Value for PPO """

    def __init__(
            self,
            device,
    ):
        super().__init__()

        self.flatten = nn.Flatten()
        self.device = device

        self.grid_encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        ).to(device)

        self.value_head = nn.Sequential(
            nn.Linear(262, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(device)

    def forward(self, tensordict):
        """ Forward function for running of nn """
        grid = tensordict.get("grid")
        present_count = tensordict.get("present_count")

        if grid.dim() > 4:
            grid_features = torch.vmap(self.grid_encoder)(grid)
        else:
            grid_features = self.grid_encoder(grid)

        combined = torch.cat([grid_features, present_count], dim=-1)

        value = self.value_head(combined)

        batch_size = tensordict.batch_size[0] if tensordict.batch_size else 1
        return TensorDict({
            "state_value": value
        }, batch_size=torch.Size([batch_size]), device=self.device)
