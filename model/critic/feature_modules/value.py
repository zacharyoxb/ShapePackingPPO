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
            # Block 1: Local features
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(2),  # HxW → H/2 x W/2

            # Block 2: Medium features
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → H/4 x W/4

            # Block 3: Global features
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, track_running_stats=False),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),

            # Final projection
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        ).to(self.device)

        self.value_head = nn.Sequential(
            nn.Linear(262, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(device)

    def forward(self, tensordict):
        """ Forward function for running of nn """
        # predict value from grid and present count
        grid = tensordict.get("grid")
        present_count = tensordict.get("present_count")

        # check for worker dim
        original_shape = grid.shape
        count_shape = present_count.shape
        if grid.dim() > 4:
            grid = grid.view(-1, *original_shape[2:])
            present_count = present_count.view(-1, *count_shape[2:])

        # put grid through CNN
        grid_features = self.grid_encoder(grid)

        # put grid features and present count together
        combined = torch.cat([grid_features, present_count], dim=-1)

        # calculate value
        value = self.value_head(combined)

        # if there was worker dim, add it again
        if len(original_shape) > 4:
            value = value.view(original_shape[0],
                               original_shape[1], *value.shape[1:])

        batch_size = tensordict.batch_size[0] if tensordict.batch_size else 1
        return TensorDict({
            "state_value": value
        }, batch_size=torch.Size([batch_size]), device=self.device)
