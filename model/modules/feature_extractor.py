""" Extracts features from data """
from torch import nn
import torch


class FeatureExtractor(nn.Module):
    """ Outputs tensor representing extracted features """

    def __init__(self):
        super().__init__()

        # Process grid of [batch, channels, height, width]
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128)
        )

        # Process grid of [batch, channels, height, width]
        self.present_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 3 * 3, 128)  # 32 channels × 3×3 spatial
        )

        # Present count encoder (simple scalar)
        self.present_count_encoder = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        self.combined_features = 128 + 128 + 32

    def forward(self, tensordict):
        """ Module forward functions - gets data features """
        # Clone to avoid modifying original tensordict
        grid = tensordict.get("grid").detach().clone()
        presents = tensordict.get("presents").detach().clone()
        present_count = tensordict.get("present_count").detach().clone()

        # Process grid: ensure [batch, 1, height, width]
        grid_features = self._process_grid(grid)

        # Process presents: handle 6 presents in parallel
        present_features = self._process_multiple_presents(presents)
        # Aggregate so batch size matches others
        present_features = present_features.mean(dim=0, keepdim=True)

        # Process present count: ensure [batch, 1]
        count_features = self._process_present_count(present_count)

        all_features = torch.cat([
            grid_features,
            present_features,
            count_features
        ], dim=1)

        return all_features

    def _process_grid(self, grid):
        grid = grid.unsqueeze(0).unsqueeze(0)  # [1, 1, height, width]
        return self.grid_encoder(grid)

    def _process_multiple_presents(self, presents):
        presents = presents.unsqueeze(1)  # [6, 3, 3] -> [6, 1, 3, 3]
        return self.present_encoder(presents)

    def _process_present_count(self, count):
        count = count.unsqueeze(0)  # [batch, 1]
        return self.present_count_encoder(count)
