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
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        self.combined_features = 128 + 128 + 32

    def forward(self, tensordict):
        """ Module forward functions - gets data features """
        grid = tensordict.get("grid").clone()
        presents = tensordict.get("presents").clone()
        present_count = tensordict.get("present_count").clone()

        grid = grid.unsqueeze(0).unsqueeze(0)
        presents = presents.unsqueeze(0)
        present_count.unsqueeze(0)

        all_features = torch.cat([
            self.grid_encoder(grid),
            self.present_encoder(
                presents.flatten(start_dim=1)),
            self.present_count_encoder(
                present_count)
        ], dim=1)

        return all_features
