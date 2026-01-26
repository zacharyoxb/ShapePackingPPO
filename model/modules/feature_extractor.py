""" Extracts features from data """
from torch import nn
import torch


class FeatureExtractor(nn.Module):
    """ Outputs tensor representing extracted features """

    def __init__(self):
        super().__init__()

        # Process grid
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

        # Process present
        self.present_encoder = nn.Sequential(
            nn.Linear(6 * 3 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Process present_count
        self.present_count_encoder = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.combined_features = 128 + 128 + 32

    def forward(self, tensordict):
        """ Module forward functions - gets data features """
        grid = tensordict.get(("observation", "grid")
                              ).unsqueeze(0).unsqueeze(0)
        presents = tensordict.get(("observation", "presents")).unsqueeze(0)
        present_count = tensordict.get(
            ("observation", "present_count")).unsqueeze(0)

        all_features = torch.cat([
            self.grid_encoder(grid),
            self.present_encoder(
                presents.flatten(start_dim=1)),
            self.present_count_encoder(
                present_count)
        ], dim=1)

        return all_features
