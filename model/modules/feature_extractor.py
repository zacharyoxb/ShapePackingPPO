""" Extracts features from data """
from torch import nn
import torch


class FeatureExtractor(nn.Module):
    """ Outputs tensor representing extracted features """

    def __init__(self, device):
        super().__init__()

        self.device = device

        # Encodes grid
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
            nn.Linear(64, 128)  # [batch, 128]
        ).to(device)

        # Encodes each present individually
        self.present_encoders = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()  # [batch, 16]
            ) for _ in range(6)
        ).to(device)

        self.present_count_encoder = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        ).to(device)  # [batch, 16]

        combined_features = 128 + (16 * 6) + 16

        self.fusion = nn.Sequential(
            nn.Linear(combined_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128)  # [batch, 128]
        ).to(device)

        self.features = 128

    def forward(self, tensordict):
        """ Module forward functions - gets data features """
        grid = tensordict.get("grid")
        presents = tensordict.get("presents")
        present_count = tensordict.get("present_count")

        # Check if batch (and channel, if using conv2d) dimensions exist
        if grid.dim() < 3:
            grid = grid.unsqueeze(0).unsqueeze(0)
        else:
            grid = grid.unsqueeze(1)

        if presents.dim() < 4:
            presents = presents.unsqueeze(0)

        if present_count.dim() < 2:
            present_count = present_count.unsqueeze(0)

        # Extract grid/present_count features
        grid_features = self.grid_encoder(grid)
        all_features = grid_features

        # Encode each present with its own encoder
        for i, encoder in enumerate(self.present_encoders):
            # [batch, present_idx, height, width] -> add channel dim
            present = presents[:, i, :, :].unsqueeze(1)
            encoded_present = encoder(present)
            all_features = torch.cat([all_features, encoded_present], dim=1)

        count_features = self.present_count_encoder(present_count)
        all_features = torch.cat([all_features, count_features], dim=1)

        # Combine all features
        fused_features = self.fusion(all_features)

        return fused_features
