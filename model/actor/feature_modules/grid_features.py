""" Extracts features from grid """
from torch import nn


class GridExtractor(nn.Module):
    """ Outputs tensor representing extracted grid features """

    def __init__(self, device, output_features=256):
        super().__init__()

        self.device = device

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(16, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, output_features)
        ).to(device)

        self.features = 256

    def forward(self, grid):
        """ Module forward functions - gets grid features """

        grid_features = self.encoder(grid)

        return grid_features
