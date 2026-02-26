""" Extracts features from grid """
from torch import nn


class GridExtractor(nn.Module):
    """ Outputs tensor representing extracted grid features """

    def __init__(self, device, output_features=256):
        super().__init__()

        self.device = device

        self.encoder = nn.Sequential(
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
            nn.Conv2d(256, output_features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        ).to(self.device)

        self.features = 256

    def forward(self, grid):
        """ Module forward functions - gets grid features """

        # if worker dim exists, combine with batch dim
        original_shape = grid.shape
        if grid.dim() > 4:
            grid = grid.view(-1, *original_shape[2:])

        grid_features = self.encoder(grid)

        # if there was worker dim, add it again
        if len(original_shape) > 4:
            grid_features = grid_features.view(original_shape[0],
                                               original_shape[1], *grid_features.shape[1:])

        return grid_features
