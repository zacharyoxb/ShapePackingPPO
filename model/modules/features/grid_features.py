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
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # HxW → H/2 x W/2

            # Block 2: Medium features
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → H/4 x W/4

            # Block 3: Global features
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
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

    def forward(self, tensordict):
        """ Module forward functions - gets grid features """
        grid = tensordict.get("grid")

        workers, batches = None, None

        # If input is not batched
        if grid.dim() == 2:
            # Add batch/channel
            grid = grid.unsqueeze(0).unsqueeze(0)
        # If input is single batched
        elif grid.dim() == 3:
            # Add channel
            grid = grid.unsqueeze(-3)
        # If input is double batched
        elif grid.dim() == 4:
            # Add channel
            grid = grid.unsqueeze(-3)
            # combine workers and batches
            workers, batches = grid.shape[0], grid.shape[1]
            grid = grid.view(workers * batches, *grid.shape[2:])

        grid_features = self.encoder(grid)

        # restore dims if double batched
        if workers and batches:
            grid_features = grid_features.view(workers, batches, -1)

        return grid_features
