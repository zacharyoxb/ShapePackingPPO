""" Extracts features from presents (3x3 shapes) """
from torch import nn


class PresentExtractor(nn.Module):
    """ 
    Outputs tensor representing all features of Present
    """

    def __init__(self, device, output_features=64):
        super().__init__()

        self.device = device
        self.output_features = output_features

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2, padding=1),
            nn.BatchNorm2d(16, track_running_stats=False),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, track_running_stats=False),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, output_features)
        ).to(self.device)

    def forward(self, present):
        """ Module forward function - gets present features """

        present_features = self.encoder(present)

        return present_features
