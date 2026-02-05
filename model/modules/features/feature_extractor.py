""" Extracts features from data """
from torch import nn
import torch

from model.modules.features.grid_features import GridExtractor
from model.modules.features.present_features import PresentExtractor

PRESENT_COUNT = 6


class FeatureExtractor(nn.Module):
    """ Outputs tensor representing extracted features """

    def __init__(self, device):
        super().__init__()

        self.device = device

        # Encodes grid
        self.grid_encoder = GridExtractor(self.device, output_features=256)

        # Encodes all presents
        self.present_encoder = PresentExtractor(
            self.device, num_presents=PRESENT_COUNT, ind_output_features=64)

        self.features = 256 + 64

    def forward(self, tensordict):
        """ Module forward functions - gets data features """
        # Extract grid / present features
        grid_features = self.grid_encoder(tensordict)
        present_features = self.present_encoder(tensordict)

        features = torch.cat(grid_features, present_features)

        return features
