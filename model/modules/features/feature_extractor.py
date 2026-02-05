""" Extracts features from data """
from torch import nn

from model.modules.features.film import FiLM
from model.modules.features.grid_features import GridExtractor
from model.modules.features.present_features import PresentExtractor


class FeatureExtractor(nn.Module):
    """ Outputs tensor representing extracted features """

    def __init__(self, device):
        super().__init__()

        self.device = device

        # Encodes grid
        self.grid_encoder = GridExtractor(self.device, output_features=256)

        # Encodes all presents
        self.present_encoder = PresentExtractor(
            self.device, ind_output_features=64)

        # Applies FiLM modulation to presents
        self.modulator = FiLM(64, 256)

        self.features = 256

    def forward(self, tensordict):
        """ Module forward functions - gets data features """
        # Extract grid / present features
        grid_features = self.grid_encoder(tensordict)
        present_features = self.present_encoder(tensordict)

        modulated_features = self.modulator(grid_features, present_features)

        return modulated_features
