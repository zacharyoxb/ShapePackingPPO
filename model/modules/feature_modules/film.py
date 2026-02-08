""" Applies Feature-wise Linear Modulation to grid and present features """
from torch import nn
import torch


class FiLM(nn.Module):
    """ Outputs a score for each modulated grid from grid and presents """

    def __init__(self, device, present_feat_dim=64, hidden_dim=256):
        super().__init__()

        self.device = device
        self.present_feat_dim = present_feat_dim
        self.hidden_dim = hidden_dim

        # FiLM generator: from shape features to modulation parameters
        self.film_generator = nn.Sequential(
            nn.Linear(self.present_feat_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2)
        ).to(device)

        # Scoring network (after modulation)
        self.scoring_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1)
        ).to(device)

    def forward(self, grid_features, present_features):
        """
        grid_features: (B, feature_dim)
        present_features: (B, feature_dim)
        """

        # Generate FiLM parameters for this shape
        film_params = self.film_generator(
            present_features)  # [batch, hidden*2]
        gamma, beta = torch.chunk(film_params, 2, dim=-1)

        # Apply FiLM modulation to grid features
        modulated_grid = gamma * grid_features + beta

        # Score this shape given its modulated view of grid
        shape_score = self.scoring_net(modulated_grid)  # [batch, 1]

        return shape_score, modulated_grid
