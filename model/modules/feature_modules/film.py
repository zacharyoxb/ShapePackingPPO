""" Applies Feature-wise Linear Modulation to grid and present features """
from torch import nn
import torch


class FiLM(nn.Module):
    """ Outputs a score for each modulated grid from grid and presents"""

    def __init__(self, present_feat_dim=64, hidden_dim=256):
        super().__init__()

        self.present_feat_dim = present_feat_dim
        self.hidden_dim = hidden_dim

        # FiLM generator: from shape features to modulation parameters
        self.film_generator = nn.Sequential(
            nn.Linear(self.present_feat_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2)
        )

        # Scoring network (after modulation)
        self.scoring_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1)
        )

    def forward(self, grid_features, present_features):
        """
        grid_features: (W, B, feature_dim) or (B, feature_dim)
        present_features: (W, B, present_idx, feature_dim) or (B, present_idx, feature_dim)
        """

        workers, batches = None, None

        # If features are double batched, reshape
        if grid_features.dim() == 3:
            workers, batches = grid_features.shape[0], grid_features.shape[1]
            grid_features = grid_features.view(
                workers * batches, *grid_features.shape[2:]
            )
            present_features = present_features.view(
                workers * batches, *present_features.shape[2:]
            )

        scored_grids = {}

        for i in range(present_features.shape[1]):
            # Get shape features
            shape_feat = present_features[:, i, :]  # [batch, shape_feat_dim]

            # Generate FiLM parameters for this shape
            film_params = self.film_generator(shape_feat)  # [batch, hidden*2]
            gamma, beta = torch.chunk(film_params, 2, dim=-1)

            # Apply FiLM modulation to grid features
            modulated_grid = gamma * grid_features + beta

            # Score this shape given its modulated view of grid
            shape_score = self.scoring_net(modulated_grid)  # [batch, 1]
            scored_grids[i] = {"score": shape_score, "grid": modulated_grid}

        return scored_grids
