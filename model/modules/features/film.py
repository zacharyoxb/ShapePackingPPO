""" Applies Feature-wise Linear Modulation to grid and present features """
import torch.nn as nn


class FiLM(nn.Module):
    """ Outputs a score for each modulated grid from grid and presents"""

    def __init__(self, shape_feat_dim, grid_feat_dim):
        super().__init__()
        # Takes shape features → produces scale & shift for grid features
        self.gamma_beta_net = nn.Sequential(
            nn.Linear(shape_feat_dim, 2 * grid_feat_dim),
            nn.Tanh()  # Optional activation
        )

    def forward(self, grid_features, shape_features):
        """
        grid_features: (B, C_grid, H, W) or (B, H, W, C_grid)
        shape_features: (B, C_shape)
        """
        # Generate scale (gamma) and shift (beta) from shape features
        gamma_beta = self.gamma_beta_net(shape_features)  # (B, 2*C_grid)
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # Each: (B, C_grid)

        # Reshape for broadcasting: (B, C_grid, 1, 1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        # Apply modulation: γ * grid_features + β
        modulated_grid = gamma * grid_features + beta

        return modulated_grid
