""" Extracts features from data """
from torch import nn
import torch

PRESENT_COUNT = 6


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

        # Encodes all presents
        self.present_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 128)
        ).to(device)

        self.present_count_encoder = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 128)
        ).to(device)

        # Cross-attention: Grid (query) attends to Presents (key/value)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=128,      # Must match your feature dimension
            num_heads=4,        # 4 heads for 128-dim (128/4=32 per head)
            batch_first=True,   # IMPORTANT: (batch, seq_len, embed_dim)
            dropout=0.1         # Optional regularization
        ).to(device)

        self.layer_norm = nn.LayerNorm(128)

        combined_features = 128 + 128 + 128

        self.fusion = nn.Sequential(
            nn.Linear(combined_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128)  # [batch, 128]
        ).to(device)

        self.features = 128

    def _add_dims(self, grid, presents, present_count):
        # If multithreaded, combine into total_batch
        workers, batches = None, None
        # If inputs are unbatched
        if grid.dim() == 2:
            # Add batch/channel
            grid = grid.unsqueeze(0).unsqueeze(0)
            # Add batch/channel
            presents = presents.unsqueeze(0).unsqueeze(0)
            # Add batch
            present_count = present_count.unsqueeze(0)
        # If inputs are single batched
        elif grid.dim() == 3:
            # Add channel
            grid = grid.unsqueeze(-3)
            # Add channel
            presents = presents.unsqueeze(-4)
        # If inputs are double batched
        elif grid.dim() == 4:
            # Add channel
            grid = grid.unsqueeze(-3)
            # Add channel
            presents = presents.unsqueeze(-4)

            # combine workers and batches
            workers, batches = grid.shape[0], grid.shape[1]
            grid = grid.view(workers * batches, *grid.shape[2:])
            presents = presents.view(workers * batches, *presents.shape[2:])
            present_count = present_count.view(
                workers * batches, *present_count.shape[2:])

        return grid, presents, present_count, workers, batches

    def _get_present_features(self, presents):
        # Single batch dimension: [batch, channels, num_presents, h, w]
        batch_size, channels, num_presents, h, w = presents.shape

        presents_reshaped = presents.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_presents, channels, h, w
        )

        features = self.present_encoder(presents_reshaped)

        # Reshape back: [batch, num_presents, feature_dim]
        features = features.view(batch_size, num_presents, -1)

        return features

    def _apply_cross_attention(self, grid_features, present_features):
        grid_query = grid_features.unsqueeze(-2)

        # Apply multi-head attention
        attended_output, attention_weights = self.cross_attention(
            query=grid_query,        # [batch, 1, 128]
            key=present_features,    # [batch, num_presents, 128]
            value=present_features,  # [batch, num_presents, 128]
            need_weights=True        # Return attention weights for visualization
        )

        attended_presents = attended_output.squeeze(1)
        attended_presents = self.layer_norm(attended_presents)

        return attended_presents, attention_weights

    def forward(self, tensordict):
        """ Module forward functions - gets data features """
        # Add dimensions required to encode features
        grid, presents, present_count, workers, batches = self._add_dims(
            tensordict.get("grid").detach().clone(),
            tensordict.get("presents").detach().clone(),
            tensordict.get("present_count").detach().clone()
        )

        # Extract grid/present/present_count features
        grid_features = self.grid_encoder(grid)
        present_features = self._get_present_features(presents)
        count_features = self.present_count_encoder(present_count)

        # Get cross-attention weights
        attended_presents, _attention_weights = self._apply_cross_attention(
            grid_features, present_features)

        combined_features = torch.cat([
            grid_features,
            attended_presents,
            count_features
        ], dim=1)

        # Combine all features
        fused_features = self.fusion(combined_features)

        # If workers and batch have been combined, restore dims
        if workers is not None:
            fused_features = fused_features.view(workers, batches, -1)

        return fused_features  # , attention_weights
