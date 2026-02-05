""" Extracts features from presents (3x3 shapes) """
from torch import nn
import torch


class PresentExtractor(nn.Module):
    """ Outputs tensor representing extracted present features """

    def __init__(self, device, num_presents=6, ind_output_features=64):
        super().__init__()

        self.device = device
        self.output_features = ind_output_features
        self.num_presents = num_presents

        self.encoder = nn.Sequential(
            # First conv: extract basic shape patterns
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # Second conv: combine local features
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Flatten and project to desired size
            nn.Flatten(),
            nn.Linear(32 * 3 * 3, 128),  # 32 channels * 3x3 = 288 → 128
            nn.ReLU(),
            nn.Linear(128, ind_output_features)  # Final output size
        ).to(self.device)

    def forward(self, tensordict):
        """ Module forward function - gets present features """
        presents = tensordict.get(
            "presents")

        # If multithreaded, combine into total_batch
        workers, batches = None, None
        # If inputs are unbatched
        if presents.dim() == 3:
            # Add batch/channel
            presents = presents.unsqueeze(0).unsqueeze(0)
        # If inputs are single batched
        elif presents.dim() == 4:
            # Add channel
            presents = presents.unsqueeze(1)
        # If inputs are double batched
        elif presents.dim() == 5:
            # Add channel
            presents = presents.unsqueeze(2)
            # combine workers and batches
            workers, batches = presents.shape[0], presents.shape[1]
            presents = presents.view(workers * batches, *presents.shape[2:])

        all_present_features = []
        for i in range(self.num_presents):
            present = presents[:, :, i, :, :]
            present_features = self.encoder(present)
            all_present_features.append(present_features)

        # Combine all features
        present_features = torch.stack(all_present_features, dim=1)

        # Restore dimensions if double batched
        if workers is not None and batches is not None:
            present_features = present_features.view(workers, batches, -1)

        return present_features
