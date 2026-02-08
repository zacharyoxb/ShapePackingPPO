""" 
First Actor module. Outputs scores for each present orientation
and the modulated grid of each orientation. 
"""
from tensordict import TensorDict
from torch import nn
import torch

from model.modules.actor_modules.td_utils import OrientationEntry
from model.modules.feature_modules.film import FiLM
from model.modules.feature_modules.grid_features import GridExtractor
from model.modules.feature_modules.present_features import PresentExtractor


class PresentSelectionActor(nn.Module):
    """ Policy nn for PresentEnv to choose which present to place. """

    def __init__(self, present_list: list[list[torch.Tensor]], device: torch.device):
        super().__init__()

        self.present_features = []

        self.grid_extractor = GridExtractor(device)
        self.present_extractor = PresentExtractor(device)
        self.film = FiLM(device)

        self._get_present_features(present_list)

    def _get_present_features(self, present_list):
        all_features = []
        for present in present_list:
            orient_features = []
            for orient in present:
                features = self.present_extractor(orient)
                orient_features.append(features)
            all_features.append(orient_features)
        self.present_features = all_features

    def _format_data(self, tensordict):
        grid = tensordict.get("grid")
        present_count = tensordict.get("present_count")

        # If multithreaded, combine into total_batch
        workers, batches = None, None
        # If inputs are unbatched
        if grid.dim() == 2:
            # Add batch/channel
            grid = grid.unsqueeze(0).unsqueeze(0)
            present_count = present_count.unsqueeze(0)
        # If inputs are single batched
        elif grid.dim() == 3:
            # Add channel
            grid = grid.unsqueeze(1)
        # If inputs are double batched
        elif grid.dim() == 4:
            # Add channel
            grid = grid.unsqueeze(2)
            # combine workers and batches
            workers, batches = grid.shape[0], grid.shape[1]
            grid = grid.view(workers * batches, *grid.shape[2:])
            present_count = present_count.view(
                workers * batches, *present_count.shape[2:])

        grid_features = self.grid_extractor(grid)

        return grid_features, present_count, workers, batches

    def _apply_film(self, grid_features, present_features, workers, batches):
        score, modulated_grid = self.film(
            grid_features, present_features)
        if workers and batches:
            modulated_grid = modulated_grid.view(
                workers, batches, -1, -1, -1)

        return score, modulated_grid

    def forward(self, tensordict):
        """ Gets scores for orientation of each present to choose which to place """
        # get data in valid format
        grid_features, present_count, workers, batches = self._format_data(
            tensordict)

        present_data = {}

        # Calculate scores for each orientation
        for present_idx, present_features in enumerate(self.present_features):
            name = "present" + str(present_idx)
            if present_count[present_idx] == 0:
                present_data["present_data"][name] = torch.tensor([])
                continue

            present_orients = TensorDict()
            for orient_idx, orient_features in enumerate(present_features):
                score, modulated_grid = self._apply_film(
                    grid_features,
                    orient_features,
                    workers,
                    batches
                )
                present_orients[orient_idx] = OrientationEntry(
                    score,
                    torch.tensor(present_idx),
                    torch.tensor(orient_idx),
                    self.present_features[present_idx][orient_idx],
                    modulated_grid
                )
            present_data["present_data"][name] = TensorDict(present_orients)

        return TensorDict(present_data)
