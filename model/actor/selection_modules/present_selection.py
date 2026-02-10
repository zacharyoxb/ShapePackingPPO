""" 
First Actor module. Outputs scores for each present orientation
and the modulated grid of each orientation. 
"""
from tensordict import TensorDict
from torch import nn
import torch

from model.actor.selection_modules.td_utils import OrientationEntry
from model.actor.feature_modules.film import FiLM
from model.actor.feature_modules.grid_features import GridExtractor
from model.actor.feature_modules.present_features import PresentExtractor


class PresentSelectionActor(nn.Module):
    """ Policy nn for PresentEnv to choose which present to place. """

    def __init__(self, presents: torch.Tensor, device: torch.device):
        super().__init__()

        # this will probably be a problem because of save: leave for now
        self.all_present_features = []

        self.grid_extractor = GridExtractor(device)
        self.present_extractor = PresentExtractor(device)
        self.film = FiLM(device)

        self._feature_init(presents)

    def _feature_init(self, presents):
        all_features = []

        for present in presents:
            orient_features = []
            for orient in present:
                # add channel dim
                present = present.unsqueeze(0)

                features = self.present_extractor(orient)
                orient_features.append(features)
            all_features.append(orient_features)
        self.all_present_features = all_features

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
        """ Gets scores for orientation / modulated grids for them """
        # get data in valid format
        grid_features, present_count, workers, batches = self._format_data(
            tensordict)

        orient_data = {}
        orient_logits = torch.tensor([])

        # Calculate scores for each orientation
        for present_idx, present_features in enumerate(self.all_present_features):
            if present_count[present_idx] == 0:
                continue

            for orient_idx, orient_features in enumerate(present_features):
                score, modulated_grid = self._apply_film(
                    grid_features,
                    orient_features,
                    workers,
                    batches
                )
                orient_logits.add(score)
                orient_data[f"{present_idx}:{orient_idx}"] = TensorDict.from_dataclass(
                    OrientationEntry(
                        torch.tensor(present_idx),
                        torch.tensor(orient_idx),
                        self.all_present_features[present_idx][orient_idx],
                        modulated_grid
                    )
                )

        return TensorDict({
            "present_data": {
                "logits": orient_logits,
                "orientations": TensorDict(orient_data)
            }})
