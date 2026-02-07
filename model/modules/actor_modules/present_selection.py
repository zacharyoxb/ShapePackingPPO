""" 
First Actor module. Outputs scores for each present orientation
and the modulated grid of each orientation. 
"""
from torch import nn
import torch

from model.modules.actor_modules.td_utils import OrientationEntry
from model.modules.feature_modules.film import FiLM
from model.modules.feature_modules.grid_features import GridExtractor
from model.modules.feature_modules.present_features import PresentExtractor


class PresentSelectionActor(nn.Module):
    """ Policy nn for PresentEnv to choose which present to place. """

    def __init__(self, device=torch.device("cpu")):
        super().__init__()

        self.grid_extractor = GridExtractor(device)
        self.present_extractor = PresentExtractor(device)
        self.film = FiLM(device)

    def _get_orientation(self, grid_features, present_orient, rot, flip):
        # Get features of orientation
        orient_features = self.present_extractor(present_orient)
        # Apply film layer, get score and modulated grid
        score, modulated_grid = self.film(grid_features, orient_features)

        return OrientationEntry(rot, flip, score, orient_features, modulated_grid)

    def _format_data(self, tensordict):
        grid = tensordict.get("grid")
        presents = tensordict.get("presents")
        present_count = tensordict.get("present_count")

        # If multithreaded, combine into total_batch
        workers, batches = None, None
        # If inputs are unbatched
        if grid.dim() == 2:
            # Add batch/channel
            grid = grid.unsqueeze(0).unsqueeze(0)
            presents = presents.unsqueeze(0).unsqueeze(0)
            present_count = present_count.unsqueeze(0)
        # If inputs are single batched
        elif grid.dim() == 3:
            # Add channel
            grid = grid.unsqueeze(1)
            presents = presents.unsqueeze(1)
        # If inputs are double batched
        elif grid.dim() == 4:
            # Add channel
            grid = grid.unsqueeze(2)
            presents = presents.unsqueeze(2)
            # combine workers and batches
            workers, batches = grid.shape[0], grid.shape[1]
            grid = grid.view(workers * batches, *grid.shape[2:])
            presents = presents.view(workers * batches, *presents.shape[2:])
            present_count = present_count.view(
                workers * batches, *present_count.shape[2:])

        return grid, presents, present_count, workers, batches

    def _unique_orientation_gen(self, present):
        seen = set()
        flat = present.flatten()

        for k in range(4):
            rotated = torch.rot90(present, k=k, dims=[-2, -1])
            flat = rotated.flatten()
            if flat not in seen:
                seen.add(flat)
                yield torch.tensor(k), torch.tensor([]), rotated

        # Flip along vertical axis, horizontal and both
        for flip_dims in ([-1], [-2], [-1, -2]):
            flipped = torch.flip(present, dims=flip_dims)

            for k in range(4):
                # Rotate the flipped version
                rotated_flipped = torch.rot90(flipped, k=k, dims=[-2, -1])
                flat = rotated_flipped.flatten()
                if flat not in seen:
                    seen.add(flat)
                    yield torch.tensor(k), torch.tensor([-2]), rotated_flipped

    def forward(self, tensordict):
        """ Gets scores for orientation of each present to choose which to place """
        # get data in valid format
        grid, presents, _present_count, workers, batches = self._format_data(
            tensordict)

        # Get raw grid features
        grid_features = self.grid_extractor(grid)

        orients = []

        # Get raw present features for each present
        for idx in range(presents.shape[-2]):
            present = presents[:, :, idx, :, :]
            for rot, flip, oriented in self._unique_orientation_gen(present):
                present_features = self.present_extractor(oriented)
                score, modulated_grid = self.film(
                    grid_features, present_features)

                orient = OrientationEntry(
                    rot, flip, score, present_features, modulated_grid)

                orients.append(orient)

        # If data came in double batched, return it as such
        if workers and batches:
            grid_features = grid_features.view(workers, batches, -1)
