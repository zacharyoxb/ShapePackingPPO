""" 
First Actor module. Outputs scores for each present orientation
and the modulated grid of each orientation. 
"""
from torch import nn

from model.modules.feature_modules.film import FiLM
from model.modules.feature_modules.grid_features import GridExtractor
from model.modules.feature_modules.present_features import PresentExtractor


class PresentPositionActor(nn.Module):
    """ Policy nn for PresentEnv to choose which present to place. """

    def __init__(self, device):
        super().__init__()

        self.grid_extractor = GridExtractor(device)
        self.present_extractor = PresentExtractor(device)
        self.film = FiLM(device)

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

    def forward(self, tensordict):
        """ Choose a present based on scores, output the present """
