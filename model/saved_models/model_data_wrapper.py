""" Wrapper for model data """
import io
from pathlib import Path

import torch

from model.saved_models.model_data import ModelData


class ModelDataWrapper:
    """ Wrapper for model data that supports file-like operations """

    def __init__(self, path: Path):
        self.path = path
        self._data = None

    def __lt__(self, other):
        return self.avg_reward < other.avg_reward

    def get_data(self) -> ModelData:
        """ Get the ModelData object """
        if self._data is not None:
            return self._data

        with open(self.path, 'rb') as f:
            buffer = io.BytesIO(f.read())
        buffer.seek(0)
        self._data = torch.load(buffer, weights_only=False)

        return self._data

    def save_to_disk(self, data):
        """ Save to disk """
        torch.save(data, self.path)

        return self

    def unlink(self):
        """ Delete the checkpoint file """
        if self.path and self.path.exists():
            self.path.unlink(missing_ok=True)

    @property
    def avg_reward(self):
        """ Get average reward """
        if self._data is not None:
            return self._data.avg_reward

        return self.get_data().avg_reward
