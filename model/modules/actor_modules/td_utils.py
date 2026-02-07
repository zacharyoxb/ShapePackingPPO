""" Manages my own custom tensordicts for easy adding / reading of data """

from dataclasses import dataclass
from tensordict import TensorDict
import torch


@dataclass
class OrientationEntry:
    """ 
    The values of an entry of an orientation of a present.

    rot: Amount present was rotated
    flip: 2 value tensor representing on which axis it was flipped
    score: The score the modulated grid was given
    features: the features of the orientation
    modulated_grid: The grid features to use when placing present
    """
    rot: torch.Tensor
    flip: torch.Tensor
    score: torch.Tensor
    features: torch.Tensor
    modulated_grid: torch.Tensor


class PresentData:
    """ Class to store all data from present_selection actor """

    def __init__(self, td):
        self._td = td

    @classmethod
    def create_empty(cls, present_num: torch.Tensor, batch_size: torch.Size):
        """ Creates empty PresentLogits instance """
        td = TensorDict(batch_size=batch_size)
        for i in range(present_num):
            # Add an entry for present
            present_name = "present" + str(i)
            td[present_name] = None
        return PresentData(td)

    def add_orientation(self, idx: torch.Tensor, orientation: OrientationEntry):
        """ Adds an orientation entry to the present. """
        present_name = "present" + str(idx)
        self._td[present_name] = TensorDict.from_dataclass(orientation)
