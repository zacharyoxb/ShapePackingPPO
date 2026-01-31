""" Manages saving of models """
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch


CHECKPOINT_DIR = Path(__file__).parent.joinpath("checkpoints")
MODEL_DIR = Path(__file__).parent.joinpath("full_models")


@dataclass
class ModelData:
    """ Layout of saved / checkpointed model data """
    avg_reward: torch.Tensor
    actor_state: dict
    critic_state: dict
    policy_state: dict
    value_state: dict
    loss_state: dict
    optim_state: dict
    scheduler_state: dict


class ModelSaveManager:
    """ 
    Singleton that manages the saving of full models and checkpointing.
    """
    _instance: Optional['ModelSaveManager'] = None

    def __new__(cls) -> 'ModelSaveManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, '_initialized'):
            self.models: list[Path] = self._get_current_models()
            self.ckpts: list[Path] = self._get_current_checkpoints()
            self.max_ckpts = 15
            self._filter_model_data()

            self._initialized = True

            if len(self.ckpts) > self.max_ckpts:
                self._remove_worst_checkpoints()

    def _filter_model_data(self) -> None:
        if not self._instance:
            return

        if not self._instance.ckpts:
            return

        last_modified_ckpt = max(
            self._instance.ckpts, key=lambda model: model.stat().st_mtime)

        if not self._instance.models:
            ckpt = torch.load(f=last_modified_ckpt, weights_only=False)
            self.save(ckpt)
        else:
            last_modified_model = max(
                self._instance.models, key=lambda model: model.stat().st_mtime)

            # if there is a more recent ckpt, save as model
            if last_modified_ckpt > last_modified_model:
                ckpt = torch.load(f=last_modified_ckpt, weights_only=False)
                self.save(ckpt)

    def _get_current_checkpoints(self) -> list[Path]:
        return list(CHECKPOINT_DIR.glob("ckpt_*.pt"))

    def _get_current_models(self) -> list[Path]:
        return list(MODEL_DIR.glob("model_*.pt"))

    def _remove_worst_checkpoints(self):
        if not self._instance:
            return

        ckpt_data = []

        for ckpt in self._instance.ckpts:
            ckpt_data.append(torch.load(ckpt, weights_only=False))

        sorted_ckpts = sorted(ckpt_data, key=lambda data: data.avg_reward)

        # Determine how many to delete
        num_to_delete = len(sorted_ckpts) - self.max_ckpts

        # Delete oldest ones
        for i in range(num_to_delete):
            old_ckpt = sorted_ckpts[i]
            old_ckpt.unlink(missing_ok=True)

        # Update the list
        self.ckpts = sorted_ckpts[num_to_delete:]

    def save(self, model_data):
        """ Saves full model. """
        if self._instance:
            model_name = "model_" + datetime.now().isoformat() + ".pt"
            model_path = Path(MODEL_DIR.joinpath(model_name))
            torch.save(model_data, f=model_path)
            self._instance.models.append(model_path)
            # clear checkpoints over max
            self._remove_worst_checkpoints()

    def save_checkpoint(self, model_data: ModelData):
        """ Adds to checkpoint buffer / saves if buffer is full """
        if self._instance:
            ckpt_name = "ckpt_" + datetime.now().isoformat() + ".pt"
            ckpt_path = Path(CHECKPOINT_DIR.joinpath(ckpt_name))
            torch.save(model_data, f=ckpt_path)
            self._instance.ckpts.append(ckpt_path)

    def load_best(self) -> Optional[ModelData]:
        """ Loads best model """
        if not self._instance:
            return None

        model_data = []

        for model in self._instance.ckpts:
            model_data.append(torch.load(model, weights_only=False))

        if not model_data:
            return None

        best_model = max(model_data, key=lambda data: data.avg_reward)
        return best_model
