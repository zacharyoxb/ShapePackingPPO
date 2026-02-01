""" Manages saving of models """
from datetime import datetime
from pathlib import Path
from typing import Optional

from model.saved_models.model_data import ModelData
from model.saved_models.model_data_wrapper import ModelDataWrapper


CHECKPOINT_DIR = Path(__file__).parent.joinpath("checkpoints")
MODEL_DIR = Path(__file__).parent.joinpath("full_models")


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
            self.ckpts: list[ModelDataWrapper] = self._get_current_checkpoints()
            self.models: list[ModelDataWrapper] = self._get_current_models()
            self.last_cleanup = 0  # ckpt saves since last cleanup
            self._initialized = True

    def _get_current_checkpoints(self) -> list[ModelDataWrapper]:
        ckpts = []
        for path in CHECKPOINT_DIR.glob("ckpt_*.pt"):
            wrapper = ModelDataWrapper(path)
            ckpts.append(wrapper)
        return ckpts

    def _get_current_models(self) -> list[ModelDataWrapper]:
        models = []
        for path in MODEL_DIR.glob("model_*.pt"):
            wrapper = ModelDataWrapper(path)
            models.append(wrapper)
        return models

    def save(self, model_data):
        """ Saves full model. """
        if self._instance:
            model_name = "model_" + datetime.now().isoformat() + ".pt"
            model_path = Path(MODEL_DIR.joinpath(model_name))
            wrapper = ModelDataWrapper(model_path)
            wrapper.save_to_disk(model_data)
            self._instance.models.append(wrapper)

    def save_checkpoint(self, model_data: ModelData):
        """ Adds to checkpoint buffer / saves if buffer is full """
        if self._instance:
            # check if cleanup is due
            if self.last_cleanup > 10:
                self.ckpt_cleanup()
                self.last_cleanup = 0
            else:
                self.last_cleanup += 1

            ckpt_name = "ckpt_" + datetime.now().isoformat() + ".pt"
            ckpt_path = Path(CHECKPOINT_DIR.joinpath(ckpt_name))
            wrapper = ModelDataWrapper(ckpt_path)
            wrapper.save_to_disk(model_data)
            self._instance.ckpts.append(wrapper)

    def load_best(self) -> Optional[ModelData]:
        """ Loads best model """
        if not self._instance:
            return None

        # no saved models/checkpoints available
        if not self.models and not self.ckpts:
            return None

        # get best model / checkpoint
        if self.models:
            best_model = max(self.models)
        else:
            best_model = max(self.ckpts)

        return best_model.get_data()

    def ckpt_cleanup(self):
        """ Deletes all but the 5 best checkpoints. """
        sorted_ckpts = sorted(self.ckpts, reverse=True)
        self.ckpts = sorted_ckpts[:5]
