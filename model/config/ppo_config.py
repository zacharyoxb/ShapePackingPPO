""" Config classes for PPO """
import multiprocessing
from dataclasses import dataclass, field

import torch


@dataclass
class ProcessingParameters:
    """ Parameters referring to processing power used. """
    num_workers = 5


@dataclass
class Hyperparameters:
    """ Hyperparameters """
    is_fork = multiprocessing.get_start_method() == "fork"
    device = (
        torch.device(0)
        if torch.cuda.is_available() and not is_fork
        else torch.device("cpu")
    )
    lr = 3e-4
    max_grad_norm = 0.5


@dataclass
class DataCollection:
    """ Data collection parameters """
    frames_per_batch = 1000
    total_frames = 1_500_000


@dataclass
class PPOParameters:
    """ Loss function weights and coefficients """
    sub_batch_size = 64
    num_epochs = 5
    clip_epsilon = 0.2
    gamma = 0.99
    lmbda = 0.95
    entropy_eps = 0.01


@dataclass
class PPOConfig:
    """Complete PPO Hyperparameters Configuration"""

    processing_parameters: ProcessingParameters = field(
        default=ProcessingParameters())
    hyper_parameters: Hyperparameters = field(default=Hyperparameters())
    data_collection: DataCollection = field(default=DataCollection())
    ppo_parameters: PPOParameters = field(default=PPOParameters())

    @property
    def num_workers(self):
        """ How many workers to use in the env """
        return self.processing_parameters.num_workers

    @property
    def is_fork(self):
        """ If process is forked """
        return self.hyper_parameters.is_fork

    @property
    def device(self):
        """ Device being used """
        return self.hyper_parameters.device

    @property
    def lr(self):
        """ Learning rate """
        return self.hyper_parameters.lr

    @property
    def max_grad_norm(self):
        """ Gradient clipping threshold """
        return self.hyper_parameters.max_grad_norm

    @property
    def frames_per_batch(self):
        """ Environment steps collected per training batch """
        return self.data_collection.frames_per_batch

    @property
    def total_frames(self):
        """ Total environment steps for training """
        return self.data_collection.total_frames

    @property
    def sub_batch_size(self):
        """ Batch Size: Mini-batch size for optimization """
        return self.ppo_parameters.sub_batch_size

    @property
    def num_epochs(self):
        """ PPO Epochs: Number of optimization passes per batch """
        return self.ppo_parameters.num_epochs

    @property
    def clip_epsilon(self):
        """ PPO clipping parameter """
        return self.ppo_parameters.clip_epsilon

    @property
    def gamma(self):
        """ Discount factor for future rewards """
        return self.ppo_parameters.gamma

    @property
    def lmbda(self):
        """ GAE (Generalized Advantage Estimation) parameter """
        return self.ppo_parameters.lmbda

    @property
    def entropy_eps(self):
        """ Entropy Coefficient: Exploration bonus """
        return self.ppo_parameters.entropy_eps
