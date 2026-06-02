""" Code for the neural network itself. """

from collections import defaultdict
import random

import torch
from tqdm import tqdm
from torchrl.collectors import SyncDataCollector
from torchrl.objectives.value import GAE
from torchrl.objectives import ClipPPOLoss
from tensordict.nn import (
    set_interaction_type,
    InteractionType
)

from model.actor import PresentActorSeq
from model.critic import PresentCritic
from model.env import PresentEnv
from model.config.ppo_config import PPOConfig
from model.saved_models.save_manager import ModelData, ModelSaveManager
from data.data_reader import get_state_td, get_all_present_orientations


class PPO:
    """ Implementation of the PPO actor-critic network """

    def __init__(self, input_name="input.txt", config=None, training_device=None):
        self.training_device = training_device or (
            torch.device('cuda') if torch.cuda.is_available(
            ) else torch.device('cpu')
        )
        self.config = config or PPOConfig()
        self.input_td = get_state_td(input_name)
        self.presents = get_all_present_orientations(
            input_name, self.training_device)

        # Set up Actor and Critic
        self.policy_module = PresentActorSeq(
            self.presents, self.training_device)

        self.value_module = PresentCritic(self.training_device)

        # Loss function config
        self.advantage_module = GAE(
            gamma=self.config.gamma,
            lmbda=self.config.lmbda,
            value_network=self.value_module,
            average_gae=True,
            device=self.training_device
        )

        self.loss_module = ClipPPOLoss(
            actor_network=self.policy_module,
            critic_network=self.value_module,
            clip_epsilon=self.config.clip_epsilon,
            entropy_bonus=bool(self.config.entropy_eps),
            entropy_coeff=self.config.entropy_eps,
            critic_coeff=1.0,
            loss_critic_type="smooth_l1",
            device=self.training_device
        )

        self.optim = torch.optim.Adam(
            self.loss_module.parameters(),
            lr=self.config.lr,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, self.config.total_frames // self.config.frames_per_batch, 0.0
        )

        self.load_from_models()

    def load_from_models(self):
        """ If model already exists in full_models, load it """
        manager = ModelSaveManager()
        best = manager.load_latest()

        if best:
            self.policy_module.load_state_dict(best.policy_state)
            self.value_module.load_state_dict(best.value_state)
            self.loss_module.load_state_dict(best.loss_state, strict=False)
            self.policy_module.eval()
            self.value_module.eval()
            self.loss_module.eval()

            self.optim.load_state_dict(best.optim_state)
            self.scheduler.load_state_dict(best.scheduler_state)

    def train(self, td=None, logs=None):
        """ Trains single set of data """

        if logs is None:
            logs = defaultdict(list)

        if td is None:
            td = random.choice(self.input_td)

        env = PresentEnv.make_parallel_env(
            start_state=td,
            num_workers=self.config.num_workers,
            device=torch.device("cpu")
        )

        collector = SyncDataCollector(
            env,
            self.policy_module,
            frames_per_batch=self.config.frames_per_batch,
            total_frames=self.config.total_frames,
            # prevents linux CUDA permission err
            device=torch.device("cpu"),
            storing_device=torch.device("cpu"),
            policy_device=self.training_device,
        )

        dataset_progress = tqdm(
            total=self.config.total_frames,
            desc="Current Dataset Progress",
            position=1,
            leave=False
        )
        dataset_metrics = tqdm(
            total=0, position=2, bar_format='{desc}', desc=""
        )

        # Collect data
        for i, batch in enumerate(collector):
            batch = batch.to(self.training_device)

            # Learn from this batch
            for _ in range(self.config.num_epochs):
                # Compute advantages
                self.advantage_module(batch)

                # Split batch into minibatches
                for minibatch in batch.split(self.config.sub_batch_size):

                    loss_vals = self.loss_module(minibatch)

                    loss = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )

                    loss.backward()

                    # Clip to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.loss_module.parameters(), self.config.max_grad_norm
                    )

                    self.optim.step()
                    self.optim.zero_grad()

                self.scheduler.step()

            # Log reward
            logs["reward"].append(
                batch["next", "reward"].mean().item())
            cum_reward_str = (
                f"avg reward={logs['reward'][-1]: 4.2f}"
            )

            # Log avg reward change
            if len(logs["reward"]) == 1:
                reward_change = torch.tensor(0, dtype=torch.float32)
            else:
                reward_change = torch.diff(torch.tensor(logs["reward"]))
            reward_change_str = (
                f"avg reward change={reward_change.mean().item(): 4.5f}"
            )

            dataset_progress.update(batch.numel())

            # Every 10 batches, evaluate the policy
            if i % 10 == 0:
                with set_interaction_type(InteractionType.DETERMINISTIC), torch.no_grad():
                    # execute a rollout with the trained policy
                    env = PresentEnv.make_transformed_env(
                        td, device=self.training_device)
                    eval_rollout = env.rollout(1000, self.policy_module)
                    logs["eval reward"].append(
                        eval_rollout["next", "reward"].mean().item())
                    logs["eval reward (sum)"].append(
                        eval_rollout["next", "reward"].sum().item()
                    )

                    del eval_rollout

            if i > 0 and i % 20 == 0:
                self.save(logs, True)

            dataset_metrics.set_description(
                ", ".join([cum_reward_str, reward_change_str])
            )

        self.save(logs, False)

    def train_all(self):
        """ Train the model on all data """
        logs = defaultdict(list)

        overall_progress = tqdm(
            total=len(self.input_td), desc="Total Progress", position=0, leave=True)

        # for every set of data in the input
        for td in self.input_td:
            self.train(td, logs)
            overall_progress.update(1)

        # final save
        self.save(logs, False)

    def save(self, logs, is_checkpoint: bool):
        """
        Save the model, optimizer, and scheduler states.

        Args:
            logs: logs containing subsequent model data
            isCheckpoint: indicates that the current save is a checkpoint and not a completed model.
        """

        data = ModelData(
            logs['reward'][-1],
            self.policy_module.state_dict(),
            self.value_module.state_dict(),
            self.loss_module.state_dict(),
            self.optim.state_dict(),
            self.scheduler.state_dict()
        )

        manager = ModelSaveManager()
        if is_checkpoint:
            manager.save_checkpoint(data)
        else:
            manager.save(data)

    def run(self):
        """ Run the model """
