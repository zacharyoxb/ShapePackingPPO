""" Code for the neural network itself. """

from collections import defaultdict

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
        best = manager.load_best()

        if best:
            self.policy_module.load_state_dict(best.policy_state)
            self.value_module.load_state_dict(best.value_state)
            self.loss_module.load_state_dict(best.loss_state, strict=False)
            self.policy_module.eval()
            self.value_module.eval()
            self.loss_module.eval()

            self.optim.load_state_dict(best.optim_state)
            self.scheduler.load_state_dict(best.scheduler_state)

    def train(self):
        """ Train the model """
        logs = defaultdict(list)

        # for every set of data in the input
        for td in tqdm(self.input_td, desc="Total progress", position=0, leave=True):
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

            pbar = tqdm(
                total=self.config.total_frames,
                desc="Current batch progress",
                position=1,
                leave=False
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

                # Processed all epochs, log data
                logs["reward"].append(
                    batch["next", "reward"].mean().item())
                pbar.update(batch.numel())
                cum_reward_str = (
                    f"average reward={logs['reward'][-1]: 4.2f}"
                )
                logs["lr"].append(self.optim.param_groups[0]["lr"])
                lr_str = f"lr policy: {logs['lr'][-1]: 4.5f}"

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

                pbar.set_description(
                    "Current batch progress:  " + ", ".join([cum_reward_str, lr_str]))

            # checkpoint model now current data has been trained on
            self.save(logs, True)

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
