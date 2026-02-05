""" Code for the neural network itself. """

from collections import defaultdict

import torch
from tqdm import tqdm
from torch import distributions as d
from torchrl.data import ReplayBuffer, LazyMemmapStorage, SamplerWithoutReplacement
from torchrl.modules import ProbabilisticActor
from torchrl.collectors import SyncDataCollector
from torchrl.objectives.value import GAE
from torchrl.objectives import ClipPPOLoss
from tensordict.nn import (
    TensorDictModule,
    CompositeDistribution,
    set_interaction_type,
    InteractionType
)

from model.modules.actor import PresentActor
from model.modules.critic import PresentCritic
from model.env import PresentEnv
from model.config.ppo_config import PPOConfig
from model.saved_models.save_manager import ModelData, ModelSaveManager
from data.data_reader import get_data


class PPO:
    """ Implementation of the PPO actor-critic network """

    def __init__(self, input_name="input.txt", config=None, training_device=None):
        self.training_device = training_device or (
            torch.device('cuda') if torch.cuda.is_available(
            ) else torch.device('cpu')
        )
        self.config = config or PPOConfig()
        self.input_data = get_data(torch.device("cpu"), input_name)

        # Set up Actor and Critic
        self.actor_net = PresentActor(self.training_device)
        td_policy_module = TensorDictModule(
            self.actor_net,
            in_keys=["observation"],
            out_keys=["params"]
        )

        self.policy_module = ProbabilisticActor(
            module=td_policy_module,
            spec=PresentEnv.get_action_spec(),
            in_keys=["params"],
            distribution_class=CompositeDistribution,
            distribution_kwargs={
                "distribution_map": {
                    "present_idx": d.Categorical,
                    "rot": d.Categorical,
                    "flip": d.Bernoulli,
                    "x": d.Normal,
                    "y": d.Normal
                },
                "name_map": {
                    "present_idx": ("action", "present_idx"),
                    "rot": ("action", "rot"),
                    "flip": ("action", "flip"),
                    "x": ("action", "x"),
                    "y": ("action", "y")
                }
            },
            return_log_prob=True
        )

        self.value_net = PresentCritic(self.training_device)
        self.value_module = TensorDictModule(
            module=self.value_net,
            in_keys=["observation"],
            out_keys=["state_value"]
        )

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
            self.actor_net.load_state_dict(best.actor_state)
            self.value_net.load_state_dict(best.critic_state)
            self.policy_module.load_state_dict(best.policy_state)
            self.value_module.load_state_dict(best.value_state)
            self.loss_module.load_state_dict(best.loss_state)
            self.optim.load_state_dict(best.optim_state)
            self.scheduler.load_state_dict(best.scheduler_state)

    def _process_sub_batch(self, replay_buffer):
        # Calculate loss on subdata sample
        subdata = replay_buffer.sample(
            self.config.sub_batch_size)

        loss_vals = self.loss_module(subdata.to(self.training_device))

        loss_value = (
            loss_vals["loss_objective"]
            + loss_vals["loss_critic"]
            + loss_vals["loss_entropy"]
        )

        # Optimise
        loss_value.backward()

        # Clear loss tensors immediately after backward
        del loss_value, loss_vals

        torch.nn.utils.clip_grad_norm_(
            self.loss_module.parameters(), self.config.max_grad_norm
        )
        self.optim.step()
        self.optim.zero_grad()

    def train(self):
        """ Train the model """

        logs = defaultdict(list)

        # for every set of data in the generator
        for td in tqdm(self.input_data, desc="Total progress", position=0):
            collector = SyncDataCollector(
                PresentEnv.make_parallel_env,  # type: ignore
                self.policy_module,
                frames_per_batch=self.config.frames_per_batch,
                total_frames=self.config.total_frames,
                create_env_kwargs={
                    "start_state": td,
                    "num_workers": 1,
                    "device": torch.device("cpu")
                },
                # prevents linux CUDA permission err
                device=torch.device("cpu"),
                storing_device=torch.device("cpu"),
                policy_device=self.training_device
            )

            # Note: memmap is cpu - only
            replay_buffer = ReplayBuffer(
                storage=LazyMemmapStorage(
                    max_size=self.config.frames_per_batch,
                    device=torch.device("cpu")
                ),
                sampler=SamplerWithoutReplacement(),
            )

            pbar = tqdm(total=self.config.total_frames,
                        desc="Current batch progress", position=1)

            for i, batch in enumerate(collector):
                dev_batch = batch.to(self.training_device)
                for _ in range(self.config.num_epochs):
                    self.advantage_module(dev_batch)
                    replay_buffer.extend(batch)
                    for _ in range(self.config.frames_per_batch // self.config.sub_batch_size):
                        self._process_sub_batch(replay_buffer)

                # Processed all epochs, log data
                logs["reward"].append(
                    dev_batch["next", "reward"].mean().item())
                pbar.update(dev_batch.numel())
                cum_reward_str = (
                    f"average reward={logs['reward'][-1]: 4.2f} (init={logs['reward'][0]: 4.2f})"
                )
                logs["lr"].append(self.optim.param_groups[0]["lr"])
                lr_str = f"lr policy: {logs['lr'][-1]: 4.5f}"

                # Every 10 batches, evaluate the policy
                if i % 10 == 0:
                    with set_interaction_type(InteractionType.DETERMINISTIC), torch.no_grad():
                        # execute a rollout with the trained policy
                        env = PresentEnv(td, device=self.training_device)
                        eval_rollout = env.rollout(1000, self.policy_module)
                        logs["eval reward"].append(
                            eval_rollout["next", "reward"].mean().item())
                        logs["eval reward (sum)"].append(
                            eval_rollout["next", "reward"].sum().item()
                        )

                        del eval_rollout

                pbar.set_description(
                    "Current batch progress:  " + ", ".join([cum_reward_str, lr_str]))

                self.scheduler.step()

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
            self.actor_net.state_dict(),
            self.value_net.state_dict(),
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
