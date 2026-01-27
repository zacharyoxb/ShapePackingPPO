""" Code for the neural network itself. """

from collections import defaultdict

import torch
from tqdm import tqdm
from torch import distributions as d
from torchrl.modules import ProbabilisticActor, ValueOperator
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
from data.reader import get_data_generator


class PPO:
    """ Implementation of the PPO actor-critic network """

    def __init__(self, input_name="input.txt", config=None, device=None):
        self.device = device or (
            torch.device('cuda') if torch.cuda.is_available(
            ) else torch.device('cpu')
        )
        self.config = config or PPOConfig()
        self.data_generator = get_data_generator(self.device, input_name)

        # Set up Actor and Critic
        self.actor_net = PresentActor(self.device)

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

        self.value_net = PresentCritic(self.device)
        td_value_module = TensorDictModule(
            self.value_net,
            in_keys=["observation"],
            out_keys=["value"]
        )

        self.value_module = ValueOperator(
            module=td_value_module,
            in_keys=["value"]
        )

        # Loss function config
        self.advantage_module = GAE(
            gamma=self.config.gamma,
            lmbda=self.config.lmbda,
            value_network=self.value_module,
            average_gae=True,
            device=torch.device(self.device)
        )

        self.loss_module = ClipPPOLoss(
            actor_network=self.policy_module,
            critic_network=self.value_module,
            clip_epsilon=self.config.clip_epsilon,
            entropy_bonus=bool(self.config.entropy_eps),
            entropy_coeff=self.config.entropy_eps,
            critic_coeff=1.0,
            loss_critic_type="smooth_l1"
        )

        self.optim = torch.optim.Adam(
            self.loss_module.parameters(),
            lr=self.config.lr
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, self.config.total_frames // self.config.frames_per_batch, 0.0
        )

    def train(self):
        """ Train the model """
        logs = defaultdict(list)
        pbar = tqdm(total=self.config.total_frames)
        eval_str = ""

        # for every set of data in the generator
        for td in self.data_generator:
            # convert td to device we are using
            # init the collector using td env params
            def make_env(start_state=td) -> PresentEnv:
                env = PresentEnv(start_state)
                env.to(self.device)
                return env

            collector = SyncDataCollector(
                make_env,  # type: ignore
                self.policy_module,
                create_env_kwargs={"start_state": td},
                frames_per_batch=self.config.frames_per_batch,
                total_frames=self.config.total_frames,
                split_trajs=True,
                device=self.device,
            )

            for i, batch in enumerate(collector):
                # calculate advantage of current batch
                self.advantage_module(batch)
                data_view = batch.reshape(-1)

                # Reuse same data batch several times
                for _ in range(self.config.num_epochs):
                    # Carry out updates on sub batches
                    for _ in range(self.config.frames_per_batch // self.config.sub_batch_size):
                        # Calculate loss
                        loss_vals = self.loss_module(data_view)
                        loss_value = (
                            loss_vals["loss_objective"]
                            + loss_vals["loss_critic"]
                            + loss_vals["loss_entropy"]
                        )

                        # Optimise
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.loss_module.parameters(), self.config.max_grad_norm)
                        self.optim.step()
                        self.optim.zero_grad()

                # Processed all epochs, log data
                logs["reward"].append(batch["next", "reward"].mean().item())
                pbar.update(batch.numel())
                cum_reward_str = (
                    f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
                )
                logs["step_count"].append(batch["step_count"].max().item())
                stepcount_str = f"step count (max): {logs['step_count'][-1]}"
                logs["lr"].append(self.optim.param_groups[0]["lr"])
                lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"

                # Every 10 batches, evaluate the policy
                if i % 10 == 0:
                    with set_interaction_type(InteractionType.DETERMINISTIC), torch.no_grad():
                        # execute a rollout with the trained policy
                        env = make_env()
                        eval_rollout = env.rollout(1000, self.policy_module)
                        logs["eval reward"].append(
                            eval_rollout["next", "reward"].mean().item())
                        logs["eval reward (sum)"].append(
                            eval_rollout["next", "reward"].sum().item()
                        )
                        logs["eval step_count"].append(
                            eval_rollout["step_count"].max().item())
                        eval_str = (
                            f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                            f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                            f"eval step-count: {logs['eval step_count'][-1]}"
                        )
                        del eval_rollout

                pbar.set_description(
                    ", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

                self.scheduler.step()

    def save(self):
        """ Save the model """

    def run(self):
        """ Run the model """
