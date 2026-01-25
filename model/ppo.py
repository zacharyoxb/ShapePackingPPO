""" Code for the neural network itself. """

from collections import defaultdict

import torch
from tqdm import tqdm
from torch import distributions as d
from torchrl.data import ReplayBuffer
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.collectors import SyncDataCollector
from torchrl.objectives.value import GAE
from torchrl.objectives import ClipPPOLoss
from tensordict.nn import TensorDictModule, CompositeDistribution
from tensordict.nn.probabilistic import InteractionType, set_interaction_type

from model.modules.actor import PresentActor
from model.modules.critic import PresentCritic
from model.env import PresentEnv
from model.config.ppo_config import PPOConfig


class PPO:
    """ Implementation of the PPO actor-critic network """

    def __init__(self, replay_buffer: ReplayBuffer, config=None, device=None):
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config or PPOConfig()

        self.env = PresentEnv().to(self.device)
        # Set up Actor and Critic
        self.actor_net = PresentActor().to(device)

        td_policy_module = TensorDictModule(
            self.actor_net,
            in_keys=[
                "grid", "presents", "present_count"
            ],
            out_keys=[
                ("params", "present_idx_logits"),
                ("params", "rot_logits"),
                ("params", "flip_logits"),
                ("params", "x"),
                ("params", "y"),
            ]
        )
        self.policy_module = ProbabilisticActor(
            module=td_policy_module,
            spec=self.env.action_spec,
            in_keys=["params"],
            distribution_class=CompositeDistribution,
            distribution_kwargs={
                "distribution_map": {
                    "present_idx": d.Categorical,
                    "rot": d.Categorical,
                    "flip": d.Bernoulli,
                    "x": lambda loc, scale: d.TransformedDistribution(
                        d.Normal(loc, scale),
                        d.TanhTransform()
                    ),
                    "y": lambda loc, scale: d.TransformedDistribution(
                        d.Normal(loc, scale),
                        d.TanhTransform()
                    ),
                }
            },
            return_log_prob=True
        )

        self.value_net = PresentCritic().to(device)
        td_value_module = TensorDictModule(
            self.value_net,
            in_keys=[
                "grid", "presents", "present_count"
            ],
            out_keys=[
                "value"
            ]
        )
        self.value_module = ValueOperator(
            module=td_value_module,
            in_keys=[
                "value"
            ]
        )

        # Collector
        self.collector = SyncDataCollector(
            self.env,
            self.actor_net,
            frames_per_batch=self.config.frames_per_batch,
            total_frames=self.config.total_frames,
            split_trajs=False,
            device=self.device
        )

        # Data for model
        self.replay_buffer = replay_buffer

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

        # We iterate over the collector until it reaches the total number of frames it was
        # designed to collect:
        for i, tensordict_data in enumerate(self.collector):
            # we now have a batch of data to work with. Let's learn something from it.
            for _ in range(self.config.num_epochs):
                # We'll need an "advantage" signal to make PPO work.
                # We re-compute it at each epoch as its value depends on the value
                # network which is updated in the inner loop.
                self.advantage_module(tensordict_data)
                data_view = tensordict_data.reshape(-1)
                self.replay_buffer.extend(data_view.cpu())
                for _ in range(self.config.frames_per_batch // self.config.sub_batch_size):
                    subdata = self.replay_buffer.sample(
                        self.config.sub_batch_size)
                    loss_vals = self.loss_module(subdata.to(self.device))
                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )

                    # Optimization: backward, grad clipping and optimization step
                    loss_value.backward()
                    # this is not strictly mandatory but it's good practice to keep
                    # your gradient norm bounded
                    torch.nn.utils.clip_grad_norm_(
                        self.loss_module.parameters(), self.config.max_grad_norm)
                    self.optim.step()
                    self.optim.zero_grad()

            logs["reward"].append(
                tensordict_data["next", "reward"].mean().item())
            pbar.update(tensordict_data.numel())
            cum_reward_str = (
                f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
            )
            logs["step_count"].append(
                tensordict_data["step_count"].max().item())
            stepcount_str = f"step count (max): {logs['step_count'][-1]}"
            logs["lr"].append(self.optim.param_groups[0]["lr"])
            lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
            if i % 10 == 0:
                # We evaluate the policy once every 10 batches of data.
                # Evaluation is rather simple: execute the policy without exploration
                # (take the expected value of the action distribution) for a given
                # number of steps (1000, which is our ``env`` horizon).
                # The ``rollout`` method of the ``env`` can take a policy as argument:
                # it will then execute this policy at each step.
                with set_interaction_type(InteractionType.DETERMINISTIC), torch.no_grad():
                    # execute a rollout with the trained policy
                    eval_rollout = self.env.rollout(1000, self.policy_module)
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

            # We're also using a learning rate scheduler. Like the gradient clipping,
            # this is a nice-to-have but nothing necessary for PPO to work.
            self.scheduler.step()

    def save(self):
        """ Save the model """

    def run(self):
        """ Run the model """
