""" Code for the neural network itself. """

from collections import defaultdict

import torch
from tqdm import tqdm
from torch import distributions as d
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.collectors import SyncDataCollector
from torchrl.objectives.value import GAE
from torchrl.objectives import ClipPPOLoss
from tensordict.nn import TensorDictModule, CompositeDistribution

from model.modules.actor import PresentActor
from model.modules.critic import PresentCritic
from model.env import PresentEnv
from model.config.ppo_config import PPOConfig
from data.reader import get_data_generator


class PPO:
    """ Implementation of the PPO actor-critic network """

    def __init__(self, config=None, device=None):
        self.device = device or (
            torch.device('cuda') if torch.cuda.is_available(
            ) else torch.device('cpu')
        )
        self.config = config or PPOConfig()
        self.data_generator = get_data_generator(self.device, "testinput.txt")

        # Set up Actor and Critic
        self.actor_net = PresentActor(self.device)

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
            spec=PresentEnv.get_action_spec(),
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

        self.value_net = PresentCritic(self.device)
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
                self.actor_net,
                create_env_kwargs={"start_state": td},
                frames_per_batch=self.config.frames_per_batch,
                total_frames=self.config.total_frames,
                split_trajs=True,
                device=self.device,
            )

            for i, tensordict_data in enumerate(collector):
                for _ in range(self.config.num_epochs):
                    test = 1 + 1

    def save(self):
        """ Save the model """

    def run(self):
        """ Run the model """
