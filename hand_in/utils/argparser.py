import argparse


class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Reinforcement argparser")

        # Reproducability seed (torch and numpy seeds are set)
        self.parser.add_argument(
            "--seed", type=int, default=3, metavar="N", help="random seed (default: 3)"
        )

        # Environment config         # "CartPole-v1", "LunarLander-v2", "LunarLanderContinuous-v2"
        self.parser.add_argument(
            "--env_name", type=str, default="LunarLanderContinuous-v2"
        )

        # Logger config
        self.parser.add_argument(
            "--log_format",
            type=str,
            default="default",
            help="'file' saves .csv, 'console' prints to terminal. 'default' does both",
        )
        self.parser.add_argument(
            "--frame_interval",
            type=int,
            default=10000,
            help="'file' saves .csv, 'console' prints to terminal. 'default' does both",
        )
        self.parser.add_argument(
            "--visualize", type=int, default=0, help="Visualize training plots"
        )

        # Training config
        self.parser.add_argument(
            "--n_steps",
            type=int,
            default=200000,
            metavar="N",
            help="number of steps (default: 100000)",
        )
        self.parser.add_argument(
            "--n_env",
            type=int,
            default=1,
            metavar="N",
            help="number of environments (default: 1)",
        )

        # learning algorithm configuration

        self.parser.add_argument(
            "--algorithm", type=str, default="REINFORCE"
        )  # DDQN, REINFORCE, AC, SAC
        self.parser.add_argument(
            "--gamma",
            type=float,
            default=0.99,
            metavar="G",
            help="discount factor for reward (default: 0.999)",
        )

        # model configurations (both actor and critic)
        self.parser.add_argument(
            "--hidden_size",
            type=int,
            default=512,
            metavar="N",
            help="number of hidden units (default: 32)",
        )
        self.parser.add_argument(
            "--lr",
            type=float,
            default=0.0001,  # 0.00001
            metavar="N",
            help="learning rate (default: 0.001)",
        )

        # algorithm specifics:
        # DQN & DDQN:
        self.parser.add_argument(
            "--eps",
            type=float,
            default=1,
        )
        self.parser.add_argument(
            "--eps_decay",
            type=float,
            default=0.001,  # 0.001
        )
        self.parser.add_argument(
            "--min_eps",
            type=float,
            default=0.05,  # 0.05
        )
        self.parser.add_argument(
            "--batch_size",
            type=int,
            default=128,
        )
        self.parser.add_argument("--use_replay", type=int, default=1)

        # AC
        self.parser.add_argument(
            "--grad_clipping",
            type=float,
            default=0,
            metavar="G",
            help="maximum grad value. If above 0 they are clipped",
        )
        self.parser.add_argument(
            "--entropy",
            type=int,
            default=0,
            help="add entropy regularization to objective to encourage learning",
        )

        # SAC
        self.parser.add_argument(
            "--alpha", type=float, default=1, help="entropy weight"
        )
        self.parser.add_argument(
            "--tau",
            type=float,
            default=0.01,  # 0.005
            help="exponential moving average constant",
        )
        self.parser.add_argument(
            "--reward_scale",
            type=float,
            default=2.0,
            help="scale our rewards, depends on problem",
        )

        self.args = self.parser.parse_args()
