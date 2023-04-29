import argparse


class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Reinforcement argparser")

        # environment configuration
        self.parser.add_argument(
            "--env_name", type=str, default="LunarLanderContinuous-v2"
        )
        self.parser.add_argument(
            "--visualize", type=bool, default=True)
        self.parser.add_argument(
            "--seed", type=int, default=3, metavar="N", help="random seed (default: 3)"
        )

        self.parser.add_argument(
            "--n_episodes",
            type=int,
            default=5000,
            metavar="N",
            help="number of episodes (default: 400)",
        )
        self.parser.add_argument(
            "--n_environments",
            type=int,
            default=10,
            metavar="N",
            help="number of episodes (default: 10)",
        )

        # learning algorithm configuration
        self.parser.add_argument("--learning_algorithm", type=str, default="AC")  # REINFORCE
        self.parser.add_argument("--entropy", type=bool, default=False,
                                 help="add entropy regularization to objective to encourage learning")
        self.parser.add_argument(
            "--gamma",
            type=float,
            default=0.99,
            metavar="G",
            help="discount factor for reward (default: 0.99)",
        )


        # network configurations (both actor and critic)
        self.parser.add_argument(
            "--hidden_size",
            type=int,
            default=64,
            metavar="N",
            help="number of episodes (default: 64)",
        )
        self.parser.add_argument(
            "--lr",
            type=float,
            default=0.001,
            metavar="N",
            help="learning rate (default: 0.001)",
        )
        self.parser.add_argument(
            "--grad_clipping",
            type=float,
            default=0,
            metavar="G",
            help="maximum grad value. If above they are clipped",
        )

        self.args = self.parser.parse_args()
