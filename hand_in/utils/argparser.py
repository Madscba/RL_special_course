import argparse


class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Reinforcement argparser")

        #Reproducability seed (torch and numpy seeds are set)
        self.parser.add_argument("--seed", type=int, default=3, metavar="N", help="random seed (default: 3)")

        #Environment config         # "CartPole-v1", "LunarLander-v2", "LunarLanderContinuous-v2"
        self.parser.add_argument("--env_name",type=str,default="LunarLander-v2")

        #Logger config
        self.parser.add_argument("--log_format", type=str, default='default',help = "'file' saves .csv, 'console' prints to terminal. 'default' does both")
        self.parser.add_argument("--frame_interval", type=int, default='1000',help = "'file' saves .csv, 'console' prints to terminal. 'default' does both")
        self.parser.add_argument("--visualize",type=bool,default=True, help="Visualize training plots")

        #Training config
        self.parser.add_argument("--n_steps",type=int,default=200000,metavar="N",help="number of steps (default: 100000)")
        self.parser.add_argument("--n_env",type=int,default=1,metavar="N",help="number of environments (default: 1)")

        # learning algorithm configuration
        self.parser.add_argument("--algorithm", type=str, default="REINFORCE") #DDQN, REINFORCE, AC, SAC
        self.parser.add_argument("--gamma",type=float,default=0.995,metavar="G",help="discount factor for reward (default: 0.999)")

        # model configurations (both actor and critic)
        self.parser.add_argument("--hidden_size",type=int,default=32,metavar="N",help="number of episodes (default: 32)")
        self.parser.add_argument("--lr",type=float,default=0.001,metavar="N",help="learning rate (default: 0.001)")

        #algorithm specifics:
        #DQN & DDQN:
        self.parser.add_argument('--eps', type=float, default= 1,)
        self.parser.add_argument('--eps_decay', type=float, default= 0.001,)
        self.parser.add_argument('--min_eps', type=float, default= 0.05,)
        self.parser.add_argument('--batch_size', type=int, default= 64,)
        self.parser.add_argument('--use_replay', type=bool, default= True)

        #AC
        self.parser.add_argument("--grad_clipping",type=float,default=1,metavar="G",help="maximum grad value. If above 0 they are clipped")
        self.parser.add_argument("--entropy",type=bool,default=True,help="add entropy regularization to objective to encourage learning")

        self.args = self.parser.parse_args()
