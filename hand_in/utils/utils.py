import numpy as np
import torch

from hand_in.environment.gym_environment import get_environment_info
from hand_in.agents.dqn_agent import DQNAgent
from hand_in.agents.reinforce_agent import ReinforceAgent
from hand_in.agents.actor_critic_agent import ACAgent


def get_agent(argparser, environments):
    action_dim, state_dim, n_actions, action_type = get_environment_info(
        envs=environments
    )

    if argparser.args.algorithm == "DQN":
        agent = DQNAgent(
            argparser=argparser,
            use_DDQN=False,
            state_dim=state_dim,
            action_dim=action_dim,
            n_actions=n_actions,
        )
        return agent
    elif argparser.args.algorithm == "DDQN":
        agent = DQNAgent(
            argparser=argparser,
            use_DDQN=True,
            state_dim=state_dim,
            action_dim=action_dim,
            n_actions=n_actions,
        )
        return agent
    elif argparser.args.algorithm == "REINFORCE":
        agent = ReinforceAgent(
            argparser=argparser,
            action_dim=action_dim,
            state_dim=state_dim,
            n_actions=n_actions,
            action_type=action_type,
        )
        return agent
    elif argparser.args.algorithm == "AC":
        agent = ACAgent(
            argparser=argparser,
            action_dim=action_dim,
            state_dim=state_dim,
            n_actions=n_actions,
            action_type=action_type,
        )
        return agent
    elif argparser.args.algorithm == "SAC":
        agent = SACAgent()
        return agent
    else:
        raise Exception(f"{argparser.args.algorithm}-agent is currently not supported")


def set_seed(seed: int = 3):
    np.random.seed(seed)
    torch.manual_seed(0)
