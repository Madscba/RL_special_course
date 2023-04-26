import gymnasium as gym
import numpy as np

from agents import PolicyGradientAgent
from models import ActorNetwork, CriticNetwork
from utils import Parser


def get_envs(
    env_id: str = "LunarLander-v2",
    num_envs: int = 10,
    asynchronous: bool = False,
    continuous: bool = True,
    gravity: float = -10.0,
    enable_wind: bool = False,
    wind_power: float = 15.0,
    turbulence_power: float = 1.5,
):
    envs = gym.vector.make(
        id=env_id,
        num_envs=num_envs,
        asynchronous=asynchronous,
        continuous=continuous,
        gravity=gravity,
        enable_wind=enable_wind,
        wind_power=wind_power,
        turbulence_power=turbulence_power,
    )
    return envs


if __name__ == "__main__":
    parser = Parser()
    #set seed
    np.random.seed(parser.args.seed)

    # Init environments
    envs = get_envs(num_envs=parser.args.n_environments)
    _ = envs.reset(seed=42)
    #
    n_state = envs.single_observation_space.shape[0]
    n_action = envs.single_action_space.shape[0]
    lr_algo = parser.args.learning_algorithm

    actor_network = ActorNetwork(
        state_dim=n_state,
        action_dim=n_action,
        envs=envs,
        hidden_dim=parser.args.hidden_size,
    )
    if lr_algo != "REINFORCE":
        critic_network = CriticNetwork(
            state_dim=n_state,
            action_dim=n_action,
            envs=envs,
            hidden_dim=parser.args.hidden_size,
        )
        agent = PolicyGradientAgent(envs=envs, actor_network=actor_network, critic_network = critic_network, parser=parser)
        n_episodes = parser.args.n_episodes * 1000 // parser.args.n_environments
    else:
        agent = PolicyGradientAgent(envs=envs, actor_network=actor_network, parser=parser)
        n_episodes = parser.args.n_episodes // parser.args.n_environments
    agent.learn_policy(n_frames=n_episodes, learning_algorithm=lr_algo)

    envs.close()
