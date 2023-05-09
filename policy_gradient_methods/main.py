import gymnasium as gym
import numpy as np
import os
from agents import PolicyGradientAgent
from models import ActorNetwork_cont,ActorNetwork_disc, CriticNetwork
from utils import Parser
from gymnasium.wrappers import RecordVideo




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
        # asynchronous=asynchronous,
        # continuous=continuous,
        # gravity=gravity,
        # enable_wind=enable_wind,
        # wind_power=wind_power,
        # turbulence_power=turbulence_power,
    )
    return envs


if __name__ == "__main__":
    parser = Parser()
    #set seed
    np.random.seed(parser.args.seed)

    # Init environments
    envs = get_envs(env_id=parser.args.env_name,num_envs=parser.args.n_environments)
    _ = envs.reset(seed=42)

    #Define the learning algorithm
    print(f"args: {parser.args}")
    lr_algo = parser.args.learning_algorithm
    n_state = envs.single_observation_space.shape[0]

    if "Discrete" in str(envs.single_action_space):
        n_action = envs.single_action_space.n
        actor_network = ActorNetwork_disc(
            state_dim=n_state,
            action_dim=n_action,
            lr=parser.args.lr,
            envs=envs,
            hidden_dim=parser.args.hidden_size,
        )
    else:
        n_action = envs.single_action_space.shape[0]
        actor_network = ActorNetwork_cont(
            state_dim=n_state,
            lr = parser.args.lr,
            action_dim=n_action,
            envs=envs,
            hidden_dim=parser.args.hidden_size,
        )
    if lr_algo != "REINFORCE":
        critic_network = CriticNetwork(
            state_dim=n_state,
            lr=parser.args.lr,
            action_dim=n_action,
            envs=envs,
            hidden_dim=parser.args.hidden_size,
        )
        parser.args.n_episodes = parser.args.n_episodes * 100 // parser.args.n_environments
        agent = PolicyGradientAgent(envs=envs, actor_network=actor_network, critic_network = critic_network, parser=parser)
    else:
        parser.args.n_episodes = parser.args.n_episodes // parser.args.n_environments
        agent = PolicyGradientAgent(envs=envs, actor_network=actor_network, parser=parser)
    agent.learn_policy(n_frames=parser.args.n_episodes, learning_algorithm=lr_algo)

    #envs.close()

    #env = gym.make(parser.args.env_name, render_mode="rgb_array")
    # wrapped_env = RecordVideo(
    #     envs,
    #     video_folder=os.getcwd(),
    #     episode_trigger=lambda x: x % 50 == 0,
    #     name_prefix="train_videos_cart",
    # )
    # agent = PolicyGradientAgent(envs=envs, actor_network=actor_network, parser=parser)
    # agent.evaluate_policy(n_frames=parser.args.n_episodes//10, learning_algorithm=lr_algo)
    #
