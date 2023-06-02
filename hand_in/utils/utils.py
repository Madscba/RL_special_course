import numpy as np
import torch
import gym
from gym.wrappers.record_video import RecordVideo
import matplotlib.pyplot as plt

from hand_in.environment.gym_environment import get_environment_info
from hand_in.agents.dqn_agent import DQNAgent
from hand_in.agents.reinforce_agent import ReinforceAgent
from hand_in.agents.actor_critic_agent import ACAgent
from hand_in.agents.sac_agent_v0 import SACAgent_v0
from hand_in.agents.sac_agent_v2 import SACAgent_v2


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
    elif argparser.args.algorithm == "DDQN":
        agent = DQNAgent(
            argparser=argparser,
            use_DDQN=True,
            state_dim=state_dim,
            action_dim=action_dim,
            n_actions=n_actions,
        )
    elif argparser.args.algorithm == "REINFORCE":
        agent = ReinforceAgent(
            argparser=argparser,
            action_dim=action_dim,
            state_dim=state_dim,
            n_actions=n_actions,
            action_type=action_type,
        )
    elif argparser.args.algorithm == "AC":
        agent = ACAgent(
            argparser=argparser,
            action_dim=action_dim,
            state_dim=state_dim,
            n_actions=n_actions,
            action_type=action_type,
        )
    elif argparser.args.algorithm == "SAC_v0":
        agent = SACAgent_v0(argparser=argparser,
                            action_dim=action_dim,
                            state_dim=state_dim,
                            n_actions=n_actions,
                            action_type=action_type,
        )
    elif argparser.args.algorithm == "SAC_v2":
        agent = SACAgent_v2(argparser=argparser,
                            action_dim=action_dim,
                            state_dim=state_dim,
                            n_actions=n_actions,
                            action_type=action_type, )
    else:
        raise Exception(f"{argparser.args.algorithm}-agent is currently not supported")
    return agent


def evaluate_agent(agent, env_name, num_episodes=1, render=True):
    env = gym.make(env_name)
    env = RecordVideo(env, './results/temporary/video', episode_trigger=lambda episode_number: True)

    episode_rewards = []

    for episode in range(num_episodes):
        episode_reward = 0
        observation = env.reset()

        done = False
        while not done:
            if render:
                env.render()

            action,_ = agent.follow_policy(observation)
            next_observation, reward, done, _ = env.step(action.squeeze())

            episode_reward += reward
            observation = next_observation

        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward}")

    env.close()

    # Plot rewards
    # plt.plot(episode_rewards)
    # plt.xlabel('Episode')
    # plt.ylabel('Reward')
    # plt.title('Agent Evaluation')
    # plt.show()

def set_seed(seed: int = 3):
    np.random.seed(seed)
    torch.manual_seed(0)
