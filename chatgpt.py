import gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


def select_action(state, policy):
    state = torch.from_numpy(state).float().unsqueeze(0)
    action_probs = policy(state)
    dist = torch.distributions.Normal(action_probs, 1)
    action = dist.sample()
    return action.numpy().squeeze(), dist.log_prob(action)


def update_policy(policy, rewards, log_probs):
    policy_loss = []
    for reward, log_prob in zip(rewards, log_probs):
        policy_loss.append(-log_prob * reward)
    policy_loss = torch.cat(policy_loss).sum()
    policy.optimizer.zero_grad()
    policy_loss.backward()
    policy.optimizer.step()


def train(policy, env, num_episodes):
    episode_rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        rewards = []
        log_probs = []
        while not done:
            action, log_prob = select_action(state, policy)
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            state = next_state
        update_policy(policy, rewards, log_probs)
        episode_reward = sum(rewards)
        episode_rewards.append(episode_reward)
        print('Episode {}: reward = {}'.format(episode, episode_reward))

    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

env = gym.make('LunarLanderContinuous-v2')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
policy = Policy(state_dim, action_dim)
train(policy, env, num_episodes=1000)