import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.data import DataLoader
from gym.vector import VectorEnv, AsyncVectorEnv
from stable_baselines3.common.vec_env import VecNormalize
import matplotlib.pyplot as plt
from tqdm import tqdm


# Define the policy network
class Policy(nn.Module):
    def __init__(self, obs_shape, action_shape):
        super(Policy, self).__init__()
        self.input_dim = obs_shape[0]
        self.output_dim = action_shape[0]
        self.fc1 = nn.Linear(obs_shape[0], 64)
        self.fc2 = nn.Linear(64, 64)
        self.mean_fc = nn.Linear(64, action_shape[0])
        self.log_std_fc = nn.Linear(64, action_shape[0])

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean_fc(x)
        log_std = self.log_std_fc(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        normal = Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)
        log_prob = normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-8)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob


def train(
    env,
    policy_network,
    optimizer,
    num_epochs=100,
    batch_size=64,
    gamma=0.99,
    max_episode_length=500,
):
    # Convert the environment to a vectorized environment
    vec_env = VecNormalize(
        AsyncVectorEnv(
            [lambda: gym.make("LunarLanderContinuous-v2") for i in range(env)]
        ),
        norm_obs=True,
        norm_reward=True,
    )

    # Initialize the running reward
    running_reward = None

    # Initialize the episode buffer
    episode_buffer = []

    # Iterate over epochs
    for epoch in tqdm(range(num_epochs)):
        # Reset the environment and get the initial observation
        obs = vec_env.reset()

        # Iterate over time steps
        for t in range(max_episode_length):
            # Sample an action from the policy network
            action, log_prob = policy_network.sample(
                torch.from_numpy(obs).reshape(-1, n_env, policy_network.input_dim)
            )

            # Step the environment forward and record the results
            next_obs, reward, done, info = vec_env.step(action.detach().cpu().numpy())

            # Add the transition to the episode buffer
            episode_buffer.append(
                (
                    obs,
                    action.detach().numpy(),
                    log_prob.detach().numpy(),
                    reward,
                    next_obs,
                    done,
                )
            )

            # Update the observation
            obs = next_obs

            # If the episode is over, update the policy network
            if done.any():
                # Extract the data from the episode buffer
                (
                    obs_batch,
                    action_batch,
                    log_prob_batch,
                    reward_batch,
                    next_obs_batch,
                    done_batch,
                ) = map(np.array, zip(*episode_buffer))

                # Compute the returns
                returns = np.zeros_like(reward_batch)
                running_return = 0
                for i in reversed(range(len(reward_batch))):
                    running_return = reward_batch[i] + gamma * running_return * (
                        1 - done_batch[i]
                    )
                    returns[i] = running_return
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

                # Convert the data to tensors and create a DataLoader
                obs_tensor = torch.tensor(obs_batch, dtype=torch.float32)
                action_tensor = torch.tensor(action_batch, dtype=torch.float32)
                log_prob_tensor = torch.tensor(log_prob_batch, dtype=torch.float32)
                return_tensor = torch.tensor(returns, dtype=torch.float32)
                dataset = torch.utils.data.TensorDataset(
                    obs_tensor, action_tensor, log_prob_tensor, return_tensor
                )
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                # Iterate over batches
                for obs_batch, action_batch, log_prob_batch, return_batch in dataloader:
                    # Compute the loss
                    mean, std = policy_network(obs_batch)
                    normal = Normal(mean, std)
                    log_prob_new = normal.log_prob(action_batch) - torch.log(
                        1 - action_batch.pow(2) + 1e-8
                    )
                    log_prob_new = log_prob_new.sum(-1, keepdim=True)
                    ratio = torch.exp(log_prob_new - log_prob_batch)
                    surr1 = ratio * return_batch
                    surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * return_batch
                    loss = -torch.min(surr1, surr2).mean()

                    # Update the policy network
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Clear the episode buffer
                episode_buffer = []

                # Update the running reward
                if running_reward is None:
                    running_reward = info[0]["episode"]["r"]
                else:
                    running_reward = (
                        0.05 * info[0]["episode"]["r"] + (1 - 0.05) * running_reward
                    )

                # Print the progress
                print(
                    f'Epoch {epoch}, Episode {info[0]["episode"]["n"]}, '
                    f'Return {info[0]["episode"]["r"]:.2f}, '
                    f"Running reward {running_reward:.2f}"
                )

                # Save the policy network weights if the running reward is higher than the previous best
                if running_reward > best_running_reward:
                    torch.save(policy_network.state_dict(), "policy_network.pth")
                    best_running_reward = running_reward

                break

            # Close the environment
            # # Plot the running reward over time
            # plt.plot(range(len(running_rewards)), running_rewards)
            # plt.xlabel('Epoch')
            # plt.ylabel('Running Reward')
            # plt.show()

            # vec_env.close()


if __name__ == "__main__":
    # Train the policy network
    env = gym.make("LunarLanderContinuous-v2")
    n_env = 4
    policy_network = Policy(env.observation_space.shape, env.action_space.shape)
    optimizer = optim.Adam(policy_network.parameters(), lr=1e-3)
    train(n_env, policy_network, optimizer, num_epochs=100)
