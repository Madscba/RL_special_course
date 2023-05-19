import torch
import numpy as np


class EpisodeBuffer:
    def __init__(
        self,
        envs,
        capacity: int = 100000,
        state_dim: int = 8,
        action_dim: int = 2,
        batch_size: int = 64,
    ):
        self.capacity = capacity
        self.event_idx = 0
        self.action_dim = action_dim
        state_dim = envs.single_observation_space.shape[0]

        self.event_tuples = torch.zeros(
            (envs.action_space.shape[0], 2 * state_dim + 3 * action_dim + 2, capacity)
        )
        # we want to save: ('state','action','reward',next_state','terminated', log_probs, entropy)
        # +2 as we need room for reward, terminated
        self.batch_size = batch_size

    def save_event(
        self, state, action, reward, next_state, terminated, log_probs, entropy
    ):
        self.event_tuples[:, :, self.event_idx % self.capacity] = torch.hstack(
            (
                torch.from_numpy(state),
                torch.from_numpy(action).reshape(-1, self.action_dim),
                torch.from_numpy(reward.reshape(-1, 1)),
                torch.from_numpy(next_state),
                torch.from_numpy(terminated.reshape(-1, 1)),
                log_probs.reshape(-1, self.action_dim),
                entropy.reshape(-1, self.action_dim),
            )
        )
        self.event_idx += 1
        if self.event_idx % 100000 == 0:
            print("starting new buffer")

    def get_episode_hist(self):
        return self.event_tuples[:, :, : self.event_idx]


    def get_batch_of_events(self):
        if self.event_idx >= self.batch_size:
            sample_idx = np.random.choice(
                np.min([self.event_idx, self.capacity]), self.batch_size, replace=False
            )
            return self.event_tuples[:, sample_idx]
