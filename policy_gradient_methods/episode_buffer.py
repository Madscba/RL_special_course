import torch
import numpy as np


class EpisodeBuffer():
    def __init__(self, envs, capacity: int = 100000, state_dim: int = 8,action_dim:int=2, batch_size: int = 64):
        self.capacity = capacity
        self.event_idx = 0
        self.event_tuples = torch.zeros((envs.action_space.shape[0],2 * state_dim + action_dim + 2, capacity))
        # ('state','action','reward',next_state','terminated')
        self.batch_size = batch_size

    def save_event(self, state, action, reward, next_state, terminated):
        self.event_tuples[:,:, self.event_idx % self.capacity] = torch.from_numpy(
            np.hstack((state, action, reward.reshape(-1,1), next_state, terminated.reshape(-1,1))))
        self.event_idx += 1
        if self.event_idx % 100000 == 0:
            print("starting new buffer")

    def get_episode_hist(self):
        return self.event_tuples[:, :,:self.event_idx]

    def get_batch_of_events(self):
        if self.event_idx >= self.batch_size:
            sample_idx = np.random.choice(np.min([self.event_idx, self.capacity]), self.batch_size, replace=False)
            return self.event_tuples[:, sample_idx]
