import torch
import numpy as np
class ReplayBuffer:
    def __init__(self, capacity: int = 100000, state_dim: int = 4,action_dim:int = 2, n_actions:int=2, batch_size: int = 64):
        self.capacity = capacity
        self.event_idx = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_actions = n_actions
        self.event_tuples = torch.zeros((2 * state_dim + n_actions + 2, capacity))
        # ('state','action','reward',next_state','terminated')
        self.batch_size = batch_size

    def save_event(self, state, action, reward, next_state, terminated):
        if isinstance(action,list):
            a = 2
        else:
            b = 2
        self.event_tuples[:, self.event_idx % self.capacity] = torch.hstack(
            (
                torch.from_numpy(state),
                torch.from_numpy(action).reshape(-1, self.n_actions),
                torch.from_numpy(reward.reshape(-1, 1)),
                torch.from_numpy(next_state),
                torch.from_numpy(terminated.reshape(-1, 1)),
            )
        )
        self.event_idx += 1
        if self.event_idx % 100000 == 0:
            print("starting new buffer")

    def get_batch_of_events(self):
        if self.event_idx >= self.batch_size:
            sample_idx = np.random.choice(
                np.min([self.event_idx, self.capacity]), self.batch_size, replace=False
            )
            return self.event_tuples[:, sample_idx]
