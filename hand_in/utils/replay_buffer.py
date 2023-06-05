import torch
import numpy as np


class ReplayBuffer:
    """
    Replay buffer for RL methods. Works for value-based methods where episode information will be saved:
        (reward, action, reward, next_state,terminated),
    and for policy-based methods the log_probability of actions and the entropy is additionally saved:
        (reward, action, reward, next_state, terminated, log_probability, entropy)
    a class attribute: self.used_for_policy_gradient_method is used to differentiate between the two cases.
    """

    def __init__(
        self,
        capacity: int = 100000,
        state_dim: int = 4,
        action_dim: int = 2,
        n_actions: int = 2,
        batch_size: int = 64,
        used_for_policy_gradient_method: bool = False,
        store_on_GPU_w_grad: bool = False
    ):
        self.capacity = int(capacity)
        self.event_idx = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_actions = n_actions
        self.used_for_policy_gradient_method = used_for_policy_gradient_method
        self.store_on_GPU_w_grad = store_on_GPU_w_grad
        self.event_tuples = self.initialize_array(self.capacity)
        # ('state','action','reward',next_state','terminated')
        self.batch_size = batch_size

    def initialize_array(self, capacity):
        if self.used_for_policy_gradient_method:
            return torch.zeros((2 * self.state_dim + 3 * self.n_actions + 2, capacity))
        else:
            return torch.zeros((2 * self.state_dim + self.n_actions + 2, capacity))

    def save_event(
        self, state, action, reward, next_state, terminated, policy_response_dict, info
    ):
        if not len(info.keys())==0: #If environment terminates, then new_state will be the state of the reset environment
            next_state = info['final_observation'][0].reshape(1,-1)

        if self.used_for_policy_gradient_method:
            if not self.store_on_GPU_w_grad:
                self.event_tuples[:, self.event_idx % self.capacity] = torch.hstack(
                    (
                        torch.from_numpy(state).cpu(),
                        torch.from_numpy(action).reshape(-1, self.n_actions).cpu(),
                        torch.from_numpy(reward.reshape(-1, 1)).cpu(),
                        torch.from_numpy(next_state).cpu(),
                        torch.from_numpy(terminated.reshape(-1, 1)).cpu(),
                        policy_response_dict["log_probs"].reshape(-1, self.n_actions).detach().cpu(),
                        policy_response_dict["entropy"].reshape(-1, self.n_actions).detach().cpu(),
                    )
                )
            else:
                self.event_tuples[:, self.event_idx % self.capacity] = torch.hstack(
                    (
                        torch.from_numpy(state),
                        torch.from_numpy(action).reshape(-1, self.n_actions),
                        torch.from_numpy(reward.reshape(-1, 1)),
                        torch.from_numpy(next_state),
                        torch.from_numpy(terminated.reshape(-1, 1)),
                        policy_response_dict["log_probs"].reshape(-1, self.n_actions),
                        policy_response_dict["entropy"].reshape(-1, self.n_actions).detach(),
                    )
                )
        else:
            self.event_tuples[:, self.event_idx % self.capacity] = torch.hstack(
                (
                    torch.from_numpy(state),
                    torch.from_numpy(action).reshape(-1, self.n_actions),
                    torch.from_numpy(reward.reshape(-1, 1)),
                    torch.from_numpy(next_state),
                    torch.from_numpy(terminated.reshape(-1, 1)),
                )
            )
            if self.event_idx % self.capacity == 0:
                print("starting new buffer")
        self.event_idx += 1

    def get_batch_of_events(self):
        """get a group of steps info"""
        if self.event_idx >= self.batch_size:
            sample_idx = np.random.choice(
                np.min([self.event_idx, self.capacity]), self.batch_size, replace=False
            )
            return self.event_tuples[:, sample_idx]

    def get_episode_hist(self):
        """get the all steps info so far saved in replay buffer"""
        return self.event_tuples[:, : self.event_idx]

    def reset(self):
        self.event_idx = 0
        self.event_tuples = self.initialize_array(capacity=1005)

    def get_latest_termination_state_and_check_for_max_episode_length(
        self, max_steps_in_episode: int = 1000
    ):
        """For REINFORCE we want to update model, if we have finished an episode either by having naturally
        terminated, or having reached the maximum allowed amount of steps."""
        if self.used_for_policy_gradient_method:
            return (
                bool(self.event_tuples[-1 - 2 * self.n_actions, self.event_idx - 1])
                or self.event_idx >= 1000
            )
        else:
            return self.event_tuples[:, self.event_idx - 1] or self.event_idx >= 1000
