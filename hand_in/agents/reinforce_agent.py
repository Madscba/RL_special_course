# A class implementing a reinforce agent.
import torch
from torch.autograd import Variable

from hand_in.agents.base_agent import BaseAgent
from hand_in.models.actor_continuous import ActorNetwork_cont
from hand_in.models.actor_discrete import ActorNetwork_disc
from hand_in.utils.replay_buffer import ReplayBuffer


class ReinforceAgent(BaseAgent):
    def __init__(self, argparser, action_dim, state_dim, n_actions, action_type):
        self.parser = argparser
        self.continuous = action_type == "continuous"
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.actor_network = self.init_actor_network(
            argparser, action_dim, state_dim, n_actions, action_type
        )
        self.replay_buffer = ReplayBuffer(
            capacity=1005,
            state_dim=state_dim,
            action_dim=action_dim,
            n_actions=n_actions,
            used_for_policy_gradient_method=True,
        )

    def initialize_policy(self):
        pass

    def update_policy(
        self, state, action, reward, new_state, terminated, policy_response_dict: dict
    ):
        terminated_last_step = (
            self.replay_buffer.get_latest_termination_state_and_check_for_max_episode_length()
        )
        if terminated_last_step:
            episode_history = self.replay_buffer.get_episode_hist()
            rel_episode_hist = episode_history.clone()
            rel_rewards = rel_episode_hist[
                self.actor_network.input_dim + self.n_actions, :
            ].clone()
            log_prob_idx = 2 * self.actor_network.input_dim + self.n_actions + 2
            # ('state','action','reward',next_state','terminated', log_probs, entropy)
            rel_log_probs = rel_episode_hist[
                log_prob_idx : log_prob_idx + self.n_actions, :
            ].clone()
            G = torch.zeros(1, 1, requires_grad=False)
            loss = Variable(torch.Tensor([0]), requires_grad=True)  # reset loss to zero
            for t in reversed(range(episode_history.shape[1])):
                G = self.parser.args.gamma * G + rel_rewards[t]
                loss = loss - (rel_log_probs[:, t] * G).sum()

            self.actor_network.optimizer.zero_grad()
            loss.backward()  # set retain_graph to True
            if self.parser.args.grad_clipping:
                torch.nn.utils.clip_grad_norm_(
                    [
                        p
                        for g in self.actor_network.optimizer.param_groups
                        for p in g["params"]
                    ],
                    1,
                )
            self.actor_network.optimizer.step()
            self.replay_buffer.reset()

    def follow_policy(self, state):
        state_tensor = torch.from_numpy(state).float()
        action, log_probs, entropy, info_dict = self.actor_network(state_tensor)

        if self.continuous:
            print("potentially worth logging mu and std")
            mu = info_dict["mu"]
            sigma_sq = info_dict["sigma_sq"]
            policy_response_dict = {
                "mu": mu,
                "sigma_sq": sigma_sq,
                "log_probs": log_probs,
                "entropy": entropy,
            }
        else:
            policy_response_dict = {"log_probs": log_probs, "entropy": entropy}
        return action.numpy(), policy_response_dict

    def init_actor_network(
        self, argparser, action_dim, state_dim, n_actions, action_type
    ):
        if self.continuous:
            return ActorNetwork_cont(
                argparser=argparser, action_dim=action_dim, state_dim=state_dim
            )
        else:
            return ActorNetwork_disc(
                argparser=argparser, action_dim=action_dim, state_dim=state_dim
            )

    def save_models(self):
        pass

    def load_models(self):
        pass

    def uses_replay_buffer(self):
        """REINFORCE use an episode buffer"""
        return True
