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
            store_on_GPU_w_grad = True
        )


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
                          ]
            log_prob_idx = 2 * self.actor_network.input_dim + self.n_actions + 2
            # ('state','action','reward',next_state','terminated', log_probs, entropy)
            rel_log_probs = rel_episode_hist[
                            log_prob_idx : log_prob_idx + self.n_actions, :
                            ].squeeze(0)
            discounted_rewards = torch.zeros_like(rel_rewards)

            for i in range(episode_history.shape[1]):
                g_total = torch.zeros(1, 1, requires_grad=False)
                r_discount = 1
                for j in range(i,episode_history.shape[1]):
                    g_total += rel_rewards[j] * r_discount
                    r_discount *= self.parser.args.gamma
                discounted_rewards[i] = g_total


            r_mean = discounted_rewards.mean()
            r_std = discounted_rewards.std()
            normalized_rewards = (discounted_rewards - r_mean) / r_std
            # for t in reversed(range(episode_history.shape[1])):
            #     G = self.parser.args.gamma * G + rel_rewards[t]
            #     loss = loss - (rel_log_probs[:, t] * G).sum()

            # loss = Variable(torch.Tensor([0]), requires_grad=True)  # reset loss to zero
            loss = 0  # reset loss to zero
            for (rew,log_prob) in zip(normalized_rewards,rel_log_probs):
                loss += -rew * log_prob

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
        if not isinstance(state, torch.Tensor):
            state = torch.Tensor([state]).to(self.actor_network.device)


        if self.continuous:
            action, log_probs, entropy, info_dict = self.actor_network(state)
        else:
            action, log_probs, entropy, info_dict = self.actor_network.get_action_and_log_prob(state)

        if self.continuous:
            policy_response_dict = {
                "log_probs": log_probs,
                "entropy": entropy,
            }
        else:
            policy_response_dict = {"log_probs": log_probs, "entropy": entropy}
        return action.numpy()[0], policy_response_dict

    def init_actor_network(
        self, argparser, action_dim, state_dim, n_actions, action_type
    ):
        if self.continuous:
            return ActorNetwork_cont(
                argparser=argparser, action_dim=action_dim, state_dim=state_dim, name = "actor_cont"
            )
        else:
            return ActorNetwork_disc(
                argparser=argparser, action_dim=action_dim, state_dim=state_dim, name = "actor_disc"
            )

    def save_models(self):
        print("saving reinforce models:")
        self.actor_network.save_model_checkpoint()

    def load_models(self):
        print("loading reinforce models:")
        self.actor_network.load_model_checkpoint()

    def uses_replay_buffer(self):
        """REINFORCE use an episode buffer"""
        return True
