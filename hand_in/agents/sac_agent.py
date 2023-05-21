#A class implementing a soft actor critic agent.
import torch

from hand_in.agents.base_agent import BaseAgent
from hand_in.models.actor_continuous import ActorNetwork_cont
from hand_in.models.actor_discrete import ActorNetwork_disc
from hand_in.models.critic_model import CriticNetwork
from hand_in.utils.replay_buffer import ReplayBuffer

#https://www.youtube.com/watch?v=_nFXOZpo50U
#https://www.youtube.com/watch?v=U20F-MvThjM&t=0s
#https://www.youtube.com/watch?v=ioidsRlf79o

class SACAgent(BaseAgent):
    def __init__(self,  argparser,action_dim, state_dim, n_actions, action_type):
        self.parser = argparser
        self.continuous = action_type == "continuous"
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.alpha = argparser.args.alpha
        self.gamma = argparser.args.gamma
        self.critic_network_primary = CriticNetwork(argparser, state_dim, action_dim, n_actions, action_type)
        self.critic_network_secondary = CriticNetwork(argparser, state_dim, action_dim, n_actions, action_type)
        self.actor_network = self.init_actor_network(argparser,action_dim, state_dim, n_actions,action_type)
        self.replay_buffer = ReplayBuffer(capacity=10e5, state_dim=state_dim, action_dim=action_dim,
                                          n_actions=n_actions, used_for_policy_gradient_method=True)


    def initialize_policy(self):
        pass
    def update_policy(self, states, actions, rewards, new_states, terminated, policy_response_dict:dict):
        log_probs,entropy = policy_response_dict['log_probs'],policy_response_dict['entropy']

        #Step 1 calculate Qvalue losses for the two networks.
        reward = torch.Tensor(rewards).reshape(-1, 1)
        entropy_term_objective = (self.alpha * log_probs)#(entropy.squeeze(0)).sum())

        critic_value_prim = self.critic_network_primary(torch.Tensor(states))
        critic_value_sec = self.critic_network_secondary(torch.Tensor(states))





        # Calculate reward and critic estimate
             self.gamma * self.critic_network(torch.Tensor(new_states)) * torch.tensor((1 - terminated.reshape(-1, 1)))

        # Calculate losses for actor and critic
        value_loss = self.critic_network.criterion(critic_value_est.clone(), reward.clone())
        action_loss = -(log_probs.squeeze(0) * (reward - critic_value_est).detach()).sum()
        # add entropy
        if self.parser.args.entropy:
            action_loss += (0.5 * (entropy.squeeze(0)).sum())  # + appears to work decently or at least better
        # For debugging purposes
        # Update networks
        # Update networks (with or without grad clipping)
        self.critic_network.optimizer.zero_grad()
        value_loss.backward(retain_graph=True)
        # if self.parser.args.grad_clipping:
        #     torch.nn.utils.clip_grad_norm_(
        #         [p for g in self.critic_network.optimizer.param_groups for p in g["params"]],
        #         self.parser.args.grad_clipping,)
        self.critic_network.optimizer.step()

        self.actor_network.optimizer.zero_grad()
        action_loss.backward()
        # if self.parser.args.grad_clipping:
        #     torch.nn.utils.clip_grad_norm_(
        #         [p for g in self.critic_network.optimizer.param_groups for p in g["params"]],
        #         self.parser.args.grad_clipping,)
        self.actor_network.optimizer.step()
    def follow_policy(self,state):
        state_tensor = torch.from_numpy(state).float()
        action, log_probs, entropy, info_dict = self.actor_network(state_tensor)

        if self.continuous:
            print("potentially worth logging mu and std")
            mu = info_dict["mu"]
            sigma_sq = info_dict["sigma_sq"]
            policy_response_dict = {"mu": mu, "sigma_sq": sigma_sq, "log_probs": log_probs, "entropy": entropy}
        else:
            policy_response_dict = {"log_probs": log_probs, "entropy": entropy}
        return action.numpy(), policy_response_dict

    def save_models(self):
        pass

    def load_models(self):
        pass

    def init_actor_network(self, argparser,action_dim, state_dim, n_actions,action_type):
        if self.continuous:
            return ActorNetwork_cont(argparser=argparser, action_dim=action_dim,state_dim= state_dim)
        else:
            return ActorNetwork_disc(argparser=argparser, action_dim=action_dim,state_dim= state_dim)
