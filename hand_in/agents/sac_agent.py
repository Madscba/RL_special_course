#A class implementing a soft actor critic agent.
import copy
import numpy as np
import torch

from hand_in.agents.base_agent import BaseAgent
from hand_in.models.actor_continuous import ActorNetwork_cont
from hand_in.models.actor_discrete import ActorNetwork_disc
from hand_in.models.critic_model import CriticNetwork
from hand_in.utils.replay_buffer import ReplayBuffer

#https://www.youtube.com/watch?v=_nFXOZpo50U
#https://www.youtube.com/watch?v=U20F-MvThjM&t=0s
#https://www.youtube.com/watch?v=ioidsRlf79o

#https://arxiv.org/pdf/1812.05905.pdf

class SACAgent(BaseAgent):
    def __init__(self,  argparser,action_dim, state_dim, n_actions, action_type):
        self.parser = argparser
        self.continuous = action_type == "continuous"
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.alpha = argparser.args.alpha
        self.gamma = argparser.args.gamma
        self.tau = argparser.args.tau
        self.critic_primary = CriticNetwork(argparser = argparser, input_dim=state_dim + n_actions, output_dim= 1, n_actions = n_actions, action_type = action_type)
        self.critic_secondary = CriticNetwork(argparser = argparser, input_dim =state_dim + n_actions, output_dim= 1, n_actions = n_actions, action_type = action_type)
        self.critic_target_primary = self.initialize_target(self.critic_primary)
        self.critic_target_secondary = self.initialize_target(self.critic_secondary)
        self.actor_network = self.init_actor_network(argparser,action_dim, state_dim, n_actions,action_type)
        self.replay_buffer = ReplayBuffer(capacity=10e5, state_dim=state_dim, action_dim=action_dim,n_actions=n_actions
                                          , used_for_policy_gradient_method=True)




    def initialize_target(self,network_to_be_copied):
        target_network = copy.deepcopy(network_to_be_copied)
        for p in target_network.parameters():
            p.requires_grad = False
        return target_network
    def update_policy(self, states, actions, rewards, new_states, terminated, policy_response_dict:dict):
        log_probs,entropy = policy_response_dict['log_probs'],policy_response_dict['entropy']
        state_action_tensor = torch.Tensor(np.vstack((states,actions)))

        #### Update Q function networks
        # find minimum of target networks
        critic_value_target_min = torch.min(self.critic_target_secondary(state_action_tensor),
                                            self.critic_target_secondary(state_action_tensor)
                                            )
        reward = torch.Tensor(rewards).reshape(-1, 1)
        entropy_term_objective = (self.alpha * log_probs)#(entropy.squeeze(0)).sum())

        # add terms together eq 6 from SAC paper:
        critic_target = reward + self.gamma * (critic_value_target_min * torch.tensor((1 - terminated.reshape(-1, 1)))
                                               - entropy_term_objective)

        critic_value_prim = self.critic_primary(state_action_tensor)
        critic_value_sec = self.critic_secondary(state_action_tensor)

        value_loss_prim = self.critic_network.criterion(critic_value_prim, critic_target)
        value_loss_sec = self.critic_network.criterion(critic_value_sec, critic_target)

        self.critic_primary.optimizer.zero_grad()
        self.critic_secondary.optimizer.zero_grad()

        value_loss_prim.backward(retain_graph=True)
        value_loss_sec.backward(retain_graph=True)

        if self.parser.args.grad_clipping:
            torch.nn.utils.clip_grad_norm_(
                [p for g in self.critic_network_primary.optimizer.param_groups for p in g["params"]],
                self.parser.args.grad_clipping,)
        if self.parser.args.grad_clipping:
            torch.nn.utils.clip_grad_norm_(
                [p for g in self.critic_network_secondary.optimizer.param_groups for p in g["params"]],
                self.parser.args.grad_clipping,)

        self.critic_primary.optimizer.step()
        self.critic_secondary.optimizer.step()


        #### Update actor networks
        print("check if log_probs.squeeze is needed")
        first_term = self.alpha * log_probs.squeeze(0)
        second_term = (self.alpha * log_probs.squeeze(0) - critic_value_target_min)
        action_loss = first_term + second_term
        self.actor_network.optimizer.zero_grad()
        action_loss.backward()
        # if self.parser.args.grad_clipping:
        #     torch.nn.utils.clip_grad_norm_(
        #         [p for g in self.critic_network.optimizer.param_groups for p in g["params"]],
        #         self.parser.args.grad_clipping,)
        self.actor_network.optimizer.step()

        #### Update Q function target networks
        for param, target_param in zip(self.critic_primary.parameters(), self.critic_target_primary.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic_secondary.parameters(), self.critic_target_secondary.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
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
