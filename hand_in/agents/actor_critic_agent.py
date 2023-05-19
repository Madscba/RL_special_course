#A class implementing a policy gradient-based actor critic agent.
from hand_in.agents.base_agent import BaseAgent

class ACAgent(BaseAgent):
    def __init__(self, argparser,action_dim, state_dim, n_actions, action_type):
        self.parser = argparser
        self.continuous = action_type == "continuous"
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.actor_network = self.init_actor_network(argparser, action_dim, state_dim, n_actions, action_type)
        self.replay_buffer = ReplayBuffer(capacity=1005, state_dim=state_dim, action_dim=action_dim,
                                          n_actions=n_actions, used_for_policy_gradient_method=True)

    def initialize_policy(self):
        pass
    def update_policy(self):
        pass

    def follow_policy(self):
        pass

    def save_models(self):
        pass

    def load_models(self):
        pass