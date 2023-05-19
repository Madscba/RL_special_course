#A class implementing a soft actor critic agent.
from hand_in.agents.base_agent import BaseAgent

class ACAgent(BaseAgent):
    def __init__(self):
        pass

    def initialize_policy(self):
        pass
    def update_policy(self, state, action, reward, new_state, terminated):
        pass

    def follow_policy(self):
        pass

    def save_models(self):
        pass

    def load_models(self):
        pass