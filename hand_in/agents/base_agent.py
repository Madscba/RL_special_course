#An abstract base class for RL agents with common functionality.

class BaseAgent():
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

    def uses_replay_buffer(self):
        return False


