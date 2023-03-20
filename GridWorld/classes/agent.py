import random
from policies import pi_selector
import numpy as np

class TabularAgent():
    """
    Agent to interact with the board environment. An agent will recieve a state description (either exactly, or parameterized),
    and will based on a policy that might depend on a value-action function decide what to do.
    """
    def __init__(self, x:int = 0,y:int = 0,grid_size: int = 5, policy: str = "random", eps: float = 0):
        self.value_action_function = np.ones((grid_size,grid_size))
        self.x = x
        self.y = y
        self.n_moves = 0
        self.hist = np.empty((3,16)) # [state, action, reward] x timestep (start w. 16 timesteps, and increase if needed)
        self.eps = eps
        self.pi = pi_selector(policy_type=policy)

    def get_action(self,Board):
        """
        Get state and valid actions from board, and use policy pi() to get action to take.
        :param Board:
        :return:
        """
        cur_state = Board.get_current_state()
        legal_actions = Board.get_valid_actions()
        action = self.pi(cur_state,legal_actions)
        return cur_state,action

    def take_action(self,Board):
        cur_state, action = self.get_action(Board)





