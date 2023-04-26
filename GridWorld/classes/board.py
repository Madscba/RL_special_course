import random
import numpy as np
from agent import TabularAgent


class Board:
    """
    The board class is responsible for keeping track of the current state in the grid world. The board is represented by a np.array with four possible values:
    0: Empty tile, no punishment or reward for the agent.
    1: Hole, which leads to termination of episode if agent moves here.
    2: Goal state, a reward is given to the agent when moving here.
    3: The player is currently here.
    """

    def __init__(
        self,
        size: int = 5,
        n_hole: int = 2,
        agent: TabularAgent = TabularAgent(),
        seed: int = 3,
    ):
        np.random.seed(seed)
        self.size = size
        self.n_hole = n_hole
        self.hole_coordinates = np.empty((n_hole, 2))
        self.agent = agent
        self.initialize_board(self.size, self.n_hole)

    def __repr__(self):
        return str(self.board)

    def initialize_board(self, board_size: int, n_hole: int):
        self.board = np.zeros((board_size, board_size))
        coordinates = np.random.randint(0, board_size - 1, size=n_hole * 2)
        for i in range(n_hole):
            x_cord = coordinates[i * 2]
            y_cord = coordinates[i * 2 + 1]
            self.hole_coordinates[i, :] = [x_cord, y_cord]
            self.board[x_cord, y_cord] = 1
        self.board[board_size - 1, board_size - 1] = 2
        self.board[self.agent.x, self.agent.y] = 3

    def get_terminal_states(self):
        yield self.hole_coordinates


if __name__ == "__main__":
    Board1 = Board()
    print(Board1)
