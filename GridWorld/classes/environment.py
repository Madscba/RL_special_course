import numpy as np
from board import Board
from agent import TabularAgent

class Env_GridWorld():
    def __init__(self):
        self.agent = TabularAgent()
        self.board = Board(agent=self.agent)
        self.terminal = False

    def env_engine(self):
        if ~(self.terminal):





if __name__ == "__main__":

