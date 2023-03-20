import numpy as np
from piece import Piece
class TabularAgent():
    def __init__(self, symbol: str = "X", state_space):
        self.symbol = symbol
        self.action_
        self.pieces = [Piece(symbol = symbol) for i in range(3)]
        self.move_counter = 0

class TabularHumanAgent(TabularAgent):


class TabularComputerAgent(TabularAgent):



