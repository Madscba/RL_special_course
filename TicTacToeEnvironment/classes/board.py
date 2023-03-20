import numpy as np


class Board():
    def __init__(self):
        self.board = np.zeros((3,3))

    def update(self,piece, cur_pos,next_pos):
        assert self.board[cur_pos]
        self.board
