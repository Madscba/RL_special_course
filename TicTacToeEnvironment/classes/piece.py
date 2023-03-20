import numpy as np

class Piece():
    def __init__(self,symbol: str, active: bool = False,pos: tuple = None):
        self.symbol = symbol
        self.active = active
        self.pos = pos

    def place_piece(self,new_pos):
        self.pos = new_pos
        self.active = True

    def update_position(self,new_pos):
        self.pos = new_pos

