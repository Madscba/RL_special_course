# Purpose of environment class:
## This class will contain all the aspects of the game. The players, the board, and the game dynamics.

class Env_TicTacToe():
    def __init__(self, board,agents):
        self.board = board
        self.agents = agents
        self.agent_turn = False

    def RunGame(self):
        if self.agent_turn:
            self.agents[0].make_move()
            self.agent_turn = ~self.agent_turn
        else:


if __name__ == "__main__":
    # board = Board()
    # player1 = ComputerAgent("computer")
    # player2 = HumanAgent("human")
    # TicTacToeEnvironment(board, [player1,player2])


