from Game import Game

class HumanPlayer():
    def __init__(self, game: Game):
        self.game = game

    def play(self, board):
        valid_moves = self.game.getValidMoves(board)
        print('\nMoves:', [i for (i, valid) in enumerate(valid_moves) if valid])

        while True:
            move = int(input())
            if valid_moves[move]: break
            else: print('Invalid move')
        return move