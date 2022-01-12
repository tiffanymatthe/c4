#!/usr/bin/python
from Board import Board


class Game():
    """
    Represents a game state of Connect 4.
    Player1 is represented by 1, player2 is represented by -1.
    """

    def __init__(self) -> None:
        pass

    def getNextState(self, board: Board, player: int, action: int):
        """
        Returns new board representing next state.
        Original board is unmodified.
        player is 1 or 0.
        Action is a column number from 0 to 6.

        Returns new board and next player.
        """
        board = Board(board.state)
        board.move(player, action)
        return board, -player

    def getValidMoves(self, board: Board):
        return board.getValidMoves()

    def getWinState(self, board: Board, player: int):
        """
        Returns 1 if player won, -1 if lost, 1e-4 if tied,
        and 0 if game is not finished.
        """
        win_value = board.getWinValue()
        if win_value == -100:
            return 0  # unfinished game
        if win_value == 0:
            return 1e-4  # draw
        if win_value == player:
            return 1
        elif win_value == -player:
            return -1
        else:
            raise ValueError(f"Unexpected win value {win_value}")

    def getCanonicalForm(self, board: Board, player: int) -> Board:
        return Board(board.state * player)

    def stringRepresentation(self, board: Board):
        return str(board)

    def getActionSize(self) -> int:
        return Board().width

    def getBoardSize(self):
        base_board = Board()
        return base_board.width, base_board.height

    def getInitBoard(self):
        return Board()

    def getSymmetries(self, board: Board, pi):
        """Board is left/right board symmetric"""
        state = board.state
        return [(state, pi), (state[:, ::-1], pi[::-1])]
