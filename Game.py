#!/usr/bin/python
from Board import Board
import numpy as np

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

    @staticmethod
    def display(board: Board):
        print(" -----------------------")
        coloured_state = np.vectorize(
            Game.__get_color_coded_background)(board.state)
        Game.__print_a_ndarray(coloured_state, row_sep="")
        print(" -----------------------")

    @staticmethod
    def __get_color_coded_background(i):
        """
        Get marker to print for a certain player or empty space.
        """
        colors = {
            0: "\033[40m - \033[0m",  # black
            -1: "\033[41m o \033[0m",  # red
            1: "\033[43m x \033[0m"  # yellow
        }

        return colors[i]

    @staticmethod
    def __print_a_ndarray(map1, row_sep=" "):
        """
        Prints array.
        https://stackoverflow.com/questions/56496731/coloring-entries-in-an-matrix-2d-numpy-array/56497272
        """
        n, m = map1.shape
        vertical_padding = 2
        m = m + vertical_padding  # for vertical axis
        fmt_str = "\n".join([row_sep.join(["{}"]*m)]*n)
        column_labels = ['0', '1', '2', '3', '4', '5', '6']
        row_labels = ['a', 'b', 'c', 'd', 'e', 'f']

        map1 = np.pad(map1, [(0, 0), (vertical_padding, 0)])

        for i, label in enumerate(row_labels):
            map1[i, 1] = ' | '
            map1[i, 0] = label

        print(fmt_str.format(*map1.ravel()))
        col_axis = '  '.join(map(str, column_labels))
        print('    ' + '―' * 20)
        print('    ' + ' ' + col_axis)
