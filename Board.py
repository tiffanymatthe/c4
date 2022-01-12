#!/usr/bin/python
import numpy as np


class Board():
    def __init__(self, state=None) -> None:
        self.height = 6
        self.width = 7
        self.win_length = 4
        self.empty_token = 0
        self.player1 = 1
        self.player2 = -1
        self.state = np.copy(state)
        if state is None:
            self.state = np.zeros((self.height, self.width)).astype(float)

    def move(self, player: int, action: int):
        try:
            col = self.state.transpose().tolist()[action]
            empty_row = self.height - 1 - col[::-1].index(self.empty_token)
            self.state[empty_row, action] = player
        except ValueError:
            raise ValueError("Column is full.")

    def getValidMoves(self):
        return self.state[0] == 0

    def fullBoard(self):
        return np.all(self.state[0] != 0)

    def getWinValue(self):
        """
        Returns 1 if player1 wins, -1 if player2 wins,
        0 if tied, and -100 if game is not terminal.
        """
        arr_list = self.state.transpose().tolist()
        arr_list.extend(self.state.tolist())

        diagonal_indices = range(-2, 4)
        for idx in diagonal_indices:
            main_diag = np.diagonal(self.state, offset=idx)
            anti_diag = np.diagonal(np.fliplr(self.state), offset=idx)
            arr_list.append(main_diag.tolist())
            arr_list.append(anti_diag.tolist())

        p1_win = self.__is_win_state_in_list(arr_list, self.player1)
        p2_win = self.__is_win_state_in_list(arr_list, self.player2)

        if p1_win:
            return self.player1
        elif p2_win:
            return self.player2
        elif self.fullBoard():
            return 0
        else:
            return -100

    def __is_win_state_in_list(self, arr_list, token) -> bool:
        """
        Parameters
        ----------
        arr_list : list
            list of arrays
        token
            value to find its longest sequence in each array
        If there is a sequence of length >= 4 of value token in one of the arrays in
        list, then returns True. Else False.
        """
        for arr in arr_list:
            arr = np.array(arr)
            if (arr == token).sum() < self.win_length:
                continue
            else:
                if self.__find_longest_seq(arr, token) >= self.win_length:
                    return True
        return False

    def __find_longest_seq(self, arr, val) -> int:
        """
        Parameters
        ----------
        arr : np.ndarray
            array containing val and other values
        val
            value in array to find the longest unbroken sequence of
        Returns length of longest sequence of val in arr.
        Inspired by https://stackoverflow.com/a/38161867
        """
        idx_pairs = np.where(np.diff(np.hstack(([False], arr == val, [False]))))[
            0].reshape(-1, 2)
        seq_lengths = np.diff(idx_pairs, axis=1)
        if len(seq_lengths) > 0:
            return max(np.diff(idx_pairs, axis=1))
        else:
            return 0

    def __str__(self):
        return str(self.state)
