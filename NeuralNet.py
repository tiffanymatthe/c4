#!/usr/bin/python
import warnings
warnings.filterwarnings("ignore")
import os
from C4Model import C4Model
from Config import Config
from Game import Game
import numpy as np
from Board import Board
from tqdm import tqdm


class NeuralNet():
    """
    The neural network does not consider the current player, and instead only deals with
    the canonical form of the board. Modified from https://github.com/suragnair/alpha-zero-general
    """

    def __init__(self, game: Game):
        config = Config()
        self.nnet = C4Model(game, config)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.config = config

    def train(self, examples):
        """
        This function trains the neural network with examples obtained from
        self-play.
        Input:
            examples: a list of training examples, where each example is of form
                      (board.state, pi, v). pi is the MCTS informed policy vector for
                      the given board, and v is its value. The examples has
                      board in its canonical form.
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)

        verbose = 1
        if self.config.multiprocessing:
            verbose = 2
            tqdm.write("\033[92mTraining model with verbose=2.\033[0m.")

        self.nnet.model.fit(x=input_boards, y=[target_pis, target_vs],
                            batch_size=self.config.batch_size, epochs=self.config.epochs,
                            verbose=verbose)
                            
        if self.config.multiprocessing:
            tqdm.write("\033[92mFinished training\033[0m.")

    def predict(self, board: Board):
        """
        Input:
            board: current board in its canonical form.
        Returns:
            pi: a policy vector for the current board- a numpy array of length
                game.getActionSize
            v: a float in [-1,1] that gives the value of the current board
        """
        state = board.state[np.newaxis, :, :]
        pi, v = self.nnet.model(state, training=False)
        return pi[0], v[0]

    def save_checkpoint(self, folder, filename):
        """
        Saves the current neural network (with its parameters) in
        folder/filename
        """
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(
                "Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder, filename, suppress=False):
        """
        Loads parameters of the neural network from folder/filename
        """
        filepath = os.path.join(folder, filename)
        self.nnet.model.load_weights(filepath)
        if not suppress:
            print('Loading Weights...')
