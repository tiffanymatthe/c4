#!/usr/bin/python
from Game import Game
from NeuralNet import NeuralNet
from Config import Config
from MCTS import MCTS
from Arena import Arena
import numpy as np
from collections import deque
from tqdm import tqdm
from random import shuffle
import os
import sys
from pickle import Pickler, Unpickler
from multiprocessing import Pool


def executeEpisodeProcess():
    """
    This function executes one episode of self-play, starting with player 1.
    As the game is played, each turn is added as a training example to
    trainExamples. The game is played till the game ends. After the game
    ends, the outcome of the game is used to assign values to each example
    in trainExamples.
    It uses a temp=1 if episodeStep < tempThreshold, and thereafter
    uses temp=0.
    Returns:
        trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                        pi is the MCTS informed policy vector, v is +1 if
                        the player eventually won the game, else -1.
    """
    trainExamples = []
    game = Game()
    config = Config()
    board = game.getInitBoard()
    curPlayer = 1
    episodeStep = 0

    nnet = NeuralNet(game)
    nnet.load_checkpoint(folder=config.checkpoint,
                         filename='temp.pth.tar', suppress=True)
    mcts = MCTS(game, nnet, config)

    while True:
        episodeStep += 1
        canonicalBoard = game.getCanonicalForm(board, curPlayer)
        temp = int(episodeStep < config.tempThreshold)

        pi = mcts.getActionProb(canonicalBoard, temp=temp)
        sym = game.getSymmetries(canonicalBoard, pi)
        for b, p in sym:
            trainExamples.append([b, curPlayer, p, None])

        action = np.random.choice(len(pi), p=pi)
        board, curPlayer = game.getNextState(
            board, curPlayer, action)

        r = game.getWinState(board, curPlayer)

        if r != 0:
            return [(x[0], x[2], r * ((-1) ** (x[1] != curPlayer))) for x in trainExamples]


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet.
    Modified from https://github.com/suragnair/alpha-zero-general
    """

    def __init__(self, game: Game, nnet: NeuralNet, config: Config):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.config = config
        self.mcts = MCTS(self.game, self.nnet, self.config)
        # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.trainExamplesHistory = []
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.
        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.
        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.config.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(
                board, self.curPlayer, action)

            r = self.game.getWinState(board, self.curPlayer)

            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        startingIndex = 1
        if self.config.iterationNum is not None:
            startingIndex = min(self.config.iterationNum, self.config.numIters + 1)

        for i in range(startingIndex, self.config.numIters + 1):
            # bookkeeping
            print(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > startingIndex:
                if self.config.multiprocessing:
                    iterationTrainExamples, pool = self.generateTrainingDataAsync()
                    pool.close()
                    pool.join()
                else:
                    iterationTrainExamples = self.generateTrainingData()

                # save the iteration examples to the history
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.config.numItersForTrainExamplesHistory:
                print(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(
                folder=self.config.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(
                folder=self.config.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.config)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.config)

            print('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            # more multiprocessing, not efficient in beginning but for most of it yes because
            # neural network takes longer and longer to fit.
            pwins, nwins, draws = arena.playGames(self.config.arenaCompare)

            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' %
                  (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.config.updateThreshold:
                print('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(
                    folder=self.config.checkpoint, filename='temp.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(
                    folder=self.config.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(
                    folder=self.config.checkpoint, filename='best.pth.tar')

    def generateTrainingData(self):
        iterationTrainExamples = deque(
            [], maxlen=self.config.maxlenOfQueue)

        for _ in tqdm(range(self.config.numEps), desc="Self Play"):
            # reset search tree
            self.mcts = MCTS(self.game, self.nnet, self.config)
            iterationTrainExamples += self.executeEpisode()

        return iterationTrainExamples

    def generateTrainingDataAsync(self):
        """Generates training data through multiprocessing.
        Returns object that stores examples (only access after pool is joined and closed) and multiprocessing pool."""
        iterationTrainExamples = deque(
            [], maxlen=self.config.maxlenOfQueue)

        print("Starting multiprocessing.")
        pbar = tqdm(total=self.config.numEps,
                    desc="Self Play", position=0, leave=True)

        def update(result):
            nonlocal iterationTrainExamples
            iterationTrainExamples += result
            pbar.update()

        pool = Pool(self.config.processes)
        for _ in range(self.config.numEps):
            pool.apply_async(executeEpisodeProcess, callback=update)

        return iterationTrainExamples, pool

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.config.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(
            folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(
            self.config.load_folder_file[0], self.config.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            print(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Try with self.load_folder_file_examples? [y|n]")
            if r != "y":
                sys.exit()
            self.loadPreviousExamples()
        else:
            print("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            print('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True

    def loadPreviousExamples(self):
        examplesFile = os.path.join(
            self.config.load_folder_file_examples[0],
            self.config.load_folder_file_examples[1]) + ".examples"
        if not os.path.isfile(examplesFile):
            print(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples for previous model found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            print('Loading done!')

            # do not skip first play since only old data and not from latest model
