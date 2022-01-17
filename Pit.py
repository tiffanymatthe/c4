#!/usr/bin/python
import Arena
from Game import Game
from NeuralNet import NeuralNet as NNet
import numpy as np
from HumanPlayer import HumanPlayer

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

human_vs_cpu = True

g = Game()

# nnet players
n1 = NNet(g)
n1.load_checkpoint('./models/','checkpoint_38.pth.tar')
n1p = lambda x: np.argmax(n1.predict(x)[0])

if human_vs_cpu:
    player2 = HumanPlayer(g).play
else:
    n2 = NNet(g)
    n2.load_checkpoint('./models/','checkpoint_38.pth.tar')
    n2p = lambda x: np.argmax(n2.predict(x))
    player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

arena = Arena.Arena(n1p, player2, g, display=Game.display)

print(arena.playGames(2, verbose=True, disable_progress=True))