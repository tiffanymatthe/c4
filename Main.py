#!/usr/bin/python
from Game import Game
from NeuralNet import NeuralNet
from Config import Config
from Coach import Coach

def main(config: Config):
    g = Game()
    nnet = NeuralNet(g)

    if config.load_model:
        print('Loading checkpoint "%s/%s"...', config.load_folder_file[0], config.load_folder_file[1])
        nnet.load_checkpoint(config.load_folder_file[0], config.load_folder_file[1])
    else:
        print('Not loading a checkpoint!')

    c = Coach(g, nnet, config)

    if config.load_model:
        print("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    print('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main(Config())