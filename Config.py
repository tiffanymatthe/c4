#!/usr/bin/python

class Config():
    def __init__(self) -> None:
        self.numMCTSSims = 25
        self.cpuct = 2.5 # 1

        self.numEps = 100
        self.numIters = 32
        self.maxlenOfQueue = 200000

        self.num_channels = 128
        self.num_residual_layers = 20
        self.lr = 1e-3
        self.dropout = 0.3
        self.epochs = 10
        self.batch_size = 64
        self.cuda = False
        self.tempThreshold = 15
        self.checkpoint = './temp/'

        self.arenaCompare = 40
        self.updateThreshold = 0.6
        self.load_folder_file = ('./temp', 'checkpoint_30.pth.tar')
        self.load_folder_file_examples = ('./temp', 'checkpoint_30.pth.tar')
        self.iterationNum = 31
        self.load_model = True
        self.numItersForTrainExamplesHistory = 20

        self.multiprocessing = True
        self.processes = 3

