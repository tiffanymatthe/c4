#!/usr/bin/python
from Game import Game
from tqdm import tqdm
from multiprocessing import Pool

def playGameAsync(player1: int, player2: int):
    """
    Executes one episode of a game.
    Returns:
        either
            winner: player who won the game (1 if player1, -1 if player2)
        or
            draw result returned from the game that is neither 1, -1, nor 0.
    """
    players = [player2, None, player1]
    curPlayer = 1
    game = Game()
    board = game.getInitBoard()
    it = 0
    while game.getWinState(board, curPlayer) == 0:
        it += 1
        action = players[curPlayer + 1](game.getCanonicalForm(board, curPlayer))
        valids = game.getValidMoves(game.getCanonicalForm(board, curPlayer))

        if valids[action] == 0:
            print(f'Action {action} is not valid!')
            print(f'valids = {valids}')
            assert valids[action] > 0
        board, curPlayer = game.getNextState(board, curPlayer, action)
    return curPlayer * game.getWinState(board, curPlayer)

class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game: Game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.
        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.
        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        while self.game.getWinState(board, curPlayer) == 0:
            it += 1
            if verbose:
                assert self.display
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)
            action = players[curPlayer + 1](self.game.getCanonicalForm(board, curPlayer))

            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer))

            if valids[action] == 0:
                print(f'Action {action} is not valid!')
                print(f'valids = {valids}')
                assert valids[action] > 0
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
        if verbose:
            assert self.display
            print("Game over: Turn ", str(it), "Result ", str(self.game.getWinState(board, 1)))
            self.display(board)
        return curPlayer * self.game.getWinState(board, curPlayer)

    def playGames(self, num, verbose=False, config=None):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.
        If config is None, no multiprocessing.
        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        if config and config.multiprocessing:
            pbar = tqdm(total=num, desc="Arena.playGames (1)", position=0, leave=True)

            def update(result):
                if result == 1:
                    nonlocal oneWon
                    oneWon += 1
                elif result == -1:
                    nonlocal twoWon
                    twoWon += 1
                else:
                    nonlocal draws
                    draws += 1
                pbar.update()

            pool = Pool(config.processes)
            for _ in range(num):
                pool.apply_async(playGameAsync, args=(self.player1, self.player2,), callback=update)
            pool.close()
            pool.join()
        else:
            for _ in tqdm(range(num), desc="Arena.playGames (1)"):
                gameResult = self.playGame(verbose=verbose)
                if gameResult == 1:
                    oneWon += 1
                elif gameResult == -1:
                    twoWon += 1
                else:
                    draws += 1

        self.player1, self.player2 = self.player2, self.player1

        if config and config.multiprocessing:
            pbar = tqdm(total=num, desc="Arena.playGames (2)", position=0, leave=True)

            def update(result):
                if result == -1:
                    nonlocal oneWon
                    oneWon += 1
                elif result == 1:
                    nonlocal twoWon
                    twoWon += 1
                else:
                    nonlocal draws
                    draws += 1
                pbar.update()

            pool = Pool(config.processes)
            for _ in range(num):
                pool.apply_async(self.playGame, args=(verbose,), callback=update)
            pool.close()
            pool.join()
        else:
            for _ in tqdm(range(num), desc="Arena.playGames (2)"):
                gameResult = self.playGame(verbose=verbose)
                if gameResult == -1:
                    oneWon += 1
                elif gameResult == 1:
                    twoWon += 1
                else:
                    draws += 1

        return oneWon, twoWon, draws