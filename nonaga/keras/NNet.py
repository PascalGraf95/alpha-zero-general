import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys
sys.path.append('../..')
from utils import *
from NeuralNet import NeuralNet

import argparse

from .NonagaNet import NonagaNet as nonaganet
from ..NonagaGameManager import NonagaGameManager as GameManager

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': True,
    'num_channels': 128,
})


class NNetWrapper:
    def __init__(self, game_manager: GameManager):
        self.network = nonaganet(game_manager, args)
        self.board_x, self.board_y = game_manager.get_board_size(game_manager.reset_board())

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_values, phases = list(zip(*examples))

        phases = np.asarray(phases)
        phase_zero_indices = np.where(phases == 0)[0]
        phase_one_two_indices = np.where(phases == 1 or phases == 2)[0]

        input_boards_phase_zero = np.asarray(input_boards[phase_zero_indices])
        target_pis_phase_zero = np.asarray(target_pis[phase_zero_indices])
        target_vs_phase_zero = np.asarray(target_values[phase_zero_indices])

        input_boards_phase_one_two = np.asarray(input_boards[phase_one_two_indices])
        target_pis_phase_one_two = np.asarray(target_pis[phase_one_two_indices])
        target_vs_phase_one_two = np.asarray(target_values[phase_one_two_indices])

        for e in range(args.epochs):
            """
            self.network.pi1_model.train_on_batch(x=input_boards_phase_zero,
                                                  y=[target_pis_phase_zero, target_vs_phase_zero],
                                                  batch_size=args.batch_size)
            self.network.pi2_model.train_on_batch(x=input_boards_phase_one_two,
                                                  y=[target_pis_phase_one_two, target_vs_phase_one_two],
                                                  batch_size=args.batch_size)
            """

            self.network.pi1_model.fit(x=input_boards_phase_zero, y=[target_pis_phase_zero, target_vs_phase_zero],
                                       batch_size=args.batch_size, epochs=1, verbose=0)

            self.network.pi2_model.fit(x=input_boards_phase_one_two,
                                       y=[target_pis_phase_one_two, target_vs_phase_one_two],
                                       batch_size=args.batch_size, epochs=1, verbose=0)

    def predict(self, game, board):
        """
        board: np array with board
        """
        # preparing input
        board = np.expand_dims(board, axis=0)
        # board = np.transpose(board, (0, 2, 3, 1))
        # run
        if game.phase == 0:
            pi, v = self.network.pi1_model.predict(board, verbose=False)
        else:
            pi, v = self.network.pi2_model.predict(board, verbose=False)
        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # change extension
        filename = filename.split(".")[0] + ".h5"

        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.network.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # change extension
        filename = filename.split(".")[0] + ".h5"

        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))

        self.network.model.load_weights(filepath)

