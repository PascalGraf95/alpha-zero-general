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
    'batch_size': 128,
    'cuda': True,
    'num_channels': 64,
})


class NNetWrapper:
    def __init__(self, game_manager: GameManager):
        self.network = nonaganet(game_manager, args)
        self.board_x, self.board_y = game_manager.get_board_size(game_manager.reset_board())

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, phases, target_pis, target_values = list(zip(*examples))
        phases = np.asarray(phases)
        phase_zero_indices = np.where(phases == 0)[0].astype(int)
        phase_one_two_indices = np.where((phases == 1) | (phases == 2))[0].astype(int)

        input_boards_phase_zero = []
        target_pis_phase_zero = []
        target_vs_phase_zero = []

        for idx in phase_zero_indices:
            input_boards_phase_zero.append(input_boards[idx])
            target_pis_phase_zero.append(target_pis[idx])
            target_vs_phase_zero.append(target_values[idx])

        input_boards_phase_zero = np.transpose(np.asarray(input_boards_phase_zero), (0, 2, 3, 1))
        target_pis_phase_zero = np.asarray(target_pis_phase_zero)
        target_vs_phase_zero = np.expand_dims(np.asarray(target_vs_phase_zero), axis=1)

        input_boards_phase_one_two = []
        target_pis_phase_one_two = []
        target_vs_phase_one_two = []

        for idx in phase_one_two_indices:
            input_boards_phase_one_two.append(input_boards[idx])
            target_pis_phase_one_two.append(target_pis[idx])
            target_vs_phase_one_two.append(target_values[idx])

        input_boards_phase_one_two = np.transpose(np.asarray(input_boards_phase_one_two), (0, 2, 3, 1))
        target_pis_phase_one_two = np.asarray(target_pis_phase_one_two)
        target_vs_phase_one_two = np.expand_dims(np.asarray(target_vs_phase_one_two), axis=1)

        for e in range(args.epochs):
            self.network.pi1_model.fit(x=input_boards_phase_zero, y=[target_pis_phase_zero, target_vs_phase_zero],
                                       batch_size=args.batch_size, epochs=1, verbose=1)

            self.network.pi2_model.fit(x=input_boards_phase_one_two,
                                       y=[target_pis_phase_one_two, target_vs_phase_one_two],
                                       batch_size=args.batch_size, epochs=1, verbose=1)

    def predict(self, game, board, dummy_values=-10):
        """
        board: np array with board
        """
        # preparing input
        board = np.expand_dims(board, axis=0)
        board = np.transpose(board, (0, 2, 3, 1))
        # run
        if game.phase == 0:
            pi, v = self.network.pi1_model.predict(board, verbose=False)
        else:
            pi, v = self.network.pi2_model.predict(board, verbose=False)
        # if dummy_values != -10:
        #    return pi[0], [dummy_values]
        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # change extension
        filename = filename.split(".")[0] + ".h5"

        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)
        self.network.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # change extension
        filename = filename.split(".")[0] + ".h5"

        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            print("No model in path {}".format(filepath))
            return

        self.network.model.load_weights(filepath)

