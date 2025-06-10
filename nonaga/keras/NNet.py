import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys

from tqdm import tqdm

sys.path.append('../..')
from utils import *
from NeuralNet import NeuralNet

import argparse

from .NonagaNet import NonagaNet as nonaganet
from ..NonagaGameManager import NonagaGameManager as GameManager

args = dotdict({
    'lr': 0.0005,
    'dropout': 0.3,
    'epochs': 15,
    'batch_size': 128,
    'num_residuals': 6,
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

        with tqdm(total=args.epochs, desc="Training Epochs Done:", leave=True) as pbar:
            for e in range(args.epochs):
                history_pi1 = self.network.pi1_model.fit(x=input_boards_phase_zero,
                                                         y=[target_pis_phase_zero, target_vs_phase_zero],
                                                         batch_size=args.batch_size, epochs=1, verbose=0)

                history_pi2 = self.network.pi2_model.fit(x=input_boards_phase_one_two,
                                           y=[target_pis_phase_one_two, target_vs_phase_one_two],
                                           batch_size=args.batch_size, epochs=1, verbose=0)
                pbar.update(1)
        log_info("Latest training losses: PI1-Loss: {:.2f}, PI2-Loss: {:.2f}, V-Loss: {:.3f}".format(
            history_pi1.history['pi1_loss'][-1], history_pi2.history['pi2_loss'][-1],
            (history_pi1.history['v_loss'][-1]+history_pi2.history['v_loss'][-1])/2))

    def predict(self, game, board):
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
        return pi[0], v[0][0]

    def predict_batch(self, boards, games, return_random=False):
        """
        Args:
            boards: list of np.array canonical boards (C, H, W)
            games: list of corresponding game instances (for phase info)
            return_random: if True, return uniform random policies and values

        Returns:
            policies: list of predicted (or random) policies
            values: list of predicted (or random) values
        """
        assert len(boards) == len(games), "Mismatch between number of boards and games"

        if return_random:
            policies = []
            values = []
            for g in games:
                action_size = g.get_action_size()
                policy = np.full(action_size, 1.0 / action_size, dtype=np.float32)
                value = np.random.uniform(-1.0, 1.0)
                policies.append(policy)
                values.append(value)
            return policies, values

        # Transpose to NHWC (batch, H, W, C)
        boards_np = np.array(boards)
        boards_np = np.transpose(boards_np, (0, 2, 3, 1))

        # Separate by phase
        idx_phase_0 = [i for i, g in enumerate(games) if g.phase == 0]
        idx_phase_1 = [i for i, g in enumerate(games) if g.phase != 0]

        policies = [None] * len(boards)
        values = [None] * len(boards)

        if idx_phase_0:
            boards_0 = boards_np[idx_phase_0]
            pi0, v0 = self.network.pi1_model.predict(boards_0, verbose=False)
            for i, pi, val in zip(idx_phase_0, pi0, v0):
                policies[i] = pi
                values[i] = val[0]

        if idx_phase_1:
            boards_1 = boards_np[idx_phase_1]
            pi1, v1 = self.network.pi2_model.predict(boards_1, verbose=False)
            for i, pi, val in zip(idx_phase_1, pi1, v1):
                policies[i] = pi
                values[i] = val[0]

        return policies, values

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # change extension
        filename = filename.split(".")[0] + ".weights.h5"

        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)
        self.network.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # change extension
        filename = filename.split(".")[0] + ".weights.h5"

        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            log_warning("No model checkpoint found!")
            return

        self.network.model.load_weights(filepath)

