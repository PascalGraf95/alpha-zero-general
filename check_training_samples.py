import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from Arena import Arena
from MCTS import MCTS
import logging

import coloredlogs

from nonaga.NonagaGameManager import NonagaGameManager as GameManager
from nonaga.keras.NNet import NNetWrapper as NeuralNetwork
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'num_iterations': 1000,
    'num_episodes': 50,             # Number of complete self-play games to simulate during a new iteration.
    'random_policy_threshold': 15,  # Only play according to the policy probability distribution for the first steps,
                                    # after that play deterministically
    'update_threshold': 0.6,        # During playoff, new neural net will be accepted if threshold of games is won.
    'max_len_queue': 300000,        # Number of game examples to train the neural networks.
    'num_mcts_sims': 40,            # Number of moves for MCTS to improve the network estimation.
    'arena_matches': 12,            # Number of games to play during arena play to determine.
    'cpuct': 1,
    'warmup': 0,

    'checkpoint': './nonaga/models',
    'load_model': True,
    'load_folder_file': ('./nonaga/models/', 'pest.h5', 'checkpoint_samples_0.pth.tar'),
    'max_history_length': 20,
    'mode': 'training'

})


class Trainer:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game_manager, network, args):
        self.game_manager = game_manager
        self.player_network = network
        self.competitor_network = self.player_network.__class__(self.game_manager)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game_manager, self.player_network, self.args)
        self.training_samples_history = []
        self.skip_first_step_self_play = False
        self.current_player = 0
        self.warm_up_done = False

    def load_training_samples(self):
        sample_file = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[2])
        if not os.path.isfile(sample_file):
            log.warning(f'File "{sample_file}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(sample_file, "rb") as f:
                self.training_samples_history = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skip_first_step_self_play = True


if __name__ == '__main__':
    log.info('Loading %s...', GameManager.__name__)
    game_manager = GameManager()

    log.info('Loading %s...', NeuralNetwork.__name__)
    network = NeuralNetwork(game_manager)

    trainer = Trainer(game_manager, network, args)
    trainer.load_training_samples()

    for sample in trainer.training_samples_history[0]:
        # Board, Phase, Policy, Value
        game_manager.display_by_board(sample[0])
        game_phase = sample[1]
        policy = sample[2]
        value = sample[3]
        print("Game Phase: {:d}, Value: {:.2f}".format(game_phase, value))
        if game_phase == 0:
            policy = np.reshape(policy, (12, 15, 6))
            # non_zeros = np.unravel_index(np.where(policy != 0), (15, 12, 6))
        else:
            policy = np.reshape(policy, (12, 15))
            # non_zeros = np.unravel_index(np.where(policy != 0), (15, 12))
        nonzero_coordinates = np.nonzero(policy)
        nonzero_coordinates_list = list(zip(*nonzero_coordinates))
        print(nonzero_coordinates_list)

        print("OK")

