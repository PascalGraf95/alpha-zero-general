import logging

import coloredlogs

from Trainer import Trainer
from nonaga.NonagaGameManager import NonagaGameManager as GameManager
from nonaga.keras.NNet import NNetWrapper as NeuralNetwork
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'num_iterations': 1000,
    'num_episodes': 60,             # Number of complete self-play games to simulate during a new iteration.
    'random_policy_threshold': 50,  # Only play according to the policy probability distribution for the first steps,
                                    # after that play deterministically
    'update_threshold': 0.51,        # During playoff, new neural net will be accepted if threshold of games is won.
    'max_len_queue': 300000,        # Number of game examples to train the neural networks.
    'num_mcts_sims': 40,            # Number of moves for MCTS to improve the network estimation.
    'arena_matches': 24,            # Number of games to play during arena play to determine.
    'cpuct': 4,

    'checkpoint': './nonaga/models',
    'load_model': True,
    'load_folder_file': ('./nonaga/models/', 'best.h5', 'checkpoint_samples_7.pth.tar'),
    'max_history_length': 20,
    'mode': 'self-play'

})


def main():
    log.info('Loading %s...', GameManager.__name__)
    game_manager = GameManager()

    log.info('Loading %s...', NeuralNetwork.__name__)
    network = NeuralNetwork(game_manager)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        network.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Trainer...')
    trainer = Trainer(game_manager, network, args)

    if args.mode == "training":
        if args.load_model:
            log.info("Loading 'trainExamples' from file...")
            trainer.load_training_samples()

        log.info('Starting the learning process 🎉')
        trainer.learn()

    elif args.mode == "self-play":
        trainer.play_games()



if __name__ == "__main__":
    main()
