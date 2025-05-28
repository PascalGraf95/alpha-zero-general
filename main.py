import logging

import coloredlogs

from Trainer import Trainer, EnhancedTrainer
from nonaga.NonagaGameManager import NonagaGameManager as GameManager
from nonaga.keras.NNet import NNetWrapper as NeuralNetwork
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'training_iterations': 1000,    # Training iterations where each iteration contains num_episodes games.
    'warmup_iterations': 1,         # Training iterations where games are played purely random to fill the buffer.
    'num_episodes': 50,             # Number of complete self-play games to simulate during a new iteration.
    'random_policy_threshold': 20,  # Only play according to the policy probability distribution for the first steps,
                                    # after that play deterministically
    'inference_batch_size': 128,
    'update_threshold': 0.55,        # During playoff, new neural net will be accepted if threshold of games is won.
    'max_len_queue': 75000,        # Number of game examples to train the neural networks.
    'num_mcts_sims': 30,            # Number of moves for MCTS to improve the network estimation.
    'arena_matches': 20,            # Number of games to play during arena play to determine.
    'cpuct': 1.2,
    'num_workers': 6,
    'temperature': 1,

    'checkpoint': './nonaga/models',
    'load_model': True,
    'load_folder_file': ('./nonaga/models/', 'best.weights.h5', 'checkpoint_samples_15.pth.tar', 'best.weights.h5'),
    'max_history_length': 20,
    'mode': 'training'

})


def main():
    log.info('Loading %s...', GameManager.__name__)
    game_manager = GameManager()

    log.info('Loading %s...', NeuralNetwork.__name__)
    player_network = NeuralNetwork(game_manager)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        player_network.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Trainer...')

    if args.mode == "training":
        trainer = EnhancedTrainer(game_manager, player_network, None, args)
        if args.load_model:
            log.info("Loading 'trainExamples' from file...")
            trainer.load_training_samples()

        log.info('Starting the learning process 🎉')
        trainer.learn()

    elif args.mode == "self-play":
        opponent_network = NeuralNetwork(game_manager)
        opponent_network.load_checkpoint(args.load_folder_file[0], args.load_folder_file[3])
        trainer = EnhancedTrainer(game_manager, player_network, opponent_network, args)
        trainer.play_games()



if __name__ == "__main__":
    main()
