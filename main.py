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
    'num_episodes': 100,             # Number of complete self-play games to simulate during a new iteration.
    'random_policy_threshold': 20,  # Only play according to the policy probability distribution for the first steps,
                                    # after that play deterministically
    'inference_batch_size': 64,
    'update_threshold': 0.55,       # During playoff, new neural net will be accepted if threshold of games is won.
    'max_len_queue': 75000,         # Number of game examples to train the neural networks.
    'num_mcts_sims': 200,            # Number of moves for MCTS to improve the network estimation.
    'arena_matches': 30,            # Number of games to play during arena play to determine.
    'cpuct': 2,
    'num_workers': 30,
    'temperature': 1,

    'checkpoint': './nonaga/models',
    'load_model': True,
    'load_folder_file': ('./nonaga/models/', 'bestest.weights.h5', 'latest_checkpoint.pth.tar', 'best.weights.h5'),
    'max_history_length': 15,
    'mode': 'training'

})


def main():
    log_info("Starting Initialization...")
    game_manager = GameManager()
    player_network = NeuralNetwork(game_manager)

    if args.load_model:
        log_info('Loading Model Checkpoint from "{}/{}"...'.format(args.load_folder_file[0],
                                                                   args.load_folder_file[1]))
        player_network.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log_warning('Not loading a model. Starting from scratch.')



    if args.mode == "training":
        trainer = EnhancedTrainer(game_manager, player_network, None, args)
        if args.load_model:
            log_info(
                'Loading training samples from "{}/{}"...'.format(args.load_folder_file[0],
                                                                  args.load_folder_file[2]))
            trainer.load_training_samples()
        log_success("Initialization Successful! Starting Training...")
        trainer.learn()

    elif args.mode == "self-play":
        log_success("Initialization Successful! Starting Self-Play...")
        opponent_network = NeuralNetwork(game_manager)
        opponent_network.load_checkpoint(args.load_folder_file[0], args.load_folder_file[3])
        trainer = EnhancedTrainer(game_manager, player_network, opponent_network, args)
        trainer.play_games()



if __name__ == "__main__":
    main()
