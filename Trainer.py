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

log = logging.getLogger(__name__)


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

    def execute_episode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        training_samples = []
        game = self.game_manager.reset_board()
        self.current_player = 1
        episode_step = 0

        while True:
            # print("\r Episode Steps: {}, MCTS States Visited: {}".format(episode_step,
            # len(self.mcts.state_visits)), end="")
            episode_step += 1
            canonical_board = self.game_manager.get_canonical_form(game, self.current_player)

            # Later in the tree search action should be more deterministic to end the episode
            random_policy_actions = int(episode_step < self.args.random_policy_threshold)

            policy = self.mcts.get_action_probabilities(game, self.current_player,
                                                        random_policy_actions=random_policy_actions)

            # region Symmetries
            # sym = self.game.get_symmetries(canonical_board, pi)
            # for b, p in sym:
            #    training_samples.append([b, self.current_player, p, None])
            # endregion

            # Training Sample: Board Configuration, Current Player, Policy, Phase, Value (which is unknown yet)
            training_samples.append([canonical_board, self.current_player, game.phase, policy, None])

            # Choose the actual action and execute
            action = np.random.choice(len(policy), p=policy)
            game, self.current_player = self.game_manager.get_next_state(game, self.current_player, action)

            winner = self.game_manager.has_game_ended(game, self.current_player)

            # If there is a winner the game has end. Return the training samples without the current player property.
            if winner != 0:
                return [(sample[0], sample[2], sample[3], winner * (-1) ** (sample[1] != self.current_player))
                        for sample in training_samples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.num_iterations + 1):
            log.info(f'Starting Training Iteration #{i} ...')

            # region Self-Play
            # In the first training iteration play a number of self-play iterations
            if not self.skip_first_step_self_play or i > 1:
                training_samples = deque([], maxlen=self.args.max_len_queue)

                # Do x episodes of self-play to fill the
                for _ in tqdm(range(self.args.num_episodes), desc="Self Play"):
                    self.mcts = MCTS(self.game_manager, self.player_network, self.args)  # reset search tree
                    training_samples += self.execute_episode()

                # Save the iteration examples to the history
                self.training_samples_history.append(training_samples)
            # endregion

            # region Sample History
            # Ring buffer sample history
            if len(self.training_samples_history) > self.args.max_history_length:
                log.warning(
                    f"Removing the oldest entry in trainExamples. "
                    f"len(trainExamplesHistory) = {len(self.training_samples_history)}")
                self.training_samples_history.pop(0)
            # Backup history to a file
            self.save_training_samples(i - 1)
            # endregion

            # region Training
            # Shuffle examples before training
            training_batch = []
            for e in self.training_samples_history:
                training_batch.extend(e)
            shuffle(training_batch)

            # Actual Training
            self.player_network.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.competitor_network.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            competitor_mcts = MCTS(self.game_manager, self.competitor_network, self.args)

            # Actual Training
            self.player_network.train(training_batch)
            player_mcts = MCTS(self.game_manager, self.player_network, self.args)
            # endregion

            # region Arena Playoff
            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(competitor_mcts.get_action_probabilities(x, -1, random_policy_actions=0)),
                          lambda x: np.argmax(player_mcts.get_action_probabilities(x, 1, random_policy_actions=0)),
                          self.game_manager)
            pwins, nwins, draws = arena.play_games(self.args.arena_matches)

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.update_threshold:
                log.info('REJECTING NEW MODEL')
                self.player_network.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.player_network.save_checkpoint(folder=self.args.checkpoint, filename=self.get_checkpoint_file(i))
                self.player_network.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
            # endregion

    @staticmethod
    def get_checkpoint_file(iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def save_training_samples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.get_checkpoint_file(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.training_samples_history)

    def load_training_samples(self):
        model_file = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        sample_file = model_file + ".examples"
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
