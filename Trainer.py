import logging
import os
import queue
import sys
import time
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from Arena import Arena
from MCTS import MCTS, RandomAgent, Worker

from nonaga.NonagaGameManager import NonagaGameManager as GameManager
from nonaga.keras import NNet as Network
from nonaga.NonagaLogic import Game
import multiprocessing as mp
import tensorflow as tf

# import ray

log = logging.getLogger(__name__)

class EnhancedTrainer:
    def __init__(self, game_manager: GameManager, network: Network, args):
        self.global_sample_queue = None
        self.result_queues = None
        self.global_inference_queue = None
        self.stop_agents_event = None
        self.workers = None
        self.game_manager = game_manager
        self.player_network = network
        self.competitor_network = self.player_network.__class__(self.game_manager)  # the competitor network
        self.args = args
        self.training_samples_history = []
        self.skip_first_step_self_play = False
        self.current_player = 0
        self.current_training_iteration = 0
        self.warmup = True
        self.stop_training_event = mp.Event()
        self.episodes_done_queue = mp.Queue()


    def learn(self):
        # The total amount of training iterations (each consisting of multiple episodes played)
        while self.current_training_iteration <= self.args.training_iterations:
            log.info('------Training Iteration {:03d}------'.format(self.current_training_iteration))

            if not self.skip_first_step_self_play:
                self.skip_first_step_self_play = False

                if self.current_training_iteration <= self.args.warmup_iterations:
                    log.info('Warmup Mode: Playing with random values and policies')
                    self.warmup = True
                elif self.warmup:
                    self.warmup = False

                # 1. Generate and Start Workers/Agents each containing its own MCTSs
                self.generate_workers()

                # 2. Start Processing Leaf Nodes in parallel from the Queue that is filled by the workers
                self.process_inference_queue()

                # 3. Wait until enough episodes have been played
                episodes_done = 0
                while episodes_done < self.args.num_episodes:
                    try:
                        self.episodes_done_queue.get(timeout=5)
                        episodes_done += 1
                    except queue.Empty:
                        continue

                self.stop_agents_event.set()
                for worker in self.workers:
                    worker.kill()

            # Once enough episodes per iteration have been played, start the actual training function.
            training_samples = self.collect_samples_from_queue(self.args.samples_per_iteration)
            self.training_samples_history.append(training_samples)

            # region Sample History
            # Ring buffer sample history
            if len(self.training_samples_history) > self.args.max_history_length:
                log.warning(
                    f"Removing the oldest entry in trainExamples. "
                    f"len(trainExamplesHistory) = {len(self.training_samples_history)}")
                self.training_samples_history.pop(0)
            # Backup history to a file
            self.save_training_samples(self.current_training_iteration - 1)
            # endregion

            training_batch = []
            for e in self.training_samples_history:
                training_batch.extend(e)
            shuffle(training_batch)

            # Store model before training and load it for the competitor
            self.player_network.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.competitor_network.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

            # Perform the actual network training
            self.player_network.train(training_batch)
            """
            # region Arena
            # Evaluate the latest network performance in an arena
            log.info('STARTING ARENA MATCHES')
            log.info('----------------------')

            log.info('PITTING AGAINST PREVIOUS VERSION')

            new_wins, old_wins, draws = arena.play_games(self.args.arena_matches)
            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (new_wins, old_wins, draws))


            if old_wins + new_wins == 0 or float(new_wins) / (old_wins + new_wins) < self.args.update_threshold:
                log.info('REJECTING NEW MODEL')
                self.player_network.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.player_network.save_checkpoint(folder=self.args.checkpoint,
                                                    filename=self.get_checkpoint_file("", i))
                self.player_network.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
            # endregion
            """

            self.current_training_iteration += 1

    def process_inference_queue(self):
        log.info("Starting inference processing loop...")
        batch_size = self.args.inference_batch_size
        wait_time = 1  # seconds to wait between checks when queue is empty

        while True:
            batch_boards = []
            batch_games = []
            metadata = []

            start_time = time.time()
            while len(batch_boards) < batch_size and (time.time() - start_time) < 15:
                try:
                    item = self.global_inference_queue.get(timeout=wait_time)

                    game, current_player, worker_id, request_id = item
                    canonical_board = self.game_manager.get_canonical_form(game, current_player)

                    batch_boards.append(canonical_board)
                    batch_games.append(game)
                    metadata.append((worker_id, request_id))

                except queue.Empty:
                    continue

            if not batch_boards:
                continue

            policies, values = self.player_network.predict_batch(batch_boards, batch_games)

            # Send results back to workers
            for (worker_id, request_id), policy, value in zip(metadata, policies, values):
                self.result_queues[worker_id].put((request_id, policy, value))

    def generate_workers(self):
        self.stop_agents_event = mp.Event()

        # Global inference queue shared between all workers and the inference loop
        self.global_inference_queue = mp.Queue()

        # Global sample queue collecting samples for training the neural network
        self.global_sample_queue = mp.Queue()

        # List to track each worker's response queue
        self.result_queues = []

        self.workers = []

        for i in range(self.args.num_workers):
            # Each worker gets its own result queue
            result_queue = mp.Queue()
            self.result_queues.append(result_queue)

            # Create and launch the worker
            agent = Worker(
                worker_id=i,
                global_inference_queue=self.global_inference_queue,
                global_sample_queue=self.global_sample_queue,
                episodes_done_queue=self.episodes_done_queue,
                result_queue=result_queue,
                stop_event=self.stop_agents_event,
                is_warmup=self.warmup,
                args=self.args
            )
            self.workers.append(agent)

            agent.daemon = True
            agent.start()

    def collect_samples_from_queue(self, num_samples):
        collected = []
        for _ in range(num_samples):
            try:
                sample = self.global_sample_queue.get(timeout=10)  # Wait up to 10s
                collected.append(sample)
            except queue.Empty:
                log.warning("Sample queue empty before reaching desired sample count.")
                break
        return collected

    def save_training_samples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.get_checkpoint_file("samples_", iteration))
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.training_samples_history)

    def load_training_samples(self):
        # model_file = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        sample_file = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[2])
        if not os.path.isfile(sample_file):
            log.warning(f'File "{sample_file}" with trainExamples not found!')
            return
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(sample_file, "rb") as f:
                self.training_samples_history = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skip_first_step_self_play = True

    @staticmethod
    def get_checkpoint_file(name, iteration):
        return 'checkpoint_' + name + str(iteration) + '.pth.tar'

class Trainer:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game_manager: GameManager, network: Network, args):
        self.game_manager = game_manager
        self.player_network = network
        self.competitor_network = self.player_network.__class__(self.game_manager)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game_manager, self.player_network, self.args)
        self.training_samples_history = []
        self.skip_first_step_self_play = False
        self.current_player = 0

    def play_games(self, stochastic_policy = False):
        game = self.game_manager.reset_board()
        self.current_player = 1
        episode_step = 0

        # Play the full episode until game has ended
        while True:
            episode_step += 1
            if game.phase == 0:
                self.game_manager.draw_board_cv2(game, self.current_player, turn_number=(episode_step-1)/3, save=True)

            # Get the current policy according to the neural network and MCTS. The neural network suggests
            # the initial policy and the mcts refines it with rollouts. The number of new states to be explored
            # is limited by num_mcts_sims
            policy = self.mcts.get_action_probabilities(game, self.current_player,
                                                        random_policy_actions=1, debug=False)
            # Choose the actual action and execute
            if stochastic_policy:
                action = np.random.choice(len(policy), p=policy)
            else:
                action = np.argmax(policy)
            game, self.current_player = self.game_manager.get_next_state(game, self.current_player, action)

            winner = self.game_manager.has_game_ended(game, self.current_player)

            # If there is a winner the game has ended. Return the training samples without the current player property.
            if winner != 0:
                self.game_manager.draw_board_cv2(game, self.current_player, turn_number=(episode_step-1)/3+1, save=True)
                # input("This is how the game ended. Winner: " + str(winner) +
                #       ". Do you want to continue? (Press Enter)")
                game = self.game_manager.reset_board()
                self.mcts = MCTS(self.game_manager, self.player_network, self.args)
                self.current_player = 1
                episode_step += 1

    def execute_episode(self, random=False):
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
        game = self.game_manager.reset_board(scenario=0)
        self.current_player = np.random.choice([-1, 1])
        episode_step = 0

        # Play the full episode until game has ended
        while True:
            if episode_step > 320:
                return []

            episode_step += 1
            # Later in the tree search action should be more deterministic to end the episode
            random_policy_actions = int(episode_step < self.args.random_policy_threshold)
            # random_policy_actions = 1

            # Get the current policy according to the neural network and MCTS. The neural network suggests
            # the initial policy and the mcts refines it with rollouts. The number of new states to be explored
            # is limited by num_mcts_sims
            policy = self.mcts.get_action_probabilities(game, self.current_player,
                                                        random_policy_actions=random_policy_actions)

            # region Symmetries
            # Add all symmetrical boards to the training samples as they are identical in policy
            symmetries = self.game_manager.get_symmetries(game, self.current_player, np.copy(policy))
            for b, p in symmetries:
                # Training Sample: Board Configuration, Current Player, Policy, Phase, Value (which is unknown yet)
                training_samples.append([b, self.current_player, game.phase, p, None])
            # endregion

            # Choose the actual action and execute
            action = np.random.choice(len(policy), p=policy)
            game, self.current_player = self.game_manager.get_next_state(game, self.current_player, action)

            winner = self.game_manager.has_game_ended(game, self.current_player)

            # If there is a winner the game has end. Return the training samples without the current player property.
            if winner != 0:
                # self.game_manager.display(game)
                # print("This is how the game ended. "
                #       "Its value is: {} for player {}".format(winner, self.current_player))
                # Board, Phase, Policy, Value
                actual_samples = [(sample[0], sample[2], sample[3], winner * (-1) ** (sample[1] != self.current_player))
                                  for sample in training_samples]

                return actual_samples

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.training_iterations + 1):
            log.info(f'Starting Training Iteration #{i} ...')

            # region Self-Play
            # In the first training iteration play a number of self-play iterations
            if not self.skip_first_step_self_play or i > 1:
                training_samples = deque([], maxlen=self.args.max_len_queue)

                # Do x episodes of self-play to fill the buffer
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

            # Store model before training and load it for the competitor
            self.player_network.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.competitor_network.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            competitor_mcts = MCTS(self.game_manager, self.competitor_network, self.args)

            # Then perform the actual training process
            self.player_network.train(training_batch)
            player_mcts = MCTS(self.game_manager, self.player_network, self.args)

            # Instantiate the pure MCTS agent
            pure_mcts = MCTS(self.game_manager, None, self.args)

            # Instantiate random agent
            random_agent = RandomAgent(self.game_manager, None, self.args)
            # endregion

            # region Arena Playoff
            log.info('STARTING ARENA MATCHES')
            log.info('----------------------')

            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(player_mcts, competitor_mcts, game_manager=self.game_manager, args=self.args)
            new_wins, old_wins, draws = arena.play_games(self.args.arena_matches)
            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (new_wins, old_wins, draws))
            """
            log.info('PITTING AGAINST MCTS PLAYER')
            arena = Arena(player_mcts, pure_mcts, game_manager=self.game_manager, args=self.args)
            pvm_wins, pvm_losses, pvm_draws = arena.play_games(self.args.arena_matches)
            log.info('PLAYER/MCTS WINS : %d / %d ; DRAWS : %d' % (pvm_wins, pvm_losses, pvm_draws))

            log.info('PITTING AGAINST RANDOM PLAYER')
            arena = Arena(player_mcts, random_agent, game_manager=self.game_manager, args=self.args)
            pvr_wins, pvr_losses, pvr_draws = arena.play_games(self.args.arena_matches)
            log.info('PLAYER/RANDOM WINS : %d / %d ; DRAWS : %d' % (pvr_wins, pvr_losses, pvr_draws))
            """

            if old_wins + new_wins == 0 or float(new_wins) / (old_wins + new_wins) < self.args.update_threshold:
                log.info('REJECTING NEW MODEL')
                self.player_network.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.player_network.save_checkpoint(folder=self.args.checkpoint, filename=self.get_checkpoint_file("", i))
                self.player_network.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
            # endregion


    @staticmethod
    def get_checkpoint_file(name, iteration):
        return 'checkpoint_' + name + str(iteration) + '.pth.tar'

    def save_training_samples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.get_checkpoint_file("samples_", iteration))
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.training_samples_history)

    def load_training_samples(self):
        # model_file = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        sample_file = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[2])
        if not os.path.isfile(sample_file):
            log.warning(f'File "{sample_file}" with trainExamples not found!')
            return
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(sample_file, "rb") as f:
                self.training_samples_history = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skip_first_step_self_play = True
