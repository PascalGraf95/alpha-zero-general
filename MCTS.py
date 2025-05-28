import logging
import math
import random

import numpy as np
# import ray
import multiprocessing as mp

from nonaga.NonagaLogic import Game
from nonaga.NonagaGameManager import NonagaGameManager as GameManager
from nonaga.keras.NNet import NNetWrapper as Network

EPS = 1e-8

log = logging.getLogger(__name__)


class Node:
    def __init__(self, game: Game, current_player, legal_moves_fn, parent=None, prior=0.0):
        self.game = game
        _, _, self.legal_moves = legal_moves_fn(game, current_player)
        self.parent = parent
        self.prior = prior
        self.current_player = current_player  # Store the player to move at this node

        self.children = {}  # action -> Node
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False

    @property
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, action_priors, next_states_fn, legal_moves_fn):
        self.is_expanded = True
        for action, prior in action_priors.items():
            next_game, next_player = next_states_fn(self.game, self.current_player, action)
            self.children[action] = Node(next_game, next_player, legal_moves_fn, parent=self, prior=prior)

    def select_child(self, cpuct=1.0):
        best_score = -float("inf")
        best_action = None
        best_child = None

        total_visits = sum(child.visit_count for child in self.children.values()) + 1e-8

        for action, child in self.children.items():
            ucb_score = child.ucb_score(total_visits, cpuct)
            if ucb_score > best_score:
                best_score = ucb_score
                best_action = action
                best_child = child

        return best_action, best_child

    def ucb_score(self, total_visits, cpuct):
        u = cpuct * self.prior * (np.sqrt(total_visits) / (1 + self.visit_count))
        return self.value + u

    def backpropagate(self, value, from_player):
        self.visit_count += 1
        # Flip value if the perspective differs
        if self.current_player != from_player:
            value = -value
        self.value_sum += value
        if self.parent:
            self.parent.backpropagate(value, from_player)


class EnhancedMCTS:
    def __init__(self, worker_id, inference_queue, result_queue, mcts_owner, args):
        super().__init__()
        self.inference_queue = inference_queue  # Shared global queue for sending inference requests
        self.result_queue = result_queue        # Dedicated queue for receiving results
        self.worker_id = worker_id
        self.args = args
        self.mcts_owner = mcts_owner

    def search(self, root_node: Node, next_states_fn, next_legal_moves_fn, num_simulations, cpuct=1.0):
        leaves = []
        search_paths = []

        for _ in range(num_simulations):
            node = root_node
            path = [node]

            # region SELECTION
            while node.is_expanded and node.children:
                _, node = node.select_child(cpuct)
                path.append(node)

            if result := node.game.check_for_game_end(node.current_player):
                path[-1].backpropagate(result, node.current_player)
            else:
                leaves.append(node)
                search_paths.append(path)
            # endregion

        # INFERENCE REQUEST
        if leaves:
            # Send leaf states to inference process
            for node in leaves:
                self.inference_queue.put([node.game, node.current_player, self.worker_id, self.mcts_owner])

            # Collect inference results
            policies, values = [], []

            for _ in range(len(leaves)):
                policy, value = self.result_queue.get()
                policies.append(policy)
                values.append(value)

            # region EXPANSION + BACKPROP
            for node, policy, value, path in zip(leaves, policies, values, search_paths):
                action_priors = {a: policy[a] for a in node.legal_moves}
                node.expand(action_priors, next_states_fn, next_legal_moves_fn)

                # Backpropagate the value from the neural net
                path[-1].backpropagate(value, node.current_player)
            # endregion
        return {action: (child.visit_count, child.value) for action, child in root_node.children.items()}


class Worker(mp.Process):
    def __init__(self, worker_id, global_inference_queue, global_sample_queue, episodes_done_queue,
                 result_queue_player, stop_event, is_training, args, result_queue_opponent=None):
        super().__init__()
        self.current_player = None
        self.worker_id = worker_id
        self.game_manager = GameManager()
        self.global_inference_queue = global_inference_queue
        self.global_sample_queue = global_sample_queue
        self.episodes_done_queue = episodes_done_queue
        self.result_queue_player = result_queue_player
        self.result_queue_opponent = result_queue_opponent
        self.stop_event = stop_event
        self.is_training = is_training
        self.args = args

        # Instantiate MCTS with access to shared queues
        self.mcts_player = EnhancedMCTS(self.worker_id, self.global_inference_queue,
                                        self.result_queue_player, "player", self.args)
        if not self.is_training:
            self.mcts_opponent = EnhancedMCTS(self.worker_id, self.global_inference_queue,
                                              self.result_queue_opponent, "opponent", self.args)
        else:
            self.mcts_opponent = None

    def run(self):
        while not self.stop_event.is_set():
            game = self.game_manager.reset_board()
            self.current_player = np.random.choice([-1, 1])
            episode_step = 0
            training_samples = []

            if self.args.mode == "self-play":
                self.game_manager.draw_board_cv2(game, self.current_player, turn_number=0, save=True)

            while not self.game_manager.has_game_ended(game, self.current_player):
                root_node = Node(game, self.current_player, self.game_manager.get_valid_moves)
                if self.current_player == 1 or self.is_training:
                    search_stats = self.mcts_player.search(root_node, self.game_manager.get_next_state,
                                                           self.game_manager.get_valid_moves,
                                                           self.args.num_mcts_sims, self.args.cpuct)
                else:
                    search_stats = self.mcts_opponent.search(root_node, self.game_manager.get_next_state,
                                                           self.game_manager.get_valid_moves,
                                                           self.args.num_mcts_sims, self.args.cpuct)

                if episode_step < self.args.random_policy_threshold and self.is_training:
                    temperature = self.args.temperature
                else:
                    temperature = 0

                _, _, legal_moves = self.game_manager.get_valid_moves(game, self.current_player)
                action, policy = self.select_action(search_stats, self.game_manager.get_action_size(game), legal_moves,
                                                    temperature=temperature)

                if self.is_training:
                    # region Symmetries
                    # Add all symmetrical boards to the training samples as they are identical in policy
                    symmetries = self.game_manager.get_symmetries(game, self.current_player, np.copy(policy))
                    for board_sym, policy_sym in symmetries:
                        # Training Sample: Board Configuration, Current Player, Policy, Phase, Value (which is unknown yet)
                        # training_samples.append([board_sym, self.current_player, game.phase, policy_sym, None])
                        training_samples.append([board_sym, game.phase, policy_sym, None, self.current_player])
                    # endregion

                game, self.current_player = self.game_manager.get_next_state(game, self.current_player, action)
                episode_step += 1

                if self.args.mode == "self-play" and game.phase == 0:
                    self.game_manager.draw_board_cv2(game, self.current_player, turn_number=int(episode_step/3), save=True)


                winner = self.game_manager.has_game_ended(game, self.current_player)

                # If there is a winner the game has end. Return the training samples without the current player property.
                if winner != 0 or episode_step > 300:
                    # actual_samples = []
                    # Board, Phase, Policy, Value
                    for board, phase, policy, _, player in training_samples:
                        value = winner if player == self.current_player else -winner
                        sample = (board, phase, policy, value)
                        self.global_sample_queue.put(sample)
                    # if not self.is_training:
                    # print(f"WINNER: {winner}, Current Player: {self.current_player}")
                    self.episodes_done_queue.put(self.current_player)
                    # self.game_manager.draw_board_cv2(game, self.current_player, turn_number=episode_step % 3 + 1)
                    break

    def select_action(self, stats, action_size, legal_moves, temperature=1.0):
        """
        Selects an action and returns π as a dense array over all actions, masked by legal ones.

        Args:
            stats: dict of {action: (visit_count, value)}
            action_size: total number of possible actions
            legal_moves: list of legal action indices
            temperature: exploration factor

        Returns:
            selected_action: int
            pi: list of floats of size `action_size`
        """
        visit_counts = np.zeros(action_size, dtype=np.float32)

        # Fill visit counts only for explored children (i.e., legal moves with visits)
        for a, (count, _) in stats.items():
            visit_counts[a] = count

        if temperature == 0:
            # Pick among best visited legal actions
            legal_visits = [(a, visit_counts[a]) for a in legal_moves]
            max_visit = max(v for _, v in legal_visits)
            best_actions = [a for a, v in legal_visits if v == max_visit]
            selected_action = np.random.choice(best_actions)

            pi = np.zeros(action_size, dtype=np.float32)
            pi[selected_action] = 1.0
            return selected_action, pi.tolist()

        # Softmax with temperature over legal moves
        adjusted = visit_counts ** (1.0 / temperature)
        masked = np.zeros_like(adjusted)

        for a in legal_moves:
            masked[a] = adjusted[a]

        sum_masked = np.sum(masked)
        if sum_masked == 0:
            # Fallback: uniform over legal moves
            for a in legal_moves:
                masked[a] = 1.0
            masked /= np.sum(masked)
        else:
            masked /= sum_masked

        selected_action = np.random.choice(np.arange(action_size), p=masked)
        return selected_action, masked.tolist()


class MCTS:
    def __init__(self, game_manager: GameManager, network: Network, args):
        self.game_manager = game_manager
        self.player_network = network
        self.args = args

        self.action_values = {}  # stores Q values for s,a (as defined in the paper)
        self.state_action_visits = {}  # stores #times edge s,a was visited
        self.state_visits = {}  # stores #times board s was visited
        self.policy_s = {}  # stores initial policy (returned by neural net)

        self.game_ended_states = {}  # stores game.getGameEnded ended for board s
        self.valid_moves_in_states = {}  # stores game.getValidMoves for board

    def reset(self):
        self.action_values = {}  # stores Q values for s,a (as defined in the paper)
        self.state_action_visits = {}  # stores #times edge s,a was visited
        self.state_visits = {}  # stores #times board s was visited
        self.policy_s = {}  # stores initial policy (returned by neural net)

        self.game_ended_states = {}  # stores game.getGameEnded ended for board s
        self.valid_moves_in_states = {}  # stores game.getValidMoves for board

    def get_action_probabilities(self, game, player, random_policy_actions=1, debug=False):
        # Perform x MCTS simulations from the current state
        for i in range(self.args.num_mcts_sims):
            self.search(game, player, player, debug=debug)

        # Get the count of how often which action has been performed for each available action in the current state.
        s = self.game_manager.get_string_representation(game, self.game_manager.get_canonical_form(game, player))
        action_counts = [self.state_action_visits[(s, a)] if (s, a) in self.state_action_visits else 0 for a in
                         range(self.game_manager.get_action_size(game))]

        action_count_indices = np.nonzero(action_counts)[0]
        action_count_dict = {}
        for a in action_count_indices:
            action_count_dict[a] = action_counts[a]

        # Act deterministically towards later stages of the tree search. Take one of the actions that have been chosen
        # the most.
        if random_policy_actions == 0:
            best_actions = np.array(np.argwhere(action_counts == np.max(action_counts))).flatten()
            best_action = np.random.choice(best_actions)
            action_probabilities = [0] * len(action_counts)
            action_probabilities[best_action] = 1
            return action_probabilities

        # Act randomly in the beginning of tree search.
        action_counts = [x ** (1. / random_policy_actions) for x in action_counts]
        # Sum all action counts and normalize by the total count of actions taken in that state
        action_counts_sum = float(sum(action_counts))
        action_probabilities = [x / action_counts_sum for x in action_counts]
        return action_probabilities

    def search(self, game, player, original_player, recurrence_depth=0, debug=False):
        canonical_board = self.game_manager.get_canonical_form(game, player)
        s = self.game_manager.get_string_representation(game, canonical_board)

        # region Terminal State Check
        if s not in self.game_ended_states:
            self.game_ended_states[s] = self.game_manager.has_game_ended(game, player)

        # If the game is in a terminal state we end the search and return the value with respect to the current player.
        if self.game_ended_states[s] != 0:
            return self.game_ended_states[s] if player == original_player else -self.game_ended_states[s]
        # endregion

        # region Unvisited State & Max Depth
        if recurrence_depth >= 60:
            self.policy_s[s], value = self.player_network.predict(game, canonical_board)
            print("Max Tree Depth Reached")
            return value if original_player == player else -value

        # Check if policy and value have not been calculated already
        if s not in self.policy_s:
            valid_moves_masked, _ = self.game_manager.get_valid_moves(game, player)

            if self.player_network is not None:
                # Use neural network to get policy and value
                self.policy_s[s], value = self.player_network.predict(game, canonical_board)
                self.policy_s[s] *= valid_moves_masked
                summed_action_probabilities = np.sum(self.policy_s[s])
                if summed_action_probabilities > 0:
                    self.policy_s[s] /= summed_action_probabilities
                else:
                    log.warning("All valid moves had zero probability, falling back to uniform.")
                    self.policy_s[s] = valid_moves_masked / np.sum(valid_moves_masked)
            else:
                # Fallback: uniform probabilities and zero value
                self.policy_s[s] = valid_moves_masked / np.sum(valid_moves_masked)
                value = 0

            self.valid_moves_in_states[s] = valid_moves_masked
            self.state_visits[s] = 0
            return value if original_player == player else -value
        # endregion

        # region Already Visited State
        valid_moves = self.valid_moves_in_states[s]
        if debug:
            _, legal_moves = self.game_manager.get_valid_moves(game, player)
        current_best = -float('inf')
        best_action = -1

        # For each action calculate the upper confidence bound
        valid_action_indices = np.nonzero(valid_moves)[0]
        ucb_dict = {}
        for a in valid_action_indices:
            # Only of the action is valid in the current state calculate the UCB
            if (s, a) in self.action_values:
                # UCB = action value + c * policy_probability * sqrt(state_visits) / (1+ state_action_visits)
                action_value = self.action_values[(s, a)]
                policy_value = self.policy_s[s][a]
                confidence_bonus = self.args.cpuct * math.sqrt(self.state_visits[s]) / (1 + self.state_action_visits[(s, a)])
                ucb =  action_value + (confidence_bonus * policy_value)
                ucb_dict[a] = {"action_value": action_value, "confidence_bonus": confidence_bonus, "policy_value":policy_value, "ucb": ucb}
            else:
                policy_value = self.policy_s[s][a]
                confidence_bonus = self.args.cpuct * math.sqrt(self.state_visits[s] + EPS)
                # ucb = self.args.cpuct * policy_value * math.sqrt(self.state_visits[s] + EPS)
                ucb = confidence_bonus * policy_value
                ucb_dict[a] = {"action_value": 0, "confidence_bonus": confidence_bonus, "policy_value": policy_value, "ucb": ucb}
            if ucb > current_best:
                current_best = ucb
                best_action = a

        next_game, next_player = self.game_manager.get_next_state(game, player, best_action)

        # From the next state the function calls itself recursively until the leaf node is found
        value = self.search(next_game, next_player, player, recurrence_depth=recurrence_depth+1)

        # Update the action values for the taken action
        if (s, best_action) in self.action_values:
            self.action_values[(s, best_action)] = (self.state_action_visits[(s, best_action)] *
                                                    self.action_values[(s, best_action)] + value) / \
                                                   (self.state_action_visits[(s, best_action)] + 1)
            self.state_action_visits[(s, best_action)] += 1

        else:
            self.action_values[(s, best_action)] = value
            self.state_action_visits[(s, best_action)] = 1

        self.state_visits[s] += 1
        # endregion

        return value if original_player == player else -value


class RandomAgent(MCTS):
    def __init__(self, game_manager: GameManager, network: Network, args):
        super().__init__(game_manager, network, args)

    def get_action_probabilities(self, game, player, random_policy_actions=1, debug=False):
        valid_moves_masked, _ = self.game_manager.get_valid_moves(game, player)
        random_action_probabilities = valid_moves_masked / np.sum(valid_moves_masked)
        return random_action_probabilities