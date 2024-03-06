import logging
import math

import numpy as np
# import ray

from nonaga.NonagaGameManager import NonagaGameManager as GameManager
from nonaga.keras.NNet import NNetWrapper as Network

EPS = 1e-8

log = logging.getLogger(__name__)




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

    def get_action_probabilities(self, game, player, random_policy_actions=1):
        # Perform x MCTS simulations from the current state
        for i in range(self.args.num_mcts_sims):
            self.search(game, player, player)

        # Get the count of how often which action has been performed for each available action in the current state.
        s = self.game_manager.get_string_representation(game, self.game_manager.get_canonical_form(game, player))
        action_counts = [self.state_action_visits[(s, a)] if (s, a) in self.state_action_visits else 0 for a in
                         range(self.game_manager.get_action_size(game))]

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

    def search(self, game, player, original_player, recurrence_depth=0):
        canonical_board = self.game_manager.get_canonical_form(game, player)
        s = self.game_manager.get_string_representation(game, canonical_board)

        # region Terminal State Check
        if s not in self.game_ended_states:
            self.game_ended_states[s] = self.game_manager.has_game_ended(game, player)

        # If the game is in a terminal state we end the search and return the value with respect to the current player.
        if self.game_ended_states[s] != 0:
            return self.game_ended_states[s] if player == original_player else -self.game_ended_states[s]
        # endregion

        # region Unvisited State
        if recurrence_depth >= 60:
            self.policy_s[s], value = self.player_network.predict(game, canonical_board)
            print("Max Depth Reached")
            return value

        # Check if policy and value have not been calculated already
        if s not in self.policy_s:
            # Leaf Node
            self.policy_s[s], value = self.player_network.predict(game, canonical_board)
            valid_moves = self.game_manager.get_valid_moves(game, player)
            # print("There are {} valid moves for player {}".format(len(np.nonzero(valid_moves)[0]), player))
            # Mask all moves that are not valid in the current state
            self.policy_s[s] = self.policy_s[s] * valid_moves

            # Normalize the probabilities over all valid actions in the current state to sum to 1.
            summed_action_probabilities = np.sum(self.policy_s[s])
            if summed_action_probabilities > 0:
                self.policy_s[s] /= summed_action_probabilities
            else:
                # In the case that the probability of all valid actions is zero, do a workaround. This should
                # not happen frequently
                log.error("All valid moves had zero probability, performing a workaround.")
                self.policy_s[s] = self.policy_s[s] + valid_moves
                self.policy_s[s] /= np.sum(self.policy_s[s])

            self.valid_moves_in_states[s] = valid_moves
            self.state_visits[s] = 0
            return value if original_player == player else -value
        # endregion

        # region Already Visited State
        valid_moves = self.valid_moves_in_states[s]
        current_best = -float('inf')
        best_action = -1

        # For each action calculate the upper confidence bound
        for a in range(self.game_manager.get_action_size(game)):
            # Only of the action is valid in the current state calculate the UCB
            if valid_moves[a]:
                if (s, a) in self.action_values:
                    # UCB = action value + c * policy_probability * sqrt(state_visits) / (1+ state_action_visits)
                    ucb = self.action_values[(s, a)] + self.args.cpuct * self.policy_s[s][a] \
                          * math.sqrt(self.state_visits[s]) / (1 + self.state_action_visits[(s, a)])
                else:
                    ucb = self.args.cpuct * self.policy_s[s][a] * math.sqrt(self.state_visits[s] + EPS)
                if ucb > current_best:
                    current_best = ucb
                    best_action = a

        next_game, next_player = self.game_manager.get_next_state(game, player, best_action)

        # From the next state the function calls itself recursively until the leaf node is found
        value = self.search(next_game, next_player, player)

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

