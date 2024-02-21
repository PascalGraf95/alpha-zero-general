import logging
import math

import numpy as np
from nonaga.NonagaGameManager import NonagaGameManager as GameManager
from nonaga.keras.NNet import NNetWrapper as Network

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS:
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game_manager: GameManager, network: Network, args):
        self.game_manager = game_manager
        self.player_network = network
        self.args = args

        self.action_values = {}  # stores Q values for s,a (as defined in the paper)
        self.state_action_visits = {}  # stores #times edge s,a was visited
        self.state_visits = {}  # stores #times board s was visited
        self.policy_s = {}  # stores initial policy (returned by neural net)

        self.game_ended_states = {}  # stores game.getGameEnded ended for board s
        self.valid_moves_in_states = {}  # stores game.getValidMoves for board s

    def get_action_probabilities(self, game, player, random_policy_actions=1):
        """
        This function performs num_mcts_simulations simulations of MCTS starting from canonical_board and
        returns the probability for each action in the current state. The probability is defined by how often
        an action has been taken in the tree search performed at the beginning.

        Returns:
            action_probabilities: a policy vector where the probability of the ith action is
                                  proportional to state_action_visits[(s,a)]**(1./temp)
        """

        # Perform x MCTS simulations from the current state
        for i in range(self.args.num_mcts_sims):
            self.search(game, player)

        # Get the count of how often which action has been performed for each available action in the current state.
        s = self.game_manager.get_string_representation(self.game_manager.get_canonical_form(game, player))
        action_counts = [self.state_action_visits[(s, a)] if (s, a) in self.state_action_visits else 0 for a in
                         range(self.game_manager.get_action_size(game))]

        # Act deterministically towards later stages of the tree search
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

    def search(self, game, player):
        """
        This function performs one iteration of MCTS. It is recursively called
        until a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy p and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of state_visits, state_action_visits, action_values are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        canonical_board = self.game_manager.get_canonical_form(game, player)
        s = self.game_manager.get_string_representation(canonical_board)

        # region Terminal State Check
        if s not in self.game_ended_states:
            self.game_ended_states[s] = self.game_manager.has_game_ended(game, player)
        if self.game_ended_states[s] != 0:
            return self.game_ended_states[s]
        # endregion

        # region Unvisited State
        # Check if policy and value have not been calculated already
        if s not in self.policy_s:
            # Leaf Node
            self.policy_s[s], value = self.player_network.predict(game, canonical_board)
            valid_moves = self.game_manager.get_valid_moves(game, player)

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
            return -value
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
                          * math.sqrt(self.state_visits[s])/ (1 + self.state_action_visits[(s, a)])
                else:
                    ucb = self.args.cpuct * self.policy_s[s][a] * math.sqrt(self.state_visits[s] + EPS)

                if ucb > current_best:
                    current_best = ucb
                    best_action = a

        a = best_action
        next_game, next_player = self.game_manager.get_next_state(game, player, a)
        # next_state = self.game_manager.get_canonical_form(next_game, next_player)

        # From the next state the function calls itself recursively until the leaf node is found
        v = self.search(next_game, next_player)

        # Update the action values for the taken action
        if (s, a) in self.action_values:
            self.action_values[(s, a)] = (self.state_action_visits[(s, a)] *
                                          self.action_values[(s, a)] + v) / (self.state_action_visits[(s, a)] + 1)
            self.state_action_visits[(s, a)] += 1

        else:
            self.action_values[(s, a)] = v
            self.state_action_visits[(s, a)] = 1

        self.state_visits[s] += 1
        # endregion
        return -v

