import logging

from tqdm import tqdm
from nonaga.NonagaGameManager import NonagaGameManager as GameManager

log = logging.getLogger(__name__)


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game_manager: GameManager, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game_manager = game_manager
        self.display = display

    def play_game(self, starting_player):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        current_player = starting_player
        game = self.game_manager.reset_board()
        num_steps = 0
        while self.game_manager.has_game_ended(game, current_player) == 0:
            num_steps += 1
            action = players[current_player + 1](game, current_player)
            game, current_player = self.game_manager.get_next_state(game, current_player, action)

            if num_steps > 300:
                return 0
        return self.game_manager.has_game_ended(game, 1)

    def play_games(self, num):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = int(num / 2)
        player_one_victories = 0
        player_two_victories = 0
        draws = 0
        for _ in tqdm(range(num), desc="Arena - Player 1 Starts"):
            game_result = self.play_game(starting_player=1)
            if game_result == 1:
                player_one_victories += 1
            elif game_result == -1:
                player_two_victories += 1
            else:
                draws += 1

        for _ in tqdm(range(num), desc="Arena - Player 2 Starts"):
            game_result = self.play_game(starting_player=-1)
            if game_result == 1:
                player_one_victories += 1
            elif game_result == -1:
                player_two_victories += 1
            else:
                draws += 1

        return player_one_victories, player_two_victories, draws
