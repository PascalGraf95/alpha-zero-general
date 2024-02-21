from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game as BaseGame
from nonaga.NonagaLogic import Game
import numpy as np


class NonagaGameManager:

    def reset_board(self):
        # return initial board (numpy board)
        return Game()

    def get_board_size(self, game: Game):
        return game.height, game.width

    def get_action_size(self, game: Game):
        return game.get_action_size()

    def get_next_state(self, game, player, action):
        new_game = Game()
        new_game.board = np.copy(game.board)
        new_game.phase = game.phase
        new_game.execute_move(action, player)
        next_player = (-1*player if new_game.phase == 0 else player)
        return new_game, next_player

    def get_valid_moves(self, game, player):
        valid_moves = game.get_legal_moves(player)
        if game.phase == 0:
            all_moves_masked = np.zeros((game.height, game.width, 6))
            for m in valid_moves:
                all_moves_masked[m[0], m[1], m[2]] = 1
        else:
            all_moves_masked = np.zeros((game.height, game.width))
            for m in valid_moves:
                all_moves_masked[m[0], m[1]] = 1
        return all_moves_masked.flatten()

    def has_game_ended(self, game, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        return game.check_for_game_end(player)

    def get_canonical_form(self, game, player):
        # return state if player==1, else return -state if player==-1
        game.board[1] *= player
        return game.board

    def get_symmetries(self, game, pi):
        return []

    def get_string_representation(self, canonical_board):
        return np.array2string(canonical_board)

    def display(self, game):
        width = 15
        height = 12
        print("    ", end="")

        # For each column print the index
        for x in range(width):
            print("{:02d}".format(x), end=" ")

        print("")
        print("---------------------------------------------------")
        for y in range(height):
            # For each row print the index
            print("{:02d} |".format(y), end="")
            for x in range(width):
                letter = " " if game.board[0][y][x] == 0 else "O"
                if game.board[1][y][x] != 0:
                    letter = "r" if game.board[1][y][x] == 1 else "b"
                print(letter, end="  ")
            print("|")
        print("---------------------------------------------------")


if __name__ == '__main__':
    game_manager = NonagaGameManager()
    game = Game()
    game_manager.display(game)

    player = 1
    num_turns = 0
    while True:
        num_turns += 1
        legal_moves = game_manager.get_valid_moves(game, player)

        if len(legal_moves) == 0:
            print("NO LEGAL MOVE")
            break

        move = legal_moves[np.random.randint(0, len(legal_moves))]
        game.execute_move(move, player)

        if num_turns % 3 == 0:
            # nonaga_game.display()
            player *= -1

            if game_manager.has_game_ended(game, player) != 0:
                print("WINNER WINNER")
                print("AFTER TURN: ", num_turns)
                break

        if num_turns > 200:
            print("RESET")
            num_turns = 0
            player = 1
            game = game_manager.reset_board()
    game_manager.display(game)
