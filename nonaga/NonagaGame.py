from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game as BaseGame
from NonagaLogic import Game
import numpy as np


class NonagaGame(BaseGame):
    def __init__(self):
        super().__init__()
        self.game = self.reset_board()

    def reset_board(self):
        # return initial board (numpy board)
        return Game()

    def get_board_size(self):
        return 15, 12

    def get_action_size(self):
        # return number of actions
        return 15 * 12

    def get_next_state(self, player, action):
        new_game = Game()
        new_game.pieces = np.copy(self.game.board)
        new_game.execute_move(action, player)
        return new_game.board, -player

    def get_valid_moves(self, player):
        return self.game.get_legal_moves(player)

    def has_game_ended(self, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        return self.game.check_for_game_end(player)

    def get_canonical_form(self, player):
        # return state if player==1, else return -state if player==-1
        self.game.board[1] *= player
        return self.game.board

    def get_symmetries(self, board, pi):
        return []

    def get_string_representation(self, game):
        return game.get_string_representation()

    def display(self):
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
                letter = " " if self.game.board[0][y][x] == 0 else "O"
                if self.game.board[1][y][x] != 0:
                    letter = "r" if self.game.board[1][y][x] == 1 else "b"
                print(letter, end="  ")
            print("|")
        print("---------------------------------------------------")


if __name__ == '__main__':
    nonaga_game = NonagaGame()
    nonaga_game.display()

    player = 1
    num_turns = 0
    while True:
        num_turns += 1
        legal_moves = nonaga_game.get_valid_moves(player)

        if len(legal_moves) == 0:
            print("NO LEGAL MOVE")
            break

        move = legal_moves[np.random.randint(0, len(legal_moves))]
        nonaga_game.game.execute_move(move, player)

        if num_turns % 3 == 0:
            # nonaga_game.display()
            player *= -1

            if nonaga_game.has_game_ended(player) != 0:
                print("WINNER WINNER")
                break

        if num_turns > 200:
            print("RESET")
            num_turns = 0
            player = 1
            nonaga_game.reset_board()
    nonaga_game.display()
