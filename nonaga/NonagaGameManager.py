from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game as BaseGame
from nonaga.NonagaLogic import Game
import numpy as np


class NonagaGameManager:

    def reset_board(self, scenario=0):
        # return initial board (numpy board)
        return Game(scenario=scenario)

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

    def get_symmetries(self, game, player, policy):
        original_canonical_board = np.copy(self.get_canonical_form(game, player))
        board_policy_list = [[original_canonical_board, policy]]

        # Only Top Bottom
        for new_board_t, new_policy_t in self.shift_generator(original_canonical_board, policy, game,
                                                              "top", return_original=False):
            board_policy_list.append([new_board_t, new_policy_t])

        for new_board_b, new_policy_b in self.shift_generator(original_canonical_board, policy, game,
                                                              "bottom", return_original=False):
            board_policy_list.append([new_board_b, new_policy_b])

        # Left Right + Top Bottom + Flip
        for new_board_r, new_policy_r in self.shift_generator(original_canonical_board, policy, game,
                                                              "right", return_original=False):
            for new_board_t, new_policy_t in self.shift_generator(new_board_r, new_policy_r, game,
                                                                  "top", return_original=True):
                board_policy_list.append([new_board_t, new_policy_t])

            for new_board_b, new_policy_b in self.shift_generator(new_board_r, new_policy_r, game,
                                                                  "bottom", return_original=False):
                board_policy_list.append([new_board_b, new_policy_b])

        for new_board_l, new_policy_l in self.shift_generator(original_canonical_board, policy, game,
                                                              "left", return_original=False):
            for new_board_t, new_policy_t in self.shift_generator(new_board_l, new_policy_l, game,
                                                                  "top", return_original=True):
                board_policy_list.append([new_board_t, new_policy_t])

            for new_board_b, new_policy_b in self.shift_generator(new_board_l, new_policy_l, game,
                                                                  "bottom", return_original=False):
                board_policy_list.append([new_board_b, new_policy_b])
        return board_policy_list

    def flip_generator(self, original_canonical_board, original_policy, game, direction="horizontal",
                       return_original=False):
        canonical_board = np.copy(original_canonical_board)
        if game.phase == 0:
            board_policy_reshaped = np.reshape(np.copy(np.copy(original_policy)), (game.height, game.width, 6))
        else:
            board_policy_reshaped = np.reshape(np.asarray(np.copy(original_policy)), (game.height, game.width))

        if return_original:
            yield canonical_board, board_policy_reshaped.flatten().tolist()

        if direction == "horizontal":
            canonical_board = np.flip(canonical_board, axis=1)
            board_policy_reshaped = np.flip(board_policy_reshaped, axis=0)
            # self.display_by_board(canonical_board)
            yield canonical_board, board_policy_reshaped.flatten().tolist()

        elif direction == "vertical":
            canonical_board = np.flip(canonical_board, axis=2)
            board_policy_reshaped = np.flip(board_policy_reshaped, axis=1)
            # self.display_by_board(canonical_board)
            yield canonical_board, board_policy_reshaped.flatten().tolist()

    def shift_generator(self, original_canonical_board, original_policy, game, direction="right", return_original=False):
        canonical_board = np.copy(original_canonical_board)
        if game.phase == 0:
            board_policy_reshaped = np.reshape(np.copy(np.copy(original_policy)), (game.height, game.width, 6))
        else:
            board_policy_reshaped = np.reshape(np.asarray(np.copy(original_policy)), (game.height, game.width))

        if return_original:
            yield canonical_board, board_policy_reshaped.flatten().tolist()

        if direction == 'right':
            while not (np.any(canonical_board[0, :, -2:] != 0) or np.any(canonical_board[1, :, -2:] != 0) or
                       np.any(canonical_board[3, :, -2:] != 0)):
                canonical_board = np.roll(canonical_board, shift=2, axis=2)
                board_policy_reshaped = np.roll(board_policy_reshaped, shift=2, axis=1)
                yield canonical_board, board_policy_reshaped.flatten().tolist()

        elif direction == 'left':
            while not (np.any(canonical_board[0, :, :2] != 0) or np.any(canonical_board[1, :, :2] != 0) or
                       np.any(canonical_board[3, :, :2] != 0)):
                canonical_board = np.roll(canonical_board, shift=-2, axis=2)
                board_policy_reshaped = np.roll(board_policy_reshaped, shift=-2, axis=1)
                yield canonical_board, board_policy_reshaped.flatten().tolist()

        elif direction == 'top':
            while not (np.any(canonical_board[0, :1, :] != 0) or np.any(canonical_board[1, :1, :] != 0) or
                       np.any(canonical_board[3, :1, :] != 0)):
                canonical_board = np.roll(canonical_board, shift=-1, axis=1)
                board_policy_reshaped = np.roll(board_policy_reshaped, shift=-1, axis=0)
                yield canonical_board, board_policy_reshaped.flatten().tolist()

        elif direction == 'bottom':
            while not (np.any(canonical_board[0, -1:, :] != 0) or np.any(canonical_board[1, -1:, :] != 0) or
                       np.any(canonical_board[3, -1:, :] != 0)):
                canonical_board = np.roll(canonical_board, shift=1, axis=1)
                board_policy_reshaped = np.roll(board_policy_reshaped, shift=1, axis=0)
                yield canonical_board, board_policy_reshaped.flatten().tolist()

    def has_game_ended(self, game, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        return game.check_for_game_end(player)

    def get_canonical_form(self, game, player):
        # return state if player==1, else return -state if player==-1
        canonical_board = np.copy(game.board)
        canonical_board[1] *= player
        return canonical_board

    def get_string_representation(self, game, canonical_board):
        tile_string = "tiles:"
        pieces_string = "_pieces:"
        last_moved_string = "_lastmoved:"
        selected_string = "_selected:"
        phase_string = "_phase:{:01d}".format(int(canonical_board[4][0][0]))

        for y in range(game.height):
            for x in range(game.width):
                if canonical_board[0][y][x] == 1:
                    tile_string += "{:02d}{:02d}_".format(y, x)
                if canonical_board[1][y][x] != 0:
                    pieces_string += "{:02d}{:02d}{:02d}_".format(int(canonical_board[1][y][x]), y, x)
                if canonical_board[2][y][x] == 1:
                    last_moved_string += "{:02d}{:02d}".format(y, x)
                if canonical_board[3][y][x] == 1:
                    selected_string += "{:02d}{:02d}".format(y, x)

        string_representation = tile_string + pieces_string + last_moved_string + selected_string + phase_string
        # print(len(string_representation))
        # print(len(np.array2string(canonical_board)))
        return string_representation

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

    def display_by_board(self, canonical_board):
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
                letter = " " if canonical_board[0][y][x] == 0 else "O"
                if canonical_board[1][y][x] != 0:
                    letter = "r" if canonical_board[1][y][x] == 1 else "b"
                print(letter, end="  ")
            print("|")
        print("---------------------------------------------------")


if __name__ == '__main__':
    game_manager = NonagaGameManager()
    game = Game(scenario=0)
    game_manager.display(game)
    game_manager.get_string_representation(game, game_manager.get_canonical_form(game,1))

    player = 1
    num_turns = 0
    while True:
        num_turns += 1
        legal_moves = game_manager.get_valid_moves(game, player)

        if len(legal_moves) == 0:
            print("NO LEGAL MOVE")
            break

        nonzeros = np.nonzero(legal_moves)[0]
        move = np.random.choice(nonzeros)
        game.execute_move(move, player, form=0)

        if num_turns % 3 == 0:
            game_manager.display(game)
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
