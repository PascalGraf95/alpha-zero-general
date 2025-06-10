from __future__ import print_function

import os
import sys
sys.path.append('..')
from Game import Game as BaseGame
from nonaga.NonagaLogic import Game
import numpy as np
import cv2


class NonagaGameManager:
    def __init__(self):
        self.previous_board = None

    def reset_board(self, scenario=0):
        # return initial board (numpy board)
        return Game(scenario=scenario)

    def get_board_size(self, game: Game):
        return game.height, game.width

    def get_observation_size(self):
        return 15, 12, 5

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
        legal_moves = game.get_legal_moves(player)
        if game.phase == 0:
            all_moves_masked = np.zeros((game.height, game.width, 6))
            for m in legal_moves:
                all_moves_masked[m[0], m[1], m[2]] = 1
        else:
            all_moves_masked = np.zeros((game.height, game.width))
            for m in legal_moves:
                all_moves_masked[m[0], m[1]] = 1
        legal_moves_indices = np.nonzero(all_moves_masked.flatten())[0]
        return all_moves_masked.flatten(), legal_moves, legal_moves_indices

    def get_symmetries(self, game, player, policy):
        # Copy Original board and policy
        original_canonical_board = self.get_canonical_form(game, player)
        original_canonical_board_copy = np.copy(original_canonical_board)
        original_policy_copy = np.copy(policy)

        board_policy_list = [[original_canonical_board, policy]]

        # Rotate
        for i in range(5):
            new_board = np.zeros((5, 12, 15))
            if game.phase == 0:
                new_policy = np.reshape(np.zeros(original_policy_copy.shape), (game.height, game.width, 6))
                original_policy_copy = np.reshape(original_policy_copy, (game.height, game.width, 6))
            else:
                new_policy = np.reshape(np.zeros(original_policy_copy.shape), (game.height, game.width))
                original_policy_copy = np.reshape(original_policy_copy, (game.height, game.width))

            for key, val in Game.rotation_mapping.items():
                if val:
                    new_board[0][val] = original_canonical_board_copy[0][key]
                    new_board[1][val] = original_canonical_board_copy[1][key]
                    new_board[2][val] = original_canonical_board_copy[2][key]
                    new_board[3][val] = original_canonical_board_copy[3][key]

                    if game.phase == 0:
                        for i2 in range(6):
                            new_policy[val][game.direction_mapping[i2]] = original_policy_copy[key][i2]
                    else:
                        try:
                            new_policy[val] = original_policy_copy[key]
                        except TypeError:
                            print("AHA!!!")
                            raise Exception
            if not self.is_board_configuration_valid(new_board):
                break
            board_policy_list.append([new_board, new_policy.flatten()])
            original_canonical_board_copy = np.copy(new_board)
            original_policy_copy = np.copy(new_policy.flatten())
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

    def is_board_configuration_valid(self, board):
        if np.sum(board[0]) != 19:
            return False
        if np.sum(board[1]) != 0:
            return False
        return True

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

    def display(self, game: Game):
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

    def draw_board_cv2(self, game: Game, current_player: int, scale: int = 40, turn_number: int = 0, save: bool = False, delete_old: bool = False):
        """
        Draws the board using OpenCV, showing last move by comparing with previous board.
        Highlights moved tiles and pieces.

        Args:
            game: Game instance with board state.
            scale: Pixel size per tile (diameter).
        """
        width = 15
        height = 12
        board = game.board

        tile_radius = scale // 2
        horizontal_spacing = int(scale * 0.6)
        vertical_spacing = int(scale * 1.2 * 0.6)

        img_height = height * vertical_spacing + scale
        img_width = width * horizontal_spacing + scale
        border = 10

        img = np.ones((img_height + 2 * border, img_width + 2 * border, 3), dtype=np.uint8) * 255

        # Compare with previous board (if exists)
        previous_tiles = None
        previous_pieces = None
        if  self.previous_board is not None and turn_number > 0:
            previous_tiles = self.previous_board[0]
            previous_pieces = self.previous_board[1]

        for y in range(height):
            for x in range(width):
                center_x = x * horizontal_spacing + tile_radius + border
                center_y = y * vertical_spacing + tile_radius + border
                center = (center_x, center_y)

                is_tile = board[0][y][x] == 1
                piece = board[1][y][x]

                # Draw tile
                if is_tile:
                    cv2.circle(img, center, tile_radius - 4, color=(230, 230, 230), thickness=-1)

                # Draw piece
                if piece == 1:
                    cv2.circle(img, center, tile_radius - 10, color=(0, 0, 255), thickness=-1)
                elif piece == -1:
                    cv2.circle(img, center, tile_radius - 10, color=(255, 0, 0), thickness=-1)

                # Draw outline for tiles
                if is_tile:
                    cv2.circle(img, center, tile_radius - 4, color=(180, 180, 180), thickness=1)

                # Highlight moved tiles
                if previous_tiles is not None and board[0][y][x] != previous_tiles[y][x]:
                    cv2.circle(img, center, tile_radius - 2, color=(0, 255, 255), thickness=2)

                # Highlight moved pieces
                if previous_pieces is not None and board[1][y][x] != previous_pieces[y][x]:
                    cv2.circle(img, center, tile_radius - 12, color=(0, 255, 0), thickness=2)

        # Coordinate labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        for x in range(width):
            label = "{:02d}".format(x)
            pos = (x * horizontal_spacing + tile_radius + border - 10, 15)
            cv2.putText(img, label, pos, font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        for y in range(height):
            label = "{:02d}".format(y)
            pos = (5, y * vertical_spacing + tile_radius + border + 5)
            cv2.putText(img, label, pos, font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        # Show turn indicator (Phase and next player, optional)
        turn_info = f"Phase: {game.phase} | Player: {'Red' if current_player == 1 else 'Blue'}"
        cv2.putText(img, turn_info, (10, img.shape[0] - 10), font, 0.5, (50, 50, 50), 1, cv2.LINE_AA)

        if save:
            if turn_number == 0 and delete_old:
                for file in os.listdir("last_game"):
                    os.remove(os.path.join("last_game", file))
            cv2.imwrite(os.path.join("last_game", "NonagaBoard_turn{:04d}.png".format(int(turn_number))), img)
        else:
            cv2.imshow("Nonaga Board", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Store board state for next comparison
        self.previous_board = np.copy(board)

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
