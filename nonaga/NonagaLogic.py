import numpy as np


class Game:
    direction_dictionary = {
        0: [0, -2],     # horizontal left
        1: [0, 2],      # horizontal right
        2: [-1, -1],    # top left
        3: [-1, 1],     # top right
        4: [1, -1],     # bottom left
        5: [1, 1]       # bottom right
    }

    direction_mapping = {
        0: 2,
        1: 5,
        2: 3,
        3: 1,
        4: 0,
        5: 4
    }

    rotation_mapping = {
        (0, 0):  None,     (0, 2):  (0, 12),     (0, 4):  (1, 13), (0, 6):  (2, 14), (0, 8):   None, (0, 10):  None,    (0, 12):  None,    (0, 14): None,
        (1, 1):  (0, 10),  (1, 3):  (1, 11),  (1, 5):  (2, 12), (1, 7):  (3, 13), (1, 9):   (4, 14), (1, 11):  None,    (1, 13):  None,
        (2, 0):  (0, 8),   (2, 2):  (1, 9),   (2, 4):  (2, 10), (2, 6):  (3, 11), (2, 8):   (4, 12), (2, 10):  (5, 13), (2, 12):  (6, 14), (2, 14): None,
        (3, 1):  (1, 7),   (3, 3):  (2, 8),   (3, 5):  (3, 9),  (3, 7):  (4, 10), (3, 9):   (5, 11), (3, 11):  (6, 12), (3, 13):  (7, 13),  # x
        (4, 0):  (1, 5),   (4, 2):  (2, 6),   (4, 4):  (3, 7),  (4, 6):  (4, 8),  (4, 8):   (5, 9),  (4, 10):  (6, 10), (4, 12):  (7, 11), (4, 14): (8, 12),  # x
        (5, 1):  (2, 4),   (5, 3):  (3, 5),   (5, 5):  (4, 6),  (5, 7):  (5, 7),  (5, 9):   (6, 8),  (5, 11):  (7, 9),  (5, 13):  (8, 10),  # x
        (6, 0):  (2, 2),   (6, 2):  (3, 3),   (6, 4):  (4, 4),  (6, 6):  (5, 5),  (6, 8):   (6, 6),  (6, 10):  (7, 7),  (6, 12):  (8, 8),  (6, 14): (9, 9),  # x
        (7, 1):  (3, 1),   (7, 3):  (4, 2),   (7, 5):  (5, 3),  (7, 7):  (6, 4),  (7, 9):   (7, 5),  (7, 11):  (8, 6),  (7, 13):  (9, 7),
        (8, 0):  (7, 3),   (8, 2):  (4, 0),   (8, 4):  (5, 1),  (8, 6):  (6, 2),  (8, 8):   (7, 3),  (8, 10):  (8, 4),  (8, 12):  (9, 5),  (8, 14): (10, 6),
        (9, 1):  None,     (9, 3):  None,     (9, 5):  (6, 0),  (9, 7):  (7, 1),  (9, 9):   (8, 2),  (9, 11):  (9, 3),  (9, 13):  (10, 4),
        (10, 0): (10, 2),  (10, 2): None,     (10, 4): None,    (10, 6): None,    (10, 8):  (8, 0),  (10, 10): (9, 1),  (10, 12): (10, 2),  (10, 14): (10, 3),
        (11, 1): None,     (11, 3): None,     (11, 5): None,    (11, 7): None,    (11, 9):  None,    (11, 11): (10, 0), (11, 13): (11, 1),
       }

    def __init__(self, scenario=0):
        """
        Set up initial board.

        Shape will be 15 x 12 x 5(layers x width x height):
        The first layer contains the placement of the tiles (0 and 1).
        The second layer contains the placement of the pieces. (-1, 0 and 1).
        The third layer contains information of the last moved tile (0 and 1).
        The fourth layer contains information of the selected tile (0 and 1).
        The fifth layer contains information about the current phase of the turn (0 - 2).
        """

        self.width = 15
        self.height = 12
        # Create the empty board array all filled with zeroes.
        self.board = np.zeros((5, self.height, self.width), dtype=float)

        # Set up the initial tile positions
        self.board[0][3][5] = self.board[0][3][7] = self.board[0][3][9] = 1
        self.board[0][4][4] = self.board[0][4][6] = self.board[0][4][8] = self.board[0][4][10] = 1
        self.board[0][5][3] = self.board[0][5][5] = self.board[0][5][7] = self.board[0][5][9] = self.board[0][5][11] = 1
        self.board[0][6][4] = self.board[0][6][6] = self.board[0][6][8] = self.board[0][6][10] = 1
        self.board[0][7][5] = self.board[0][7][7] = self.board[0][7][9] = 1

        # Default start constellation
        if scenario == 0:
            # Set up initial piece position
            self.board[1][3][5] = self.board[1][7][5] = self.board[1][5][11] = 1
            self.board[1][3][9] = self.board[1][7][9] = self.board[1][5][3] = -1
        # Player 1 needs to do the right move prevent player -1 from winning
        elif scenario == 1:
            self.board[1][3][5] = self.board[1][7][5] = self.board[1][5][11] = 1
            self.board[1][3][9] = self.board[1][4][10] = self.board[1][5][3] = -1
        elif scenario == 2:
            self.board[0][2][4] = self.board[0][1][3] = self.board[0][0][2] = 1
            self.board[0][3][3] = self.board[0][2][2] = self.board[0][1][1] = self.board[0][0][0] = 1
            self.board[0][4][2] = self.board[0][3][1] = self.board[0][2][0] = 1
            # self.board[0][5][1] = self.board[0][5][13] = self.board[0][5][7] = 1
            # self.board[0][1][7] = self.board[0][9][7] = 1
            # self.board[0][2][6] = self.board[0][2][8] = self.board[0][8][6] = self.board[0][8][8] = 1
            # self.board[0][6][12] = self.board[0][7][11] = 1


        self.legal_moves = []

        # One turn consists of three phases (0-2)
        self.phase = 0

    def get_action_size(self):
        if self.phase == 0:
            return self.width * self.height * 6
        else:
            return self.width * self.height

    def get_string_representation(self):
        return np.array2string(self.board)

    def get_legal_moves(self, player):
        """
        Returns all the legal moves for the given color (1 for red, -1 for black).

        A legal move consists of 4 indices, piece start & end as well as tile start & end.
        """
        legal_moves = []  # stores the legal moves.

        if self.phase == 0:
            # region - Piece Moves -
            # Iterate through the second (piece) layer of the board
            for y in range(self.height):
                for x in range(self.width):
                    # Moves can only start at positions where the player is.
                    if self.board[1][y][x] == player:
                        # Check for all directions of movement
                        # Horizontal - Left
                        if self._is_piece_direction_legal(x, y, -2, 0):
                            legal_moves.append([y, x, 0])
                        # Horizontal - Right
                        if self._is_piece_direction_legal(x, y, 2, 0):
                            legal_moves.append([y, x, 1])
                        # Diagonal - Left Top
                        if self._is_piece_direction_legal(x, y, -1, -1):
                            legal_moves.append([y, x, 2])
                        # Diagonal - Right Top
                        if self._is_piece_direction_legal(x, y, 1, -1):
                            legal_moves.append([y, x, 3])
                        # Diagonal - Left Bottom
                        if self._is_piece_direction_legal(x, y, -1, 1):
                            legal_moves.append([y, x, 4])
                        # Diagonal - Right Bottom
                        if self._is_piece_direction_legal(x, y, 1, 1):
                            legal_moves.append([y, x, 5])

            # endregion
        
        elif self.phase == 1:
            # region - Tile Start -
            # Iterate through the first (tile) layer of the board
            for y in range(self.height):
                for x in range(self.width):
                    if self._is_legal_start_tile(x, y):
                        legal_moves.append([y, x])
            # endregion

        elif self.phase == 2:
            # region - Tile Target -
            # Iterate through the first (tile) layer of the board
            for y in range(self.height):
                for x in range(self.width):
                    if self._is_legal_target_tile(x, y):
                        legal_moves.append([y, x])
            # endregion
        self.legal_moves = legal_moves
        return legal_moves

    def _is_piece_direction_legal(self, x_start, y_start, x_i, y_i):
        new_y = y_start + y_i
        new_x = x_start + x_i

        if new_y < 0 or new_y >= self.height:
            return False

        if new_x < 0 or new_x >= self.width:
            return False

        # Check if tile is available in direction and tile is not blocked
        if self.board[0][new_y][new_x] == 1 and self.board[1][new_y][new_x] == 0:
            return True
        return False

    def _move_piece_in_direction(self, x_start, y_start, direction):
        new_x = x_start
        new_y = y_start

        y_i, x_i = Game.direction_dictionary[direction]

        while True:
            target_x = new_x
            target_y = new_y

            new_y += y_i
            new_x += x_i

            if new_y < 0 or new_y >= self.height:
                break

            if new_x < 0 or new_x >= self.width:
                break

            # Check if tile is available in direction and tile is not blocked
            if self.board[0][new_y][new_x] == 1 and self.board[1][new_y][new_x] == 0:
                continue
            else:
                break
        return target_y, target_x

    def _is_legal_start_tile(self, x, y):
        # If a tile is occupied by a piece it cannot be moved
        if self.board[1][y][x] != 0:
            return False

        # Start tile has to be present & cannot be the one moved last.
        if self.board[0][y][x] != 1 or self.board[2][y][x] != 0:
            return False

        # A maximum of four surrounding tiles can be occupied.
        neighbors_occupied = 0
        # Top Left
        if x > 0 and y > 0:
            neighbors_occupied += (1 if self.board[0][y-1][x-1] == 1 else 0)
        # Top right
        if x < (self.width-1) and y > 0:
            neighbors_occupied += (1 if self.board[0][y-1][x+1] == 1 else 0)
        # Left
        if x > 1:
            neighbors_occupied += (1 if self.board[0][y][x-2] == 1 else 0)
        # Right
        if x < (self.width-2):
            neighbors_occupied += (1 if self.board[0][y][x+2] == 1 else 0)
        # Bottom Left
        if x > 0 and y < (self.height-1):
            neighbors_occupied += (1 if self.board[0][y+1][x-1] == 1 else 0)
        # Top right
        if x < (self.width-1) and y < (self.height-1):
            neighbors_occupied += (1 if self.board[0][y+1][x+1] == 1 else 0)

        if neighbors_occupied > 4:
            return False
        return True

    def _is_legal_target_tile(self, x, y):
        start_y, start_x = np.asarray(np.where(self.board[3] == 1)).T[0]

        # Field cannot be occupied by another tile
        if self.board[0][y][x] == 1:
            return False

        # Cannot place directly under a tile
        if y > 0:
            if self.board[0][y-1][x] == 1:
                return False
        # Cannot place directly over a tile
        if y < self.height - 1:
            if self.board[0][y+1][x] == 1:
                return False
        # Cannot place directly right of a tile
        if x > 0:
            if self.board[0][y][x-1] == 1:
                return False
        # Cannot place directly left of a tile
        if x < self.width - 1:
            if self.board[0][y][x+1] == 1:
                return False

        # Can only place if at least 2 of the six neighbor fields are occupied
        neighbors_occupied = 0
        # Top Left
        if x > 0 and y > 0:
            if not (x-1 == start_x and y-1 == start_y):
                neighbors_occupied += (1 if self.board[0][y-1][x-1] == 1 else 0)
        # Top right
        if x < (self.width-1) and y > 0:
            if not (x+1 == start_x and y-1 == start_y):
                neighbors_occupied += (1 if self.board[0][y-1][x+1] == 1 else 0)
        # Left
        if x > 1:
            if not (x-2 == start_x and y == start_y):
                neighbors_occupied += (1 if self.board[0][y][x-2] == 1 else 0)
        # Right
        if x < (self.width-2):
            if not (x+2 == start_x and y == start_y):
                neighbors_occupied += (1 if self.board[0][y][x+2] == 1 else 0)
        # Bottom Left
        if x > 0 and y < (self.height-1):
            if not (x-1 == start_x and y+1 == start_y):
                neighbors_occupied += (1 if self.board[0][y+1][x-1] == 1 else 0)
        # Top right
        if x < (self.width-1) and y < (self.height-1):
            if not (x+1 == start_x and y+1 == start_y):
                neighbors_occupied += (1 if self.board[0][y+1][x+1] == 1 else 0)

        if neighbors_occupied >= 2:
            return True

    def execute_move(self, move, player, form=0):
        """Perform the given move on the board; (1=red,-1=black)
        """
        if form == 0:
            if self.phase == 0:
                input_move = np.zeros((self.width * self.height * 6))
                input_move[move] = 1
                reshaped_input = np.reshape(input_move, (self.height, self.width, 6))
                move_array = np.nonzero(reshaped_input)
                move = [move_array[0][0], move_array[1][0], move_array[2][0]]
            else:
                input_move = np.zeros((self.width * self.height))
                input_move[move] = 1
                reshaped_input = np.reshape(input_move, (self.height, self.width))
                move_array = np.nonzero(reshaped_input)
                move = [move_array[0][0], move_array[1][0]]

        if self.phase == 0:
            # region - Piece Moves -
            self.board[1][move[0]][move[1]] = 0
            target_y, target_x = self._move_piece_in_direction(move[1], move[0], move[2])
            self.board[1][target_y][target_x] = player
            # endregion

        elif self.phase == 1:
            # region - Tile Start -
            self.board[3, move[0], move[1]] = 1
            # endregion

        elif self.phase == 2:
            # region - Tile Target -
            # Get & Reset Starting Tile
            start_y, start_x = np.asarray(np.where(self.board[3] == 1)).T[0]
            self.board[3] = np.zeros((self.height, self.width))

            # Move Tile
            self.board[0, start_y, start_x] = 0
            self.board[0, move[0], move[1]] = 1

            # Re-Set immovable tile
            self.board[2] = np.zeros((self.height, self.width))
            self.board[2, move[0], move[1]] = 1
            # endregion

        # region - Update Phase -
        self.phase += 1
        self.phase %= 3
        self.board[4] = np.ones((self.height, self.width)) * self.phase
        # endregion

    def check_for_game_end(self, player):
        # returns 1 if player in player has won, -1 if other player won, 0 if not ended
        # Check all possibilities
        for x in range(self.width):
            for y in range(self.height):
                # Horizontal Line
                if self.same_player_left(player, x, y) and self.same_player_right(player, x, y) and self.player_on_current_field(player, x, y):
                    return 1
                if self.same_player_left(-player, x, y) and self.same_player_right(-player, x, y) and self.player_on_current_field(-player, x, y):
                    return -1

                # Triangle Top
                if self.same_player_bottom_left(player, x, y) and self.same_player_bottom_right(player, x, y) and self.player_on_current_field(player, x, y):
                    return 1
                if self.same_player_bottom_left(-player, x, y) and self.same_player_bottom_right(-player, x, y) and self.player_on_current_field(-player, x, y):
                    return -1

                # Triangle Bottom
                if self.same_player_top_left(player, x, y) and self.same_player_top_right(player, x, y) and self.player_on_current_field(player, x, y):
                    return 1
                if self.same_player_top_left(-player, x, y) and self.same_player_top_right(-player, x, y) and self.player_on_current_field(-player, x, y):
                    return -1

                # Triangle Left
                if self.same_player_top_right(player, x, y) and self.same_player_bottom_right(player, x, y) and self.player_on_current_field(player, x, y):
                    return 1
                if self.same_player_top_right(-player, x, y) and self.same_player_bottom_right(-player, x, y) and self.player_on_current_field(-player, x, y):
                    return -1

                # Triangle Right
                if self.same_player_top_left(player, x, y) and self.same_player_bottom_left(player, x, y) and self.player_on_current_field(player, x, y):
                    return 1
                if self.same_player_top_left(-player, x, y) and self.same_player_bottom_left(-player, x, y) and self.player_on_current_field(-player, x, y):
                    return -1

                # /-
                if self.same_player_bottom_left(player, x, y) and self.same_player_right(player, x, y) and self.player_on_current_field(player, x, y):
                    return 1
                if self.same_player_bottom_left(-player, x, y) and self.same_player_right(-player, x, y) and self.player_on_current_field(-player, x, y):
                    return -1

                # -\
                if self.same_player_bottom_right(player, x, y) and self.same_player_left(player, x, y) and self.player_on_current_field(player, x, y):
                    return 1
                if self.same_player_bottom_right(-player, x, y) and self.same_player_left(-player, x, y) and self.player_on_current_field(-player, x, y):
                    return -1

                # _/
                if self.same_player_left(player, x, y) and self.same_player_top_right(player, x, y) and self.player_on_current_field(player, x, y):
                    return 1
                if self.same_player_left(-player, x, y) and self.same_player_top_right(-player, x, y) and self.player_on_current_field(-player, x, y):
                    return -1

                # \_
                if self.same_player_top_left(player, x, y) and self.same_player_right(player, x, y) and self.player_on_current_field(player, x, y):
                    return 1
                if self.same_player_top_left(-player, x, y) and self.same_player_right(-player, x, y) and self.player_on_current_field(-player, x, y):
                    return -1

                # \
                if self.same_player_top_left(player, x, y) and self.same_player_bottom_right(player, x, y) and self.player_on_current_field(player, x, y):
                    return 1
                if self.same_player_top_left(-player, x, y) and self.same_player_bottom_right(-player, x, y) and self.player_on_current_field(-player, x, y):
                    return -1

                # /
                if self.same_player_top_right(player, x, y) and self.same_player_bottom_left(player, x, y) and self.player_on_current_field(player, x, y):
                    return 1
                if self.same_player_top_right(-player, x, y) and self.same_player_bottom_left(-player, x, y) and self.player_on_current_field(-player, x, y):
                    return -1
        if len(self.get_legal_moves(player)) == 0:
            return -1
        return 0

    def player_on_current_field(self, player, x, y):
        return self.board[1][y][x] == player

    def same_player_right(self, player, x, y):
        if x+2 < self.width:
            return self.board[1][y][x+2] == player
        return False

    def same_player_left(self, player, x, y):
        if x-2 >= 0:
            return self.board[1][y][x-2] == player
        return False

    def same_player_top_left(self, player, x, y):
        if x-1 >= 0 and y-1 >= 0:
            return self.board[1][y-1][x-1] == player
        return False

    def same_player_top_right(self, player, x, y):
        if x+1 < self.width and y - 1 >= 0:
            return self.board[1][y-1][x+1] == player
        return False

    def same_player_bottom_left(self, player, x, y):
        if x-1 >= 0 and y + 1 < self.height:
            return self.board[1][y+1][x-1] == player
        return False

    def same_player_bottom_right(self, player, x, y):
        if x+1 < self.width and y + 1 < self.height:
            return self.board[1][y+1][x+1] == player
        return False


if __name__ == '__main__':
    game = Game()
    for i in range(3):
        legal_moves = game.get_legal_moves(1)
        game.execute_move(legal_moves[0], 1, form=1)

    # print(len(legal_moves))
    # print(legal_moves)
