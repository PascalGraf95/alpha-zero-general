import numpy as np

class Board():
    def __init__(self):
        """
        Set up initial board.

        Shape will be 3 x 15 x 12 (layers x width x height):
        The first layer contains the placement of the plates (0 and 1).
        The second layer contains the placement of the tokens. (-1, 0 and 1).
        The third layer contains information of the last moved plate (0 and 1).
        """

        self.width = 15
        self.height = 12
        # Create the empty board array all filled with zeroes.
        self.pieces = [None] * 3
        for i_layer in range(3):
            self.pieces[i_layer] = [None] * self.width
            for i_width in range(self.width):
                self.pieces[i_layer][i_width] = [0] * self.height

        # Set up the initial plate positions
        self.pieces[0][5][3] = self.pieces[0][7][3] = self.pieces[0][9][3] = 1
        self.pieces[0][4][4] = self.pieces[0][6][4] = self.pieces[0][8][4] = self.pieces[0][10][4] = 1
        self.pieces[0][3][5] = self.pieces[0][5][5] = self.pieces[0][7][5] = self.pieces[0][9][5] = self.pieces[0][11][5] = 1
        self.pieces[0][4][6] = self.pieces[0][6][6] = self.pieces[0][8][6] = self.pieces[0][10][6] = 1
        self.pieces[0][5][7] = self.pieces[0][7][7] = self.pieces[0][9][7] = 1

        # Set up initial token position
        self.pieces[1][5][3] = self.pieces[1][5][7] = self.pieces[1][11][5] = 1
        self.pieces[1][9][3] = self.pieces[1][9][7] = self.pieces[1][3][5] = -1

    """
    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.pieces[index]
    """

    def get_legal_move_direction(self, x_start, y_start, x_i, y_i):
        new_x = x_start
        new_y = y_start
        while True:
            target_x = new_x
            target_y = new_y

            new_y += y_i
            new_x += x_i

            # Check if plate is available in direction and plate is not blocked
            if self.pieces[0][new_x][new_y] == 1 and self.pieces[1][new_x][new_y] == 0:
                continue

            if x_start != target_x:
                return [[x_start, y_start], [target_x, target_y]]
            else:
                return None

    def add_to_legal_moves(self, list_to_add_to, legal_move):
        if legal_move is not None:
            list_to_add_to.append(legal_move)
        return list_to_add_to

    def get_legal_moves(self, color):
        """
        Returns all the legal moves for the given color (1 for red, -1 for black).

        A legal move consists of 4 indices, token start & end as well as plate start & end.
        """
        legal_moves = []  # stores the legal moves.

        # region - Token Moves -
        legal_token_moves = []
        # Iterate through the second (token) layer of the board
        for y in range(self.height):
            for x in range(self.width):
                # Moves can only start at positions where the player is.
                if self.pieces[1][x][y] == color:
                    # Check for all directions of movement
                    # Horizontal - Left
                    legal_token_moves = self.add_to_legal_moves(legal_token_moves, self.get_legal_move_direction(x, y, -2, 0))

                    # Horizontal - Right
                    legal_token_moves = self.add_to_legal_moves(legal_token_moves, self.get_legal_move_direction(x, y, 2, 0))

                    # Diagonal - Left Bottom
                    legal_token_moves = self.add_to_legal_moves(legal_token_moves, self.get_legal_move_direction(x, y, -1, 1))

                    # Diagonal - Right Bottom
                    legal_token_moves = self.add_to_legal_moves(legal_token_moves, self.get_legal_move_direction(x, y, 1, 1))

                    # Diagonal - Left Top
                    legal_token_moves = self.add_to_legal_moves(legal_token_moves, self.get_legal_move_direction(x, y, -1, -1))

                    # Diagonal - Left Top
                    legal_token_moves = self.add_to_legal_moves(legal_token_moves, self.get_legal_move_direction(x, y, 1, -1))
        # endregion

        # region - Plate Moves -
        # Iterate through the first (plate) layer of the board
        for y in range(self.height):
            for x in range(self.width):
                # Incorporate knowledge about plates becoming available after a token move
                for token_move in legal_token_moves:
                    if self.check_for_legal_plate_starts(x, y, token_move[0]):
                        legal_plate_moves = self.get_legal_plate_moves(x, y)
                        for plate_move in legal_plate_moves:
                            legal_moves.append([token_move[0], token_move[1], plate_move[0], plate_move[1]])
        # endregion
        return legal_moves

    def check_for_legal_plate_starts(self, x, y, new_available_plate):
        # If a plate is occupied by a token it cannot be moved except for if it's released with this token move
        if [x, y] != new_available_plate:
            if self.pieces[1][x][y] != 0:
                return False

        # Start plate has to be present & cannot be the one moved last.
        if self.pieces[0][x][y] != 1 or self.pieces[2][x][y] != 0:
            return False

        # A maximum of four surrounding plates can be occupied.
        # Can only place if at least 2 of the six neighbor fields are occupied
        neighbors_occupied = 0
        # Top Left
        if x > 0 and y > 0:
            neighbors_occupied += (1 if self.pieces[0][x-1][y-1] == 1 else 0)
        # Top right
        if x < (self.width-1) and y > 0:
            neighbors_occupied += (1 if self.pieces[0][x+1][y-1] == 1 else 0)
        # Left
        if x > 1:
            neighbors_occupied += (1 if self.pieces[0][x-2][y] == 1 else 0)
        # Right
        if x < (self.width-2):
            neighbors_occupied += (1 if self.pieces[0][x+2][y] == 1 else 0)
        # Bottom Left
        if x > 0 and y < (self.height-1):
            neighbors_occupied += (1 if self.pieces[0][x-1][y+1] == 1 else 0)
        # Top right
        if x < (self.width-1) and y < (self.height-1) :
            neighbors_occupied += (1 if self.pieces[0][x+1][y+1] == 1 else 0)

        if neighbors_occupied > 4:
            return False
        return True

    def get_legal_plate_moves(self, start_x, start_y):
        legal_moves = []
        # Iterate through the first (plate) layer of the board
        for y in range(self.height):
            for x in range(self.width):
                # Cannot place to the same position
                if x == start_x and y == start_y:
                    continue
                # Cannot place directly over a plate
                if y > 0:
                    if self.pieces[0][x][y-1] == 1:
                        continue
                # Cannot place directly under a plate
                if y < self.height - 1:
                    if self.pieces[0][x][y+1] == 1:
                        continue
                # Cannot place directly left a plate
                if x > 0:
                    if self.pieces[0][x-1][y] == 1:
                        continue
                # Cannot place directly right of a plate
                if x < self.width - 1:
                    if self.pieces[0][x+1][y] == 1:
                        continue

                # Can only place if at least 2 of the six neighbor fields are occupied
                neighbors_occupied = 0
                # Top Left
                if x > 0 and y > 0:
                    if not (x-1 == start_x and y-1 == start_y):
                        neighbors_occupied += (1 if self.pieces[0][x-1][y-1] == 1 else 0)
                # Top right
                if x < (self.width-1) and y > 0:
                    if not (x+1 == start_x and y-1 == start_y):
                        neighbors_occupied += (1 if self.pieces[0][x+1][y-1] == 1 else 0)
                # Left
                if x > 1:
                    if not (x-2 == start_x and y == start_y):
                        neighbors_occupied += (1 if self.pieces[0][x-2][y] == 1 else 0)
                # Right
                if x < (self.width-2):
                    if not (x+2 == start_x and y == start_y):
                        neighbors_occupied += (1 if self.pieces[0][x+2][y] == 1 else 0)
                # Bottom Left
                if x > 0 and y < (self.height-1):
                    if not (x-1 == start_x and y+1 == start_y):
                        neighbors_occupied += (1 if self.pieces[0][x-1][y+1] == 1 else 0)
                # Top right
                if x < (self.width-1) and y < (self.height-1):
                    if not (x+1 == start_x and y+1 == start_y):
                        neighbors_occupied += (1 if self.pieces[0][x+1][y+1] == 1 else 0)

                if neighbors_occupied >= 2:
                    legal_moves.append([[start_x, start_y], [x, y]])
        return legal_moves

    def execute_move(self, move, color):
        """Perform the given move on the board; (1=red,-1=black)
        """
        token_start = move[0]
        token_end = move[1]
        plate_start = move[2]
        plate_end = move[3]

        # Move Token
        self.pieces[1][token_start[0]][token_start[1]] = 0
        self.pieces[1][token_end[0]][token_end[1]] = color

        # Move Plate
        self.pieces[1][plate_start[0]][plate_start[1]] = 0
        self.pieces[1][plate_end[0]][plate_end[1]] = 1


if __name__ == '__main__':
    board = Board()
    legal_moves = board.get_legal_moves(1)
    print(legal_moves)