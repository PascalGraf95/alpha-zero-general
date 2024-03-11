
import numpy as np
import cv2
from nonaga.NonagaLogic import Game
from nonaga.NonagaGameManager import NonagaGameManager as GameManager
from scipy.ndimage import rotate


def main():

    mapping = {
        (0, 0):  None,     (0, 2):  (0, 12),     (0, 4):  (1, 13), (0, 6):  (2, 14), (0, 8):   None, (0, 10):  None,    (0, 12):  None,    (0, 14): None,
        (1, 1):  (0, 10),  (1, 3):  (1, 11),  (1, 5):  (2, 12), (1, 7):  (3, 13), (1, 9):   (4, 14), (1, 11):  None,    (1, 13):  None,
        (2, 0):  (0, 8),   (2, 2):  (1, 9),   (2, 4):  (2, 10), (2, 6):  (3, 11), (2, 8):   (4, 12), (2, 10):  (5, 13), (2, 12):  (6, 14), (2, 14): None,
        (3, 1):  (1, 7),   (3, 3):  (2, 8),   (3, 5):  (3, 9),  (3, 7):  (4, 10), (3, 9):   (5, 11), (3, 11):  (6, 12), (3, 13):  (7, 13),
        (4, 0):  (1, 5),   (4, 2):  (2, 6),   (4, 4):  (3, 7),  (4, 6):  (4, 8),  (4, 8):   (5, 9),  (4, 10):  (6, 10), (4, 12):  (7, 11), (4, 14): (8, 12),
        (5, 1):  (2, 4),   (5, 3):  (3, 5),   (5, 5):  (4, 6),  (5, 7):  (5, 7),  (5, 9):   (6, 8),  (5, 11):  (7, 9),  (5, 13):  (8, 10),
        (6, 0):  (2, 2),   (6, 2):  (3, 3),   (6, 4):  (4, 4),  (6, 6):  (5, 5),  (6, 8):   (6, 6),  (6, 10):  (7, 7),  (6, 12):  (8, 8),  (6, 14): (9, 9),
        (7, 1):  (3, 1),   (7, 3):  (4, 2),   (7, 5):  (5, 3),  (7, 7):  (6, 4),  (7, 9):   (7, 5),  (7, 11):  (8, 6),  (7, 13):  (9, 7),
        (8, 0):  None,     (8, 2):  (4, 0),   (8, 4):  (5, 1),  (8, 6):  (6, 2),  (8, 8):   (7, 3),  (8, 10):  (8, 4),  (8, 12):  (9, 5),  (8, 14): (10, 6),
        (9, 1):  None,     (9, 3):  None,     (9, 5):  (6, 0),  (9, 7):  (7, 1),  (9, 9):   (8, 2),  (9, 11):  (9, 3),  (9, 13):  (10, 4),
        (10, 0): None,     (10, 2): None,     (10, 4): None,    (10, 6): None,    (10, 8):  (8, 0),  (10, 10): (9, 1),  (10, 12): (10, 2),  (10, 14): (10, 3),
        (11, 1): None,     (11, 3): None,     (11, 5): None,    (11, 7): None,    (11, 9):  None,    (11, 11): (10, 0), (11, 13): (11, 1),
       }

    values = list(mapping.values())

    for v in values:
        if v is not None:
            if values.count(v) > 1:
                print(v)
                print(values.count(v))


    game = Game(scenario=2)
    game_manager = GameManager()
    original_board = game.board
    game_manager.display_by_board(original_board)
    game_manager.get_symmetries(game, 1, np.zeros(1080))


    # original_board = np.roll(original_board, shift=1, axis=2)
    # original_board = np.roll(original_board, shift=3, axis=1)


    #print(original_board.shape)
    #original_board[0] = np.arange(12*15).reshape(12, 15).astype(int)

    game_manager.display_by_board(original_board)
    for i in range(6):
        new_board = np.zeros((5, 12, 15))

        for key, val in mapping.items():
            if val:
                new_board[0][val] = original_board[0][key]
                new_board[1][val] = original_board[1][key]
        game_manager.display_by_board(new_board)
        original_board = new_board




if __name__ == '__main__':
    main()