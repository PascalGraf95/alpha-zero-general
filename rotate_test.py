
import numpy as np
import cv2
from nonaga.NonagaLogic import Game
from nonaga.NonagaGameManager import NonagaGameManager as GameManager
from scipy.ndimage import rotate


def main():


    game = Game(scenario=2)
    game_manager = GameManager()
    original_board = game.board

    game_manager.display_by_board(original_board)
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