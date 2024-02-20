import sys
sys.path.append('..')
from utils import *

import argparse
from keras.models import *
from keras.layers import Conv2D, Input, BatchNormalization, Flatten, Dense, Dropout, Softmax
from keras.optimizers import Adam


class NonagaNet:
    def __init__(self, game_manager, args):
        # game params
        self.board_width, self.board_height = game_manager.get_board_size(game_manager.reset_board())
        self.args = args

        # ToDo: Preprocess input to channels last
        # Neural Net
        self.input_boards = Input(shape=(5, self.board_width, self.board_height))  # batch_size  x board_x x board_y x 3
        x = Conv2D(args.num_channels, kernel_size=5, activation="relu", padding="same", data_format="channels_first")(self.input_boards)
        x = BatchNormalization()(x)
        x = Conv2D(args.num_channels, kernel_size=5, activation="relu", padding="same", data_format="channels_first")(x)
        x = BatchNormalization()(x)
        x = Conv2D(args.num_channels, kernel_size=5, activation="relu", padding="same", data_format="channels_first")(x)
        x = BatchNormalization()(x)
        x = Conv2D(args.num_channels, kernel_size=5, activation="relu", padding="same", data_format="channels_first")(x)
        x = BatchNormalization()(x)
        x = Conv2D(args.num_channels/2, kernel_size=1, data_format="channels_first")(x)
        x = Flatten()(x)

        # Policy Head 1 - piece Movement
        # 6 layers of left/right/top right/top left/bottom right/bottom left
        self.pi1 = Dense(self.board_width*self.board_height*6, activation="softmax")(x)

        # Policy Head 2 - tile Start and Target
        # 1 layer for tile to
        self.pi2 = Dense(self.board_width*self.board_height, activation="softmax")(x)

        # Value
        v = Dense(512, activation="relu")(x)
        v = Dense(256, activation="relu")(v)
        self.v = Dense(1, activation='tanh', name='v')(v)

        self.pi1_model = Model(inputs=self.input_boards, outputs=[self.pi1, self.v])
        self.pi1_model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(args.lr))
        self.pi2_model = Model(inputs=self.input_boards, outputs=[self.pi2, self.v])
        self.pi2_model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(args.lr))
        self.model = Model(inputs=self.input_boards, outputs=[self.pi1, self.pi2, self.v])

        self.model.summary()

