import os
os.environ["KERAS_BACKEND"] = "torch"
import sys
sys.path.append('..')
from utils import *

import argparse
from keras.models import *
from keras.layers import Conv2D, Input, BatchNormalization, Flatten, Dense, Dropout, Softmax, Concatenate, Add
from keras.layers import Conv2D
from keras.optimizers import Adam


class NonagaNet:
    def __init__(self, game_manager, args):
        # game params
        self.board_width, self.board_height = game_manager.get_board_size(game_manager.reset_board())
        self.args = args

        # Neural Net
        self.input_boards = Input(shape=(self.board_width, self.board_height, 5))  # batch_size  x board_x x board_y x 3
        x = Conv2D(args.num_channels, kernel_size=5, activation="relu", padding="same", use_bias=False)(self.input_boards)
        x = BatchNormalization()(x)
        y = Conv2D(args.num_channels, kernel_size=5, activation="relu", padding="same", use_bias=False)(x)
        y = BatchNormalization()(y)
        y = Conv2D(args.num_channels, kernel_size=5, activation="relu", padding="same", use_bias=False)(y)
        y = BatchNormalization()(y)
        x = Add()([y, x])
        y = Conv2D(args.num_channels, kernel_size=5, activation="relu", padding="same", use_bias=False)(x)
        y = BatchNormalization()(y)
        y = Conv2D(args.num_channels, kernel_size=5, activation="relu", padding="same", use_bias=False)(y)
        y = BatchNormalization()(y)
        x = Add()([y, x])
        y = Conv2D(args.num_channels, kernel_size=5, activation="relu", padding="same", use_bias=False)(x)
        y = BatchNormalization()(y)
        y = Conv2D(args.num_channels, kernel_size=5, activation="relu", padding="same", use_bias=False)(y)
        y = BatchNormalization()(y)
        x = Add()([y, x])
        x = Conv2D(args.num_channels/8, kernel_size=5, activation="relu", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)

        # x = Dense(256, activation="relu")(x)




        # Policy Head 1 - piece Movement
        # 6 layers of left/right/top right/top left/bottom right/bottom left
        #pi = Conv2D(args.num_channels/8, kernel_size=1)(x)
        #pi = Flatten()(pi)
        self.pi1 = Dense(self.board_width*self.board_height*6, activation="softmax", name="pi1")(x)

        # Policy Head 2 - tile Start and Target
        # 1 layer for tile to
        self.pi2 = Dense(self.board_width*self.board_height, activation="softmax", name="pi2")(x)

        # Value
        # v = Flatten()(x)
        # v = Dense(256, activation="relu")(v)
        # v = Dropout(args.dropout)(v)
        v = Dense(512, activation="relu")(x)
        self.v = Dense(1, activation='tanh', name='v')(v)

        self.pi1_model = Model(inputs=self.input_boards, outputs=[self.pi1, self.v])
        self.pi1_model.compile(loss=['categorical_crossentropy', 'mse'], optimizer=Adam(args.lr))
        self.pi2_model = Model(inputs=self.input_boards, outputs=[self.pi2, self.v])
        self.pi2_model.compile(loss=['categorical_crossentropy', 'mse'], optimizer=Adam(args.lr))
        self.model = Model(inputs=self.input_boards, outputs=[self.pi1, self.pi2, self.v])

        self.model.summary()

