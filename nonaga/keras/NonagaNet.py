import os
os.environ["KERAS_BACKEND"] = "torch"
import sys
sys.path.append('..')
from utils import *

import argparse
from keras.models import *
from keras.layers import Conv2D, Input, BatchNormalization, Flatten, Dense, Dropout, Softmax, Concatenate, Add, Activation, GlobalAveragePooling2D
from keras.layers import Conv2D
from keras.optimizers import Adam

def residual_block(filters, kernel_size=5):
    def apply(x):
        """A basic residual block with two Conv-BN-ReLU layers and a skip connection."""
        y = Conv2D(filters, kernel_size=kernel_size, padding="same", use_bias=False)(x)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)
        y = Conv2D(filters, kernel_size=kernel_size, padding="same", use_bias=False)(y)
        y = BatchNormalization()(y)
        x = Add()([x, y])
        x = Activation("relu")(x)
        return x
    return apply


class NonagaNet:
    def __init__(self, game_manager, args):
        # game params
        self.board_width, self.board_height = game_manager.get_board_size(game_manager.reset_board())
        self.args = args

        # Neural Net
        self.input_boards = Input(shape=(self.board_width, self.board_height, 5))  # batch_size  x board_x x board_y x 5

        # Initial conv layer
        x = Conv2D(args.num_channels, kernel_size=5, padding="same", use_bias=False)(self.input_boards)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # Residual blocks
        for _ in range(args.num_residuals):  # ToDo: make this a parameter like `args.res_blocks`
            x = residual_block(args.num_channels)(x)

        # Global pooling and dropout
        x = GlobalAveragePooling2D()(x)
        x = Dropout(args.dropout)(x)

        # --- Policy Head 1: Piece Movement ---
        self.pi1 = Dense(self.board_width * self.board_height * 6, activation="softmax", name="pi1")(x)

        # --- Policy Head 2: Tile Selection or Placement ---
        self.pi2 = Dense(self.board_width * self.board_height, activation="softmax", name="pi2")(x)

        # --- Value Head ---
        v = Dense(256, activation="relu")(x)
        v = Dropout(args.dropout)(v)
        self.v = Dense(1, activation='tanh', name='v')(v)


        self.pi1_model = Model(inputs=self.input_boards, outputs=[self.pi1, self.v])
        self.pi1_model.compile(loss=['categorical_crossentropy', 'mse'], optimizer=Adam(args.lr))
        self.pi2_model = Model(inputs=self.input_boards, outputs=[self.pi2, self.v])
        self.pi2_model.compile(loss=['categorical_crossentropy', 'mse'], optimizer=Adam(args.lr))
        self.model = Model(inputs=self.input_boards, outputs=[self.pi1, self.pi2, self.v])
        self.model.summary()

        # self.model.summary()

