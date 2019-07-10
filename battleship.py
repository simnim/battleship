#!/usr/bin/env python
"""
#FIXME: doc string
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.utils import Sequence

RANDOM_STATE = np.random.RandomState(seed=0)
BOARD_SIZE = 10

MAX_REVEAL_FRAC = .5

# 0 -> Water, 1 -> Fog, 2 -> Boat
DISPLAY_CLASSES = [
                    b'\xf0\x9f\x8c\x8a'.decode('utf-8'), # water
                    b'\xe2\x98\x81\xef\xb8\x8f '.decode('utf-8'), # fog
                    b'\xf0\x9f\x9a\xa2'.decode('utf-8'), # boat
                  ]

TARGETS_FANCY = [
                    b'\xf0\x9f\x92\xa8'.decode('utf-8'), # Whoosh
                    b'\xf0\x9f\x92\xa5'.decode('utf-8'), # BOOM
                ]

BOATS = [
            ['Carrier',5],
            ['Battleship',4],
            ['Cruiser',3],
            ['Submarine',3],
            ['Destroyer',2 ],
        ]


def render_board(board, target=None):
    ret = []
    for state in board.keys():
        ret.extend( [
            f'### STATE: {state} ###\n',
            '\n'.join([
                     ''.join([ DISPLAY_CLASSES[i] for i in row])
                     for row in board[state]
                   ])
            ] )
    return '\n'.join(ret)


def create_board():
    # Start with water
    board = np.full((BOARD_SIZE,BOARD_SIZE), 0)
    # Fill in boats
    for ship_name, boat_size in BOATS:
        # There's no way after 100,000 tries it'll fail to place a boat... if it fails it's from a bug
        for num_tries in range(100000):
            orientation_lr = RANDOM_STATE.randint(2)
            ul_coord_r = RANDOM_STATE.randint(BOARD_SIZE - (boat_size-1 if orientation_lr else 0))
            ul_coord_c = RANDOM_STATE.randint(BOARD_SIZE - (boat_size-1 if not orientation_lr else 0))
            if orientation_lr:
                # If we can place it, do so, otherwise we'll try another random spot...
                if not any(board[ul_coord_r:ul_coord_r+boat_size, ul_coord_c]==2):
                    board[ul_coord_r:ul_coord_r+boat_size, ul_coord_c] = 2
                    break
            else:
                # If we can place it, do so, otherwise we'll try another random spot...
                if not any(board[ul_coord_r, ul_coord_c:ul_coord_c+boat_size]==2):
                    board[ul_coord_r, ul_coord_c:ul_coord_c+boat_size] = 2
                    break
        else: # Didn't managed to place the boat... #nobreak
            raise Exception("Hmm, Never managed to place the boat... probably a bug in boat placing code...")
    return dict( hidden = board,
                  observed = np.full((BOARD_SIZE,BOARD_SIZE), 1),
                )

def reveal_some_of_the_board(board):
    # Pick a random fraction of the board and reveal it. (up to MAX_REVEAL_FRAC)
    positions_to_reveal = RANDOM_STATE.choice(  BOARD_SIZE*BOARD_SIZE
                                            , RANDOM_STATE.randint(BOARD_SIZE*BOARD_SIZE*MAX_REVEAL_FRAC)
                                            , replace=False
                                        )
    for pos in positions_to_reveal:
        board['observed'][int(pos/BOARD_SIZE), pos%BOARD_SIZE] = board['hidden'][int(pos/BOARD_SIZE), pos%BOARD_SIZE]
    return board

def get_model(weights_path=None):
    "This is a proof of concept, not a PhD thesis..."
    #create model.
    model = Sequential()
    # add model layers
    # Add some water around the edges
    model.add(ZeroPadding2D(1, input_shape=(BOARD_SIZE,BOARD_SIZE,1)))
    # Use adjacent pixels to help decide on a particular pixel.
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    #model.add(Conv2D(8, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    if weights_path:
        model.load_weights(weights_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


class battleship_batch_generator(Sequence):
    def __init__(self, batch_size):
        self.batch_size = batch_size
    def __len__(self):
        return 10000
    def __getitem__(self, idx):
        # Yeah... I'll just make it fresh each time.
        boards  = [ create_board() for i in range(self.batch_size) ]
        batch_x = [ b['observed'] for b in boards ]
        batch_y = [ b['hidden'].flatten() for b in boards ]
        return np.array(batch_x), np.array(batch_y)


def make_a_move(board, model):
    # First predict what pixels look good.
    preds = model.predict([board['observed']])


def play_game():
    # get the model
    model = get_model()
    #
