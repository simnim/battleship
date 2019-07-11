#!/usr/bin/env python3
"""
This is a proof of concept Battleship AI implementation.
I've not spent much time optimizing the model,
so please tweak the model if actually want good performance.

There are two ways to run this code, either `train` or `play`.

Usage:
    battleship.py train [options]
    battleship.py play [options]

Options:
    -h, --help              Show this help message and exit
    --num-boards NUM        How many boards do you want to use to train? [default: 100000]
    --model-path PATH       Where do you want to store the model? [default: ~/models/battleship/model.joblib]

"""
import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense, Conv2D, Flatten
# from keras.utils import Sequence
import time
import os
import xgboost as xgb
from docopt import docopt
from joblib import dump, load

ARGS_DEFAULT = docopt(__doc__, argv=['train'])

RANDOM_STATE = np.random.RandomState(seed=0)
BOARD_SIZE = 10

MAX_REVEAL_FRAC = .5

# 0 -> Water, 1 -> Fog, 2 -> Boat
DISPLAY_CLASSES = [
                    b'\xf0\x9f\x8c\x8a'.decode('utf-8'), # water
                    b'\xe2\x98\x81\xef\xb8\x8f '.decode('utf-8'), # fog
                    b'\xf0\x9f\x9a\xa2'.decode('utf-8'), # boat
                  ]

HIT_OR_MISS_EMOJI = [
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
                     ''.join([
                                    HIT_OR_MISS_EMOJI[int(cell==2)]
                                  if target == (row_i*BOARD_SIZE+col_i)
                                  else
                                    DISPLAY_CLASSES[cell]
                                for col_i, cell in enumerate(row)
                            ])
                     for row_i, row in enumerate(board[state])
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



def get_features_given_board_and_rc(board, row_i, col_i):
    # row_i, col_i, pixel, num_hits, num_misses,
    #    up 1, left 1, down 1, right 1
    #    up 2, left 2, down 2, right 2
    #    left up 1, left down 1, right up 1, right down 1,
    ob_board = board['observed']
    features = [ row_i, col_i, ob_board[row_i,col_i], sum(sum(ob_board==2)), sum(sum(ob_board==0)) ]
    for distance in range(1,3):
        features.append( ob_board[ row_i-distance, col_i ] if row_i >= distance else 0 )
        features.append( ob_board[ row_i, col_i-distance ] if col_i >= distance else 0 )
        features.append( ob_board[ row_i+distance, col_i ] if row_i < (BOARD_SIZE-distance) else 0 )
        features.append( ob_board[ row_i, col_i+distance ] if col_i < (BOARD_SIZE-distance) else 0 )
    features.append( ob_board[ row_i-1, col_i-1 ] if (row_i >= 1 and col_i >= 1) else 0 )
    features.append( ob_board[ row_i+1, col_i-1 ] if (row_i < (BOARD_SIZE-1) and col_i >= 1) else 0 )
    features.append( ob_board[ row_i-1, col_i+1 ] if (row_i >= 1 and col_i < (BOARD_SIZE-1)) else 0 )
    features.append( ob_board[ row_i+1, col_i+1 ] if (row_i < (BOARD_SIZE-1) and col_i < (BOARD_SIZE-1)) else 0 )
    return features


def slice_up_board_into_features(board):
    pixel_features = []
    for row_i in range(BOARD_SIZE):
        for col_i in range(BOARD_SIZE):
            pixel_features.append( get_features_given_board_and_rc(board, row_i, col_i) )
    return pixel_features


def pick_next_spot_to_target(board, model):
    # First predict what pixels look good.
    preds_proba = model.predict_proba(slice_up_board_into_features(board))
    best_i = sorted(
                  zip(  range(BOARD_SIZE*BOARD_SIZE)
                      , preds_proba[:,1]
                      , (board['observed']==1).flatten()
                  )
                , key=lambda i: (not i[2], -i[1])
             )[0][0]
    return best_i


def get_model():
    return xgb.XGBClassifier(n_estimators=1000, max_depth=10)

def fit_model(  model_path = ARGS_DEFAULT['--model-path'],
                num_boards = ARGS_DEFAULT['--num-boards']
            ):
    model = get_model()
    # First get the boards
    boards = [ reveal_some_of_the_board(create_board()) for i in range(int(num_boards)) ]
    features = np.concatenate([
                    slice_up_board_into_features(board)
                    for board in boards
                ])
    labels = np.concatenate([ board['hidden'].flatten() for board in boards ])
    model.fit(features, labels)
    dump(model, os.path.expanduser(model_path))
    return model

def play_game(model_path = ARGS_DEFAULT['--model-path']):
    model = load(os.path.expanduser(model_path))

    board = create_board()
    print(render_board(board))
    for move_i in range(BOARD_SIZE*BOARD_SIZE):
        # Pick a spot to target, hit it, then keep going
        target_pos = pick_next_spot_to_target(board, model)
        # Update board with targetted position
        board['observed'][int(target_pos/BOARD_SIZE), target_pos%BOARD_SIZE] = board['hidden'][int(target_pos/BOARD_SIZE), target_pos%BOARD_SIZE]
        print(render_board(board, target_pos))
        # If we hit every boat:
        if sum(sum(board['observed']==2)) == sum([ b[1] for b  in BOATS ]):
            print("AI game over")
            break
        else:
            time.sleep(1)

def main():
    args = docopt(__doc__)
    # Make sure we have the directory
    os.system('mkdir -p %s'%os.path.expanduser(os.path.dirname(args['--model-path'])))
    if args['play']:
        assert os.path.exists(os.path.expanduser(args['--model-path'])), "You have to train the model first. (Did you provide the right model path?)"
        play_game(args['--model-path'])
    elif args['train']:
        fit_model(args['--model-path'], args['--num-boards'])
    else:
        sys.stderr.write("How did you get here???\n")
        sys.exit(1)


if __name__ == '__main__':
    main()
