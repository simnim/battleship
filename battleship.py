#!/usr/bin/env python3
"""
This is a proof of concept Battleship AI implementation.
I've not spent much time optimizing the model. I just used what I thought would
work well enough for the POC. Tweak it to make it better.
Right now I'm using a RandomForestClassifier with 1000 trees and maxdepth=10.

To improve this model, simply tweak the get_blast_model function and make sure the
object follows sklearn's api. (supports .fit() and .predict_proba() )

There are two ways to run this code, either `train` or `play`.
If you want to use my prertrained model you'll probably be ok as is,
just have to copy it into the model directory. I trained it using
scikit-learn==0.19.2 and numpy==1.16.1 so if it's not working try with
those versions.

Usage:
    battleship.py train-blast [options]
    battleship.py train-place [options]
    battleship.py play [options]

Options:
    -h, --help                Show this help message and exit
    --num-boards NUM          How many boards do you want to use to train? [default: 100000]
    --blast-model-path PATH   Where do you want to store the model? [default: ~/models/battleship/blast_model.joblib]
    --place-model-path PATH   Where do you want to store the model? [default: ~/models/battleship/place_model.joblib]
    --print-delay SECONDS     In play mode, how many seconds to wait between printing AI moves? (try .1 for it to run faster) [default: 1]
"""
import numpy as np
import time
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from docopt import docopt
from joblib import dump, load
import fastnumbers
import datetime
from itertools import repeat

ARGS_DEFAULT = docopt(__doc__, argv=['train-blast'])

#np.random.seed(0)
BOARD_SIZE = 10

MAX_REVEAL_FRAC = .5

# 0 -> Water, 1 -> Fog, 2 -> Boat
DISPLAY_CLASSES = [
                    b'\xf0\x9f\x8c\x8a '.decode('utf-8'), # water
                    b'\xe2\x98\x81\xef\xb8\x8f '.decode('utf-8'), # fog
                    b'\xf0\x9f\x9a\xa2 '.decode('utf-8'), # boat
                  ]

HIT_OR_MISS_EMOJI = [
                    b'\xf0\x9f\x92\xa8 '.decode('utf-8'), # Whoosh
                    b'\xf0\x9f\x92\xa5 '.decode('utf-8'), # BOOM
                ]

BOATS = [
            ['Carrier',5],
            ['Battleship',4],
            ['Cruiser',3],
            ['Submarine',3],
            ['Destroyer',2 ],
        ]

NUM_FAKE_BOARDS = 1000

def render_board(board, target=None):
    ret = []
    for state in board.keys():
        ret.extend( [
            f'### STATE: {state} ###',
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
            orientation_lr = np.random.randint(2)
            ul_coord_r = np.random.randint(BOARD_SIZE - (boat_size-1 if orientation_lr else 0))
            ul_coord_c = np.random.randint(BOARD_SIZE - (boat_size-1 if not orientation_lr else 0))
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
    positions_to_reveal = np.random.choice(  BOARD_SIZE*BOARD_SIZE
                                            , np.random.randint(BOARD_SIZE*BOARD_SIZE*MAX_REVEAL_FRAC)
                                            , replace=False
                                        )
    for pos in positions_to_reveal:
        board['observed'][int(pos/BOARD_SIZE), pos%BOARD_SIZE] = board['hidden'][int(pos/BOARD_SIZE), pos%BOARD_SIZE]
    return board



def get_blast_features_given_board_and_rc(board, row_i, col_i):
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


def slice_up_board_into_blast_features(board):
    pixel_features = []
    for row_i in range(BOARD_SIZE):
        for col_i in range(BOARD_SIZE):
            pixel_features.append( get_blast_features_given_board_and_rc(board, row_i, col_i) )
    return pixel_features


def pick_next_blast_spot_to_target(board, blast_model):
    # First predict what pixels look good.
    preds_proba = blast_model.predict_proba(slice_up_board_into_blast_features(board))
    best_i = sorted(
                  zip(  range(BOARD_SIZE*BOARD_SIZE)
                      , preds_proba[:,1]
                      , (board['observed']==1).flatten()
                  )
                , key=lambda i: (not i[2], -i[1])
             )[0][0]
    return best_i


def get_blast_model():
    return RandomForestClassifier(n_estimators = 1000, max_depth = 10)

def fit_blast_model(  blast_model_path = ARGS_DEFAULT['--blast-model-path'],
                      num_boards = ARGS_DEFAULT['--num-boards'],
            ):
    blast_model = get_blast_model()
    # First get the boards
    boards = [ reveal_some_of_the_board(create_board()) for i in range(int(num_boards)) ]
    features = np.concatenate([
                    slice_up_board_into_blast_features(board)
                    for board in boards
                ])
    labels = np.concatenate([ board['hidden'].flatten() for board in boards ])
    blast_model.fit(features, labels)
    dump(blast_model, os.path.expanduser(blast_model_path))
    return blast_model




def get_place_model():
    return RandomForestRegressor()

def fit_place_model(    place_model_path = ARGS_DEFAULT['--place-model-path'],
                        blast_model_path = ARGS_DEFAULT['--blast-model-path'],
                        num_boards = ARGS_DEFAULT['--num-boards'],
                    ):
    # Train:
    #     For N random boards:
    #         Rows:
    #             Hidden board pixels
    #         Labels:
    #             How many shots AI took
    #     Plug matrix into Placement Model
    place_model = get_place_model()
    blast_model = load(os.path.expanduser(blast_model_path))
    # Maybe this will speed it up?
    blast_model.n_jobs = -1

    # Figure out what the background probabilty distribution is
    blast_probas = get_praba_distribution_from_blast_model(blast_model)

    boards = [ create_board() for i in range(int(num_boards)) ]
    #features = np.array( [b['hidden'].flatten() for b in boards] )
    features = np.array( [slice_up_board_into_place_features(b) for b in boards] )
    scores = np.array([ play_blast_game( blast_model, board, 0, False) for board in boards ])
    print("got here %s"%datetime.datetime.now())
    place_model.fit(features, scores)
    dump(place_model, os.path.expanduser(place_model_path))
    return place_model


def get_praba_distribution_from_blast_model(blast_model):
    " Figure out where the blast model likes shooting for it's first move "
    target_counts = np.full((BOARD_SIZE*BOARD_SIZE,), 1.0)
    for i in range(NUM_FAKE_BOARDS):
        fake_board = dict(observed = np.full((BOARD_SIZE,BOARD_SIZE), 1) )
        preds_proba = blast_model.predict_proba(slice_up_board_into_blast_features(fake_board))
        target_counts += preds_proba[:,1]
    return target_counts / sum(target_counts)


def slice_up_board_into_place_features(board):
    #FIXME: Add more features!
    h_board = board['hidden']
    # How many adjacent tiles have a boat?
    adjacency_sum = 0
    for row_i in range(BOARD_SIZE):
        for col_i in range(BOARD_SIZE):
            adjacency_sum += ( h_board[ row_i+1, col_i ] if (
                                            h_board[ row_i, col_i ]
                                        and row_i < BOARD_SIZE-1
                            ) else 0 )
            adjacency_sum += ( h_board[ row_i, col_i+1 ] if (
                                            h_board[ row_i, col_i ]
                                        and col_i < BOARD_SIZE-1
                            ) else 0 )
    # How many boats are one diagonal away?
    diagonal_sum = 0
    for row_i in range(BOARD_SIZE):
        for col_i in range(BOARD_SIZE):
            diagonal_sum += ( h_board[ row_i+1, col_i+1 ] if (
                                                    h_board[ row_i, col_i ]
                                                and row_i < BOARD_SIZE-1
                                                and col_i < BOARD_SIZE-1
                            ) else 0 )
            diagonal_sum += ( h_board[ row_i-1, col_i+1 ] if (
                                                    h_board[ row_i, col_i ]
                                                and row_i >= 1
                                                and col_i < BOARD_SIZE-1
                            ) else 0 )
    # Given the AI's first move probas, how does the initial board placement look?
    return [adjacency_sum/2, diagonal_sum/2]


def play_blast_game(    blast_model,
                        board,
                        print_delay= ARGS_DEFAULT['--print-delay'],
                        verbose = True
                    ):
    if verbose: print(render_board(board))
    for move_i in range(BOARD_SIZE*BOARD_SIZE):
        if verbose: print("######## Move %s ########"%move_i)
        # Use the model to pick a spot to target, hit it, then keep going
        target_pos = pick_next_blast_spot_to_target(board, blast_model)
        # Update board with targetted position
        board['observed'][int(target_pos/BOARD_SIZE), target_pos%BOARD_SIZE] = board['hidden'][int(target_pos/BOARD_SIZE), target_pos%BOARD_SIZE]
        if verbose: print(render_board(board, target_pos))
        # If we hit every boat:
        if sum(sum(board['observed']==2)) == sum([ b[1] for b  in BOATS ]):
            if verbose: print("All boats sunk!")
            return move_i
            break
        else:
            time.sleep(float(print_delay))

def main():
    args = docopt(__doc__)
    # Make sure we have the directory
    os.system('mkdir -p %s'%os.path.expanduser(os.path.dirname(args['--blast-model-path'])))
    if args['play']:
        assert os.path.exists(os.path.expanduser(args['--blast-model-path'])), "You have to train the model first. (Did you provide the right model path?)"
        assert fastnumbers.isfloat(args['--print-delay']), "Delay must be a float or int."
        blast_model = load(os.path.expanduser(args['--blast-model-path']))
        board = create_board()
        play_blast_game( blast_model, board, args['--print-delay'])
    elif args['train-blast']:
        fit_blast_model(args['--blast-model-path'], args['--num-boards'])
    elif args['train-place']:
        fit_place_model(args['--place-model-path'], args['--blast-model-path'], args['--num-boards'])
    else:
        sys.stderr.write("How did you get here???\n")
        sys.exit(1)


if __name__ == '__main__':
    main()
