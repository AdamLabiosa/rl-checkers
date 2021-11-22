#---------------------------------------------
# Representing Board State for Chess AI
# Created By: Adam Labiosa
# Inspration From: Jonathan Zia
# Last Edited: 2021
# University of Wisconsin - Madison
# ----------------------------------------------------
import tensorflow as tf
import numpy as np
import pieces as p
import random as r
import state as s
import time as t
import copy as c
import math
import os


def initialize_pieces(random=False, keep_prob=1.0):
    """Construct list of pieces as objects"""

    # Args: (1) random: Whether board is initialized to random initial state
    #		(2) keep_prob: Probability of retaining piece
    # Returns: Python list of pieces
    # 1,1 = a1 ... 8,8 = h8

    piece_list = [p.Piece('white', 1, 1), p.Piece('white', 3, 1), p.Piece('white', 5, 1), p.Piece('white', 7, 1),
                  p.Piece('white', 2, 2), p.Piece('white', 4, 2), p.Piece(
                      'white', 6, 2), p.Piece('white', 8, 2),
                  p.Piece('white', 1, 3), p.Piece('white', 3, 3), p.Piece(
                      'white', 5, 3), p.Piece('white', 7, 3),

                  p.Piece('black', 2, 8), p.Piece('black', 4, 8), p.Piece(
                      'black', 6, 8), p.Piece('black', 8, 8),
                  p.Piece('black', 1, 7), p.Piece('black', 3, 7), p.Piece(
                      'black', 5, 7), p.Piece('black', 7, 7),
                  p.Piece('black', 2, 6), p.Piece('black', 4, 6), p.Piece('black', 6, 6), p.Piece('black', 8, 6)]

    # If random is True, randomize piece positions and activity
    if random:
        # For piece in piece list...
        for piece in piece_list:
            # Toggle activity based on uniform distribution (AND PIECE IS NOT KING)
            if r.random() >= keep_prob:
                piece.remove()
            # If the piece was not removed, randomize file and rank
            else:
                newfile = r.randint(1, 8)
                newrank = r.randint(1, 8)

                # Is even
                if (newfile + newrank) % 2 == 0:
                    # If there is another piece in the target tile, swap places
                    for other_piece in piece_list:
                        if other_piece.is_active and other_piece.file == newfile and other_piece.rank == newrank:
                            # Swap places
                            other_piece.file = piece.file
                            other_piece.rank = piece.rank
                    # Else, and in the previous case, update the piece's file and rank
                    piece.file = newfile
                    piece.rank = newrank
                    piece.move_count += 1

    return piece_list


def board_state(piece_list):
    """Configuring inputs for value function network"""

    # Args: (1) piece list

    # The output contains M planes of dimensions (N X N) where (N X N) is the size of the board.
    # There are M planes "stacked" in layers where each layer represents a different "piece group"
    # (e.g. white pawns, black rooks, etc.) in one-hot format where 1 represents a piece in those
    # coordinates and 0 represents the piece is not in those coordinates.

    # Define parameters
    N = 8  # N = board dimensions (8 x 8)
    M = 2  # M = piece groups (6 per player)

    # Initializing board state with dimensions N x N x (MT + L)
    board = np.zeros((N, N, M))

    # The M layers each represent a different piece group. The order of is as follows:
    # 0: White Pieces
    # 0: Black Pieces
    # Note that the number of pieces in each category may change upon piece promotion or removal
    # (hence the code below will remain general).

    # Fill board state with pieces
    for piece in piece_list:
        # Place active white pawns in plane 0 and continue to next piece
        if piece.is_active and piece.color == 'white':
            # print(piece.name)
            board[piece.file - 1, piece.rank - 1, 0] = 1

        # Place active white knights in plane 1 and continue to next piece
        elif piece.is_active and piece.color == 'white':
            board[piece.file - 1, piece.rank - 1, 1] = 1

    # Return board state
    return board


def visualize_state(piece_list):
    """Visualizing board in terminal"""

    # Args: (1) piece list

    # The output is an 8x8 grid indicating the present locations for each piece

    # Initializing empty grid
    visualization = np.empty([8, 8], dtype=object)
    for i in range(0, 8):
        for j in range(0, 8):
            visualization[i, j] = ' '

    for piece in piece_list:
        # Load active pawns
        if piece.is_active and piece.color == 'white' and piece.name == 'Piece':
            visualization[piece.file - 1, piece.rank - 1] = 'P'

        elif piece.is_active and piece.color == 'black' and piece.name == 'Piece':
            visualization[piece.file - 1, piece.rank - 1] = 'p'

        elif piece.is_active and piece.color == 'white' and piece.name == 'King':
            visualization[piece.file - 1, piece.rank - 1] = 'K'

        elif piece.is_active and piece.color == 'black' and piece.name == 'King':
            visualization[piece.file - 1, piece.rank - 1] = 'k'

    # Return visualization
    return visualization


def action_space(piece_list, player):
    """Determining available moves for evaluation"""

    # Args: (1) piece list, (2) player color

    # The output is a P x 8 matrix where P is the number of pieces and 8 is the maximum
    # possible number of moves for any piece. For pieces which have less than  possible
    # moves, zeros are appended to the end of the row. A value of 1 indicates that a
    # move is available while a value of 0 means that it is not.

    # See pieces.py for move glossary

    # Initializing action space with dimensions P x 8
    action_space = np.zeros((12, 8))

    # For each piece...
    for i in range(0, 12):
        # If it is white's turn to move...
        if player == 'white':
            # Obtain vector of possible actions and write to corresponding row
            action_space[i, :] = piece_list[i].actions(piece_list)
        else:
            action_space[i, :] = piece_list[i + 12].actions(piece_list)

    # Return action space
    return action_space


def points(piece_list):
    """Calculating point differential for the given board state"""

    # Args: (1) piece list
    # Returns: differential (white points - black points)

    # The points are calculated via the standard chess value system:
    # Pawn = 1, King = 3, Bishop = 3, Rook = 5, Queen = 9
    # King = 100 (arbitrarily large)

    differential = 0
    # For all white pieces...
    for i in range(0, 12):
        # If the piece is active, add its points to the counter
        if piece_list[i].is_active:
            differential = differential + piece_list[i].value
    # For all black pieces...
    for i in range(12, 24):
        # If the piece is active, subtract its points from the counter
        if piece_list[i].is_active:
            differential = differential - piece_list[i].value

    # Return point differential
    return differential
