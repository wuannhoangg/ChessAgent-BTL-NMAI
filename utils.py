# utils.py
import chess
import numpy as np
import torch

PIECE_TO_INDEX = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}


def board_to_tensor(board):

    matrix = np.zeros((13, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        
        row = 7 - (square // 8)  
        col = square % 8         

        symbol = piece.symbol()  
        if symbol in PIECE_TO_INDEX:
            channel = PIECE_TO_INDEX[symbol]
            matrix[channel][row][col] = 1.0

    if board.turn == chess.WHITE:
        matrix[12, :, :] = 1.0
    else:
        matrix[12, :, :] = -1.0

    return torch.from_numpy(matrix)
