import random
import chess

class RandomAgent:
    def __init__(self):
        pass

    def select_move(self, board):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        return random.choice(legal_moves)