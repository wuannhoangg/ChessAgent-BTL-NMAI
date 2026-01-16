# agents/mlp_agent.py
import chess
import random
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import board_to_tensor

from training.model_mlp import ChessMLP_Scratch, xp 


class MLPAgent:
    def __init__(self, model_path='training/best_model_mlp.npz'):

        self.device_name = "GPU (CuPy)" if xp.__name__ == 'cupy' else "CPU (NumPy)"
        print(f"[MLPAgent] Init using backend: {self.device_name}")
        
        self.model = ChessMLP_Scratch()
        
        try:
            if os.path.exists(model_path):
                self.model.load_weights(model_path)
                print(f"[MLPAgent] Đã load weights thành công từ {model_path}")
            else:
                print(f"[MLPAgent] CẢNH BÁO: Không tìm thấy file {model_path}")
                self.model = None
        except Exception as e:
            print(f"[MLPAgent] LỖI load model: {e}")
            self.model = None

    def select_move(self, board):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        for move in legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return move
            board.pop()

        if self.model is None:
            return random.choice(legal_moves)

        potential_boards = []  # danh sách tensor tương ứng các trạng thái sau nước đi
        move_list = []         # danh sách nước đi hợp lệ

        for move in legal_moves:
            board.push(move)

            tensor = board_to_tensor(board).numpy()
            potential_boards.append(tensor)
            move_list.append(move)

            board.pop()

        batch_input = xp.asarray(np.array(potential_boards))

        output = self.model.forward(batch_input)

        if xp.__name__ == 'cupy':
            scores = xp.asnumpy(output).flatten()
        else:
            scores = output.flatten()

        if board.turn == chess.WHITE:
            for i, move in enumerate(move_list):
                board.push(move)
                if board.is_stalemate() or board.can_claim_draw():
                    scores[i] -= 0.5
                board.pop()
            best_idx = scores.argmax()

        else:
            for i, move in enumerate(move_list):
                board.push(move)
                if board.is_stalemate() or board.can_claim_draw():
                    scores[i] += 0.5
                board.pop()
            best_idx = scores.argmin()

        return move_list[best_idx]