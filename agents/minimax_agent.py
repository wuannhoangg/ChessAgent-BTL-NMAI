import chess
import math

class MinimaxAgent:
    def __init__(self, depth=3):
        self.depth = depth
        
        self.mg_value = {
            chess.PAWN: 82, chess.KNIGHT: 337, chess.BISHOP: 365,
            chess.ROOK: 477, chess.QUEEN: 1025, chess.KING: 0
        }
        self.eg_value = {
            chess.PAWN: 94, chess.KNIGHT: 281, chess.BISHOP: 297,
            chess.ROOK: 512, chess.QUEEN: 936, chess.KING: 0
        }

        self.phase_weights = {
            chess.KNIGHT: 1,
            chess.BISHOP: 1,
            chess.ROOK: 2,
            chess.QUEEN: 4
        }
        self.total_phase = 24

        ######### Bảng điểm trung cuộc ##################
        # Tốt MG:
        # - Khuyến khích chiếm/giữ trung tâm (d4, e4, d5, e5)
        # - Hàng 2 (ban đầu) có bonus nhẹ để hỗ trợ phát triển
        # - Các ô quá cao ở MG chưa cần đẩy mạnh nên không thưởng lớn
        self.mg_pawn = [
             0,  0,  0,  0,  0,  0,  0,  0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
             5,  5, 10, 25, 25, 10,  5,  5,
             0,  0,  0, 20, 20,  0,  0,  0,
             5, -5,-10,  0,  0,-10, -5,  5,
             5, 10, 10,-20,-20, 10, 10,  5,
             0,  0,  0,  0,  0,  0,  0,  0
        ]
        
        # Mã MG:
        # - Bonus lớn ở trung tâm vì nhiều nước đi nhất và kiểm soát tốt ô quan trọng
        # - Phạt ở biên/góc vì bị hạn chế tầm hoạt động
        self.mg_knight = [
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
        ]

        # Tượng MG:
        # - Tránh góc/bị kẹt
        # - Ưu tiên ô mở ra đường chéo dài, kiểm soát trung tâm từ xa
        self.mg_bishop = [
            -20,-10,-10,-10,-10,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5, 10, 10,  5,  0,-10,
            -10,  5,  5, 10, 10,  5,  5,-10,
            -10,  0, 10, 10, 10, 10,  0,-10,
            -10, 10, 10, 10, 10, 10, 10,-10,
            -10,  5,  0,  0,  0,  0,  5,-10,
            -20,-10,-10,-10,-10,-10,-10,-20
        ]

        # Xe MG:
        # - PST ít ảnh hưởng hơn vì Xe phụ thuộc mạnh vào cấu trúc động:
        #   cột mở, hàng 7, chiếm file.
        # - Bảng chỉ thưởng nhẹ cho hàng 2 (tiếp cận hàng 7 địch)
        self.mg_rook = [
             0,  0,  0,  0,  0,  0,  0,  0,
             5, 10, 10, 10, 10, 10, 10,  5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
             0,  0,  0,  5,  5,  0,  0,  0
        ]

        # Hậu MG:
        # - Phạt nhẹ việc ra quá sớm (hàng 1-2) vì dễ bị tempo đuổi
        # - Thưởng nhẹ trung tâm khi đã phát triển
        self.mg_queen = [
            -20,-10,-10, -5, -5,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5,  5,  5,  5,  0,-10,
             -5,  0,  5,  5,  5,  5,  0, -5,
              0,  0,  5,  5,  5,  5,  0, -5,
            -10,  5,  5,  5,  5,  5,  0,-10,
            -10,  0,  5,  0,  0,  0,  0,-10,
            -20,-10,-10, -5, -5,-10,-10,-20
        ]

        # Vua MG:
        # - Mục tiêu trung cuộc là an toàn: nhập thành, tránh trung tâm.
        # - Vua ở trung tâm bị phạt rất nặng.
        # - Hàng 1 góc (g1/c1) được thưởng do an toàn hơn.
        self.mg_king = [
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -20,-30,-30,-40,-40,-30,-30,-20,
            -10,-20,-20,-20,-20,-20,-20,-10,
             20, 20,  0,  0,  0,  0, 20, 20,
             20, 30, 10,  0,  0, 10, 30, 20
        ]

        self.eg_pawn = [
             0,  0,  0,  0,  0,  0,  0,  0,
            80, 80, 80, 80, 80, 80, 80, 80,
            50, 50, 50, 50, 50, 50, 50, 50,
            30, 30, 30, 30, 30, 30, 30, 30,
            20, 20, 20, 20, 20, 20, 20, 20,
            10, 10, 10, 10, 10, 10, 10, 10,
            10, 10, 10, 10, 10, 10, 10, 10,
             0,  0,  0,  0,  0,  0,  0,  0
        ]
        

        ######### Bảng điểm tàn cuộc ##################
        # Mã EG:
        # - Thường không thay đổi nhiều so với MG (vẫn là quân tầm ngắn)
        self.eg_knight = self.mg_knight
        
        # Tượng EG:
        # - Dùng chung MG vì nguyên tắc đường chéo dài vẫn giữ.
        self.eg_bishop = self.mg_bishop
        
        # Xe EG:
        # - Cũng phụ thuộc mạnh vào file/hàng mở nên PST giữ gần MG.
        self.eg_rook = self.mg_rook
        
        # Hậu EG:
        # - Ít bị phạt ra sớm nữa.
        # - Thưởng rõ hơn ở trung tâm để phản ánh vai trò “quân đa năng”.
        self.eg_queen = [
            -10,-10,-10,-10,-10,-10,-10,-10,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5,  5,  5,  5,  0,-10,
            -10,  0,  5, 10, 10,  5,  0,-10,
            -10,  0,  5, 10, 10,  5,  0,-10,
            -10,  0,  5,  5,  5,  5,  0,-10,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,-10,-10,-10,-10,-10,-10,-10
        ]
        
        # Vua EG:
        # - Nguyên tắc tàn cuộc: Vua phải chủ động, đi ra giữa.
        # - Trung tâm có bonus lớn vì giúp:
        #   + cản tốt đối phương
        #   + hộ tống tốt phong cấp
        #   + chiếm không gian
        self.eg_king = [
            -50,-40,-30,-20,-20,-30,-40,-50,
            -30,-20,-10,  0,  0,-10,-20,-30,
            -30,-10, 20, 30, 30, 20,-10,-30,
            -30,-10, 30, 40, 40, 30,-10,-30,
            -30,-10, 30, 40, 40, 30,-10,-30,
            -30,-10, 20, 30, 30, 20,-10,-30,
            -30,-30,  0,  0,  0,  0,-30,-30,
            -50,-30,-30,-30,-30,-30,-30,-50
        ]
        
        self.tables = {
            'MG': {
                chess.PAWN: self.mg_pawn,
                chess.KNIGHT: self.mg_knight,
                chess.BISHOP: self.mg_bishop,
                chess.ROOK: self.mg_rook,
                chess.QUEEN: self.mg_queen,
                chess.KING: self.mg_king
            },
            'EG': {
                chess.PAWN: self.eg_pawn,
                chess.KNIGHT: self.eg_knight,
                chess.BISHOP: self.eg_bishop,
                chess.ROOK: self.eg_rook,
                chess.QUEEN: self.eg_queen,
                chess.KING: self.eg_king
            }
        }

    def select_move(self, board):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        best_move = None
        alpha = -math.inf
        beta = math.inf

        is_white_turn = board.turn
        best_value = -math.inf if is_white_turn else math.inf

        for move in legal_moves:
            board.push(move)
            value = self.minimax(board, self.depth - 1, alpha, beta)
            board.pop()

            if is_white_turn:
                if value > best_value:
                    best_value = value
                    best_move = move
                alpha = max(alpha, best_value)
            else:
                if value < best_value:
                    best_value = value
                    best_move = move
                beta = min(beta, best_value)

        return best_move

    def minimax(self, board, depth, alpha, beta):
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board)

        is_maximizing = board.turn
        legal_moves = list(board.legal_moves)

        if is_maximizing:
            max_eval = -math.inf
            for move in legal_moves:
                board.push(move)
                eval_val = self.minimax(board, depth - 1, alpha, beta)
                board.pop()

                max_eval = max(max_eval, eval_val)
                alpha = max(alpha, eval_val)

                if beta <= alpha:
                    break

            return max_eval

        else:
            min_eval = math.inf
            for move in legal_moves:
                board.push(move)
                eval_val = self.minimax(board, depth - 1, alpha, beta)
                board.pop()

                min_eval = min(min_eval, eval_val)
                beta = min(beta, eval_val)

                if beta <= alpha:
                    break

            return min_eval

    def evaluate_board(self, board):
        if board.is_checkmate():
            return 99999 if board.outcome().winner == chess.WHITE else -99999

        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        mg_score = 0
        eg_score = 0
        current_phase = 0

        for i in range(64):
            piece = board.piece_at(i)
            if not piece:
                continue

            ptype = piece.piece_type
            color = piece.color

            if ptype in self.phase_weights:
                current_phase += self.phase_weights[ptype]

            mg_val = self.mg_value[ptype]
            eg_val = self.eg_value[ptype]

            if color == chess.WHITE:
                mg_pst = self.tables['MG'][ptype][i]
                eg_pst = self.tables['EG'][ptype][i]
            else:
                mi = chess.square_mirror(i)
                mg_pst = self.tables['MG'][ptype][mi]
                eg_pst = self.tables['EG'][ptype][mi]

            if color == chess.WHITE:
                mg_score += (mg_val + mg_pst)
                eg_score += (eg_val + eg_pst)
            else:
                mg_score -= (mg_val + mg_pst)
                eg_score -= (eg_val + eg_pst)

        current_phase = min(current_phase, self.total_phase)

        mg_weight = current_phase
        eg_weight = self.total_phase - current_phase

        final_score = (mg_score * mg_weight + eg_score * eg_weight) / self.total_phase

        return int(final_score)
