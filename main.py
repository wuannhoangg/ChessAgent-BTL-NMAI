# main.py
import chess
import time
import os
import sys
from agents.random_agent import RandomAgent
from agents.minimax_agent import MinimaxAgent

try:
    from agents.mlp_agent import MLPAgent
    HAS_MLP_SCRATCH = True
except ImportError:
    HAS_MLP_SCRATCH = False
    print("[CẢNH BÁO] Không tìm thấy file agents/mlp_agent.py. Chế độ MLP sẽ bị vô hiệu hóa.")


def play_game(agent_white, agent_black, pause_time=0.1):
    
    board = chess.Board()
    
    print("\n" + "="*40)
    print(" -----START----- ")
    print("="*40 + "\n")
    print(board)
    print("\n")

    move_count = 0
    
    while not board.is_game_over():
        move_count += 1
        
        is_white_turn = board.turn
        current_agent = agent_white if is_white_turn else agent_black
        player_name = "WHITE (Trắng)" if is_white_turn else "BLACK (Đen)"

        print(f"[{move_count}] {player_name} đang tính toán...")
        
        start_time = time.time()
        
        try:
            move = current_agent.select_move(board)
        except Exception as e:
            print(f"[LỖI CRITICAL] Agent {player_name} gặp sự cố: {e}")
            import traceback
            traceback.print_exc()
            break
            
        end_time = time.time()
        if move is None:
            print(f"GAME OVER: {player_name} chịu thua hoặc không trả về nước đi hợp lệ.")
            break

        if move in board.legal_moves:
            board.push(move)
            print(f">>> {player_name} đi: {move} (Mất {end_time - start_time:.4f}s)")
            print(board)
            print("-" * 40)
        else:
            print(f"[LỖI] Nước đi {move} sai luật! {player_name} bị xử thua (Disqualified).")
            break
            
        time.sleep(pause_time)

    print("\n" + "="*40)
    print("RESULT")
    print("="*40)
    
    result = board.outcome()
    if result:
        print(f"Lý do kết thúc: {result.termination.name}")
        winner = result.winner
        
        if winner == chess.WHITE:
            print("NGƯỜI THẮNG: TRẮNG (WHITE)")
        elif winner == chess.BLACK:
            print("NGƯỜI THẮNG: ĐEN (BLACK)")
        else:
            print("KẾT QUẢ: HÒA (DRAW)")
    else:
        print("Game kết thúc bất thường (Crash hoặc lỗi logic).")


if __name__ == "__main__":

    print("\n--- CHỌN CHẾ ĐỘ ĐẤU (MATCHMAKING) ---")
    print("1. Minimax (Trắng) vs Random (Đen)")
    print("2. Random (Trắng)  vs Minimax (Đen)")
    print("3. Minimax (Trắng) vs Minimax (Đen)")
    print("-" * 35)
    
    if HAS_MLP_SCRATCH:
        print("4. MLP (Trắng) vs Random (Đen)")
        print("5. MLP (Trắng) vs Minimax (Đen)")
    
    choice = input(">>> Nhập số lựa chọn của bạn: ")
    
    random_player = RandomAgent()
    minimax_player = MinimaxAgent(depth=3)
    
    white_player = None
    black_player = None

    if choice == '1':
        white_player = minimax_player
        black_player = random_player
    elif choice == '2':
        white_player = random_player
        black_player = minimax_player
    elif choice == '3':
        white_player = minimax_player
        black_player = minimax_player
    
    elif choice in ['4', '5'] and HAS_MLP_SCRATCH:
        model_path = 'training/best_model_mlp.npz'
        
        if os.path.exists(model_path):
            print(f"[INFO] Đang khởi động AI từ file: {model_path}...")
            mlp_player = MLPAgent(model_path=model_path) 
            
            if choice == '4':
                white_player = mlp_player
                black_player = random_player
            else:
                white_player = mlp_player
                black_player = minimax_player
        else:
            print(f"[LỖI CRITICAL] Không tìm thấy file trọng số '{model_path}'.")
            print("Vui lòng chạy lệnh: 'python training/train_mlp.py' trước để huấn luyện AI.")
            sys.exit()
    else:
        print("[LỖI] Lựa chọn không hợp lệ hoặc thiếu thư viện hỗ trợ.")
        sys.exit()

    if white_player and black_player:
        play_game(white_player, black_player)
