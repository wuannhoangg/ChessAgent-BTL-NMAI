# benchmark.py
import chess
import time
import sys
import os
from agents.random_agent import RandomAgent
from agents.minimax_agent import MinimaxAgent
try:
    from agents.mlp_agent import MLPAgent
    HAS_MLP_AGENT = True
except ImportError:
    HAS_MLP_AGENT = False


def play_single_game(agent_white, agent_black, game_id, quiet=True):
    board = chess.Board()
    moves_count = 0
    
    while not board.is_game_over():
        moves_count += 1
        
        if board.turn == chess.WHITE:
            move = agent_white.select_move(board)
        else:
            move = agent_black.select_move(board)
        
        if move is None:
            break
        
        board.push(move)
    
    outcome = board.outcome()
    result_str = "DRAW"
    winner = None
    
    if outcome and outcome.winner == chess.WHITE:
        result_str = "WHITE WIN"
        winner = chess.WHITE
    elif outcome and outcome.winner == chess.BLACK:
        result_str = "BLACK WIN"
        winner = chess.BLACK
    
    if not quiet:
        termination = outcome.termination.name if outcome else "UNKNOWN"
        print(f"Game {game_id}: {result_str} sau {moves_count} nước. ({termination})")
    
    return winner


def run_tournament(agent_white, agent_black, num_games, label_white="White", label_black="Black"):
    stats = {
        "white_wins": 0,
        "black_wins": 0,
        "draws": 0
    }
    
    print(f"\n>>> KHỞI ĐỘNG GIẢI ĐẤU ({num_games} VÁN) <<<")
    print(f"Phe Trắng: {label_white}")
    print(f"Phe Đen  : {label_black}")
    print("-" * 50)
    
    start_time = time.time()
    
    for i in range(1, num_games + 1):
        sys.stdout.write(f"\rĐang chạy ván {i}/{num_games} ...")
        sys.stdout.flush()
        
        # Chạy một ván dưới chế độ quiet để tối ưu tốc độ.
        winner = play_single_game(agent_white, agent_black, i, quiet=True)
        
        # Cập nhật thống kê
        if winner == chess.WHITE:
            stats["white_wins"] += 1
        elif winner == chess.BLACK:
            stats["black_wins"] += 1
        else:
            stats["draws"] += 1
            
    total_time = time.time() - start_time
    
    print(f"\n{'-' * 50}")
    print("GIẢI ĐẤU HOÀN TẤT!")
    print(f"Tổng thời gian: {total_time:.2f}s")
    avg_time = total_time / num_games if num_games > 0 else 0
    print(f"Tốc độ trung bình: {avg_time:.2f} s/ván")
    
    # Tính phần trăm tỉ lệ thắng/hòa
    if num_games > 0:
        win_rate_white = (stats['white_wins'] / num_games) * 100
        win_rate_black = (stats['black_wins'] / num_games) * 100
        draw_rate = (stats['draws'] / num_games) * 100
    else:
        win_rate_white = win_rate_black = draw_rate = 0.0
    
    print("\n" + "=" * 50)
    print("BẢNG XẾP HẠNG CHI TIẾT")
    print("=" * 50)
    print(f"TRẮNG ({label_white}): {stats['white_wins']} thắng \t({win_rate_white:.1f}%)")
    print(f"ĐEN   ({label_black}): {stats['black_wins']} thắng \t({win_rate_black:.1f}%)")
    print(f"HÒA                  : {stats['draws']} ván   \t({draw_rate:.1f}%)")
    print("=" * 50)
    
    if label_white.startswith("Minimax") and label_black == "Random":
        if win_rate_white >= 90:
            print("[KẾT LUẬN] Thuật toán Minimax ĐẠT yêu cầu (>90% win rate so với Random).")
        else:
            print(f"[KẾT LUẬN] Minimax chưa đạt mục tiêu (Hiện tại: {win_rate_white:.1f}%).")
            print("          Cần xem lại hàm đánh giá (heuristic) hoặc tăng depth.")
        
    if (label_white.startswith("ML Agent") or label_white.startswith("MLP Agent")) and label_black == "Random":
        if win_rate_white >= 60:
            print("[KẾT LUẬN] Mô hình MLP ĐẠT yêu cầu cơ bản (>60% win rate so với Random).")
        else:
            print(f"[KẾT LUẬN] MLP cần được train thêm (Hiện tại: {win_rate_white:.1f}%).")


if __name__ == "__main__":
    print("\n--- CHESS AI BENCHMARK TOOL ---")
    print("Công cụ kiểm thử hiệu năng và tỉ lệ thắng của các AI.")
    print("-" * 40)
    print("1. Minimax (Trắng) vs Random (Đen)")
    print("2. Random (Trắng) vs Minimax (Đen)")
    print("3. Minimax vs Minimax")
    
    menu_mode = "BASIC"
    
    if HAS_MLP_AGENT:
        menu_mode = "MLP_ONLY"
        print("-" * 40)
        print("4. MLP Agent vs Random")
        print("5. MLP Agent vs Minimax")
    
    choice = input("\n>>> Chọn cặp đấu (nhập số): ")
    
    try:
        num_games = int(input(">>> Số lượng ván muốn chạy (ví dụ: 10, 50, 100): "))
    except ValueError:
        num_games = 10
        print("Input không hợp lệ, mặc định chạy 10 ván.")

    minimax_depth = 3
    minimax_p1 = MinimaxAgent(depth=minimax_depth)
    minimax_p2 = MinimaxAgent(depth=minimax_depth)
    random_p = RandomAgent()
    
    def load_mlp():
        if not HAS_MLP_AGENT:
            return None
        model_file = 'training/best_model_mlp.npz'
        if os.path.exists(model_file):
            print(f"[INFO] Đang nạp model từ {model_file}...")
            return MLPAgent(model_path=model_file)
        print(f"[LỖI] Không tìm thấy file trọng số {model_file}. Vui lòng train trước.")
        return None
    
    if choice == '1':
        run_tournament(minimax_p1, random_p, num_games, f"Minimax(D{minimax_depth})", "Random")
    elif choice == '2':
        run_tournament(random_p, minimax_p2, num_games, "Random", f"Minimax(D{minimax_depth})")
    elif choice == '3':
        run_tournament(minimax_p2, minimax_p1, num_games, "Minimax A", "Minimax B")
            
    elif menu_mode == "MLP_ONLY":
        if choice == '4':
            agent = load_mlp()
            if agent:
                run_tournament(agent, random_p, num_games, "MLP Agent (Scratch)", "Random")
        elif choice == '5':
            agent = load_mlp()
            if agent:
                run_tournament(agent, minimax_p2, num_games, "MLP Agent (Scratch)", "Minimax")
        else:
            print("Lựa chọn không hợp lệ.")
    else:
        print("Lựa chọn không hợp lệ hoặc model MLP chưa sẵn sàng.")
