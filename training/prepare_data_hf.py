# prepare_data_hf.py
import chess
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import random

def prepare_from_huggingface():
    print("Đang kết nối Hugging Face (Streaming mode)...")
    dataset = load_dataset("parquet", data_files={'train': 'training/dataset.parquet'}, split="train", streaming=True)
    
    data_entries = []
    
    MAX_GAMES_TO_PROCESS = 100000  
    
    print(f"Đang xử lý {MAX_GAMES_TO_PROCESS} ván cờ để tạo dữ liệu...")
    
    processed_games = 0
    
    for game in tqdm(dataset):
        winner = game.get('winner')
        moves_uci = game.get('moves_uci')
        
        if winner not in ['white', 'black']:
            continue
            
        label = 1.0 if winner == 'white' else -1.0
        
        board = chess.Board()
        
        n_moves = len(moves_uci)
        if n_moves < 10: continue
        
        for i, move_str in enumerate(moves_uci):
            try:
                move = chess.Move.from_uci(move_str)
                board.push(move)
            except:
                break
            
            if 8 < i < (n_moves - 5):
                if random.random() < 0.1:
                    fen = board.fen()
                    data_entries.append({'fen': fen, 'eval': label})
        
        processed_games += 1
        if processed_games >= MAX_GAMES_TO_PROCESS:
            break

    print(f"Đã trích xuất được {len(data_entries)} thế cờ từ {processed_games} ván.")
    
    df = pd.DataFrame(data_entries)
    output_path = 'training/dataset_large.csv'
    df.to_csv(output_path, index=False)
    print(f"Đã lưu file tại: {output_path}")

if __name__ == "__main__":
    prepare_from_huggingface()