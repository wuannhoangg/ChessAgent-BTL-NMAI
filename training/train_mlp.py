# training/train_mlp.py
import numpy as np
import pandas as pd
import chess
import sys
import os
import argparse
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import board_to_tensor
from training.model_mlp import ChessMLP_Scratch, xp


def parse_args():
    parser = argparse.ArgumentParser(description="Script huấn luyện / Fine-tune model Chess MLP")

    parser.add_argument(
        '--data',
        type=str,
        default='training/dataset_large.csv',
        help='Đường dẫn dataset gốc dùng để train từ đầu (base training data).'
    )
    
    parser.add_argument(
        '--finetune-data',
        type=str,
        default=None,
        help='(Tuỳ chọn) Đường dẫn dataset MỚI để fine-tune. Nếu có, script sẽ ưu tiên dùng nó thay vì --data.'
    )

    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help=(
            "Đường dẫn file .npz chứa trọng số cũ. "
            "Nếu được cung cấp, model sẽ KHÔNG khởi tạo từ ngẫu nhiên mà học tiếp từ đây "
            "(Transfer Learning / Resume Training)."
        )
    )
    
    parser.add_argument(
        '--output-path',
        type=str,
        default='training/best_model_mlp.npz',
        help='Đường dẫn file .npz để lưu model sau khi train xong mỗi epoch.'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Số vòng lặp qua toàn bộ dataset (mỗi vòng = 1 lượt duyệt hết data).'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Số mẫu được xử lý song song trong một bước cập nhật trọng số (mini-batch).'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='Learning Rate (tốc độ học). Khi Fine-tune trên dữ liệu nhỏ, nên giảm xuống (vd: 1e-3).'
    )

    return parser.parse_args()


def load_data_batch(df, start_idx, end_idx):
    batch_data = df.iloc[start_idx:end_idx]
    X_list = []
    y_list = []
    
    for _, row in batch_data.iterrows():
        board = chess.Board(row['fen'])
        tensor = board_to_tensor(board).numpy()
        X_list.append(tensor)
        y_list.append(float(row['eval']))
        
    X_batch = np.array(X_list)
    y_batch = np.array(y_list)
    
    return xp.asarray(X_batch), xp.asarray(y_batch)


def train(args):
    if args.finetune_data:
        current_data_path = args.finetune_data
        print(f"\n[INFO] Phát hiện chế độ FINETUNE. Đang sử dụng dữ liệu chuyên biệt: {current_data_path}")
    else:
        current_data_path = args.data
        print(f"\n[INFO] Đang sử dụng dữ liệu mặc định: {current_data_path}")

    print(f"--- TRAINING CONFIGURATION ---")
    print(f"- Backend      : {'GPU (CuPy) - Tốc độ cao' if xp.__name__ == 'cupy' else 'CPU (NumPy) - Tốc độ thường'}")
    print(f"- Epochs       : {args.epochs}")
    print(f"- Batch Size   : {args.batch_size}")
    print(f"- Learning Rate: {args.lr}")
    print(f"- Output Path  : {args.output_path}")
    
    if not os.path.exists(current_data_path):
        print(f"[LỖI CRITICAL] Không tìm thấy file dữ liệu: {current_data_path}")
        return

    df = pd.read_csv(current_data_path)
    print(f"• Đã nạp thành công {len(df)} mẫu dữ liệu từ {current_data_path}.")

    df = df.sample(frac=1).reset_index(drop=True)
    n_samples = len(df)
    
    model = ChessMLP_Scratch()

    if args.resume_from:
        if os.path.exists(args.resume_from):
            print(f"\n[INFO] Kích hoạt chế độ RESUME/FINETUNE.")
            print(f"       Đang nạp trọng số từ: {args.resume_from}")
            try:
                model.load_weights(args.resume_from)
                print("       >>> Load thành công! Model sẽ học tiếp dựa trên trọng số hiện có.")
            except Exception as e:
                print(f"[LỖI] File checkpoint bị lỗi cấu trúc: {e}. Dừng chương trình để tránh train trên model hỏng.")
                return 
        else:
            print(f"[LỖI] Không tìm thấy file checkpoint: {args.resume_from}")
            return
    else:
        print(f"\n[INFO] Train From Scratch. Model khởi tạo ngẫu nhiên.")

    print("\n--- BẮT ĐẦU TRAINING ---")
    for epoch in range(args.epochs):
        df = df.sample(frac=1).reset_index(drop=True)
        
        total_loss = 0.0
        n_batches = int(np.ceil(n_samples / args.batch_size))
        progress_bar = tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for i in progress_bar:
            start = i * args.batch_size
            end = min((i + 1) * args.batch_size, n_samples)
            
            X, y = load_data_batch(df, start, end)
            
            y_pred = model.forward(X)  # Shape: (1, batch_size)
            
            y_true = y.reshape(1, -1)

            loss = xp.mean((y_pred - y_true) ** 2)
            
            model.backward(y_pred, y)
            
            model.update(args.lr)
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch+1} hoàn tất. Loss trung bình: {avg_loss:.5f}")
        
        model.save_weights(args.output_path)
    
    print(f"\n Hoàn tất! Model đã được lưu an toàn tại: {args.output_path}")


if __name__ == '__main__':
    arguments = parse_args()
    train(arguments)
