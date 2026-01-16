import pygame
import chess
import sys
import os
import threading
import time
import requests
from agents.random_agent import RandomAgent
from agents.minimax_agent import MinimaxAgent

try:
    from agents.mlp_agent import MLPAgent
    HAS_MLP = True
except ImportError:
    HAS_MLP = False

WIDTH, HEIGHT = 640, 640
DIMENSION = 8
SQ_SIZE = HEIGHT // DIMENSION
MAX_FPS = 60

COLOR_BOARD_LIGHT = (240, 217, 181)
COLOR_BOARD_DARK  = (181, 136, 99)
COLOR_HIGHLIGHT   = (100, 240, 100, 100)
COLOR_SELECTED    = (255, 255, 0, 100)
COLOR_HINT_DOT    = (100, 100, 100, 100)

# Asset Management
ASSET_DIR = "assets"
IMAGES = {}

def download_images():
    if not os.path.exists(ASSET_DIR):
        os.makedirs(ASSET_DIR)
        print("[INFO] Đang khởi tạo thư mục assets...")

    base_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/"
    
    piece_urls = {
        'wp': "4/45/Chess_plt45.svg/60px-Chess_plt45.svg.png",
        'wR': "7/72/Chess_rlt45.svg/60px-Chess_rlt45.svg.png",
        'wN': "7/70/Chess_nlt45.svg/60px-Chess_nlt45.svg.png",
        'wB': "b/b1/Chess_blt45.svg/60px-Chess_blt45.svg.png",
        'wQ': "1/15/Chess_qlt45.svg/60px-Chess_qlt45.svg.png",
        'wK': "4/42/Chess_klt45.svg/60px-Chess_klt45.svg.png",
        'bp': "c/c7/Chess_pdt45.svg/60px-Chess_pdt45.svg.png",
        'bR': "f/ff/Chess_rdt45.svg/60px-Chess_rdt45.svg.png",
        'bN': "e/ef/Chess_ndt45.svg/60px-Chess_ndt45.svg.png",
        'bB': "9/98/Chess_bdt45.svg/60px-Chess_bdt45.svg.png",
        'bQ': "4/47/Chess_qdt45.svg/60px-Chess_qdt45.svg.png",
        'bK': "f/f0/Chess_kdt45.svg/60px-Chess_kdt45.svg.png",
    }

    pieces = list(piece_urls.keys())
    headers = {'User-Agent': 'Mozilla/5.0'} # Giả lập trình duyệt để không bị chặn

    missing_files = [p for p in pieces if not os.path.exists(f"{ASSET_DIR}/{p}.png")]
    
    if missing_files:
        print(f"[INFO] Đang tải {len(missing_files)} ảnh quân cờ...")
        for p in missing_files:
            url = base_url + piece_urls[p]
            try:
                r = requests.get(url, headers=headers, timeout=10)
                if r.status_code == 200:
                    with open(f"{ASSET_DIR}/{p}.png", 'wb') as f:
                        f.write(r.content)
                else:
                    print(f"[LỖI TẢI] {p}: Status {r.status_code}")
            except Exception as e:
                print(f"[LỖI KẾT NỐI] {p}: {e}")

def load_images():
    pieces = ['wp', 'wR', 'wN', 'wB', 'wQ', 'wK', 'bp', 'bR', 'bN', 'bB', 'bQ', 'bK']
    font_fallback = pygame.font.SysFont('Arial', 30, bold=True)

    for piece in pieces:
        path = f"{ASSET_DIR}/{piece}.png"
        try:
            if os.path.exists(path):
                image = pygame.image.load(path)
                IMAGES[piece] = pygame.transform.smoothscale(image, (SQ_SIZE, SQ_SIZE))
            else:
                raise FileNotFoundError(f"Không tìm thấy {path}")
        except Exception as e:
            print(f"[CẢNH BÁO] Không load được ảnh {piece} ({e}). Dùng text thay thế.")
            
            if os.path.exists(path):
                try: os.remove(path)
                except: pass

            surf = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
            
            color = (255, 255, 255) if piece[0] == 'w' else (0, 0, 0)
        
            text_render = font_fallback.render(piece[1], True, color) # Lấy ký tự quân (P, K, Q...)
            
            rect = text_render.get_rect(center=(SQ_SIZE//2, SQ_SIZE//2))
            
            bg_color = (200, 200, 200, 150) if piece[0] == 'w' else (50, 50, 50, 150)
            pygame.draw.circle(surf, bg_color, (SQ_SIZE//2, SQ_SIZE//2), SQ_SIZE//2 - 5)
            
            surf.blit(text_render, rect)
            IMAGES[piece] = surf

class ChessGameGUI:
    def __init__(self, screen):
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 24, bold=True)
        self.font_small = pygame.font.SysFont('Arial', 18)
        
        # Trạng thái bàn cờ
        self.board = chess.Board()

        self.last_game_moves = []
        self.player_white_type = "Human"
        self.player_black_type = "Random"
        
        # Agents Instances
        self.agents = {
            "Random": RandomAgent(),
            "Minimax": MinimaxAgent(depth=3),
            "MLP": None
        }
        if HAS_MLP and os.path.exists('training/best_model_mlp.npz'):
            try:
                self.agents["MLP"] = MLPAgent(model_path='training/best_model_mlp.npz')
            except:
                print("[GUI] Lỗi load MLP Agent")

        # Trạng thái UI
        self.selected_square = None 
        self.valid_moves = []
        self.last_move = None
        self.game_over = False
        self.result_text = ""
        
        # Xử lý đa luồng cho AI
        self.ai_thinking = False
        self.ai_thread = None

        self.ai_move_result = None

    def reset_game(self):
        self.board = chess.Board()
        self.selected_square = None
        self.valid_moves = []
        self.last_move = None
        self.game_over = False
        self.result_text = ""
        self.ai_thinking = False
        self.ai_move_result = None

    def draw_board_background(self):
        colors = [COLOR_BOARD_LIGHT, COLOR_BOARD_DARK]
        for r in range(DIMENSION):
            for c in range(DIMENSION):
                color = colors[(r + c) % 2]
                pygame.draw.rect(self.screen, color, pygame.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))

    def draw_highlights(self):
        if self.last_move:
            s = pygame.Surface((SQ_SIZE, SQ_SIZE))
            s.set_alpha(100) # Transparency
            s.fill(pygame.Color('yellow'))
            
            # Convert chess index (0=A1) to pygame row/col
            # Pygame (0,0) is top-left. Chess A1 is bottom-left.
            # Row pygame = 7 - (square // 8)
            
            start_sq = self.last_move.from_square
            end_sq = self.last_move.to_square
            
            for sq in [start_sq, end_sq]:
                r, c = 7 - (sq // 8), sq % 8
                self.screen.blit(s, (c*SQ_SIZE, r*SQ_SIZE))

        if self.selected_square is not None:
            r, c = 7 - (self.selected_square // 8), self.selected_square % 8
            s = pygame.Surface((SQ_SIZE, SQ_SIZE))
            s.set_alpha(150)
            s.fill(pygame.Color('blue'))
            self.screen.blit(s, (c*SQ_SIZE, r*SQ_SIZE))

        for move in self.valid_moves:
            target_sq = move.to_square
            r, c = 7 - (target_sq // 8), target_sq % 8
            
            center = (c*SQ_SIZE + SQ_SIZE//2, r*SQ_SIZE + SQ_SIZE//2)
            radius = SQ_SIZE // 6
            
            if self.board.piece_at(target_sq):
                pygame.draw.circle(self.screen, (200, 50, 50, 150), center, radius)
            else:
                pygame.draw.circle(self.screen, (100, 100, 100, 150), center, radius)

    def draw_pieces(self):
        for square in range(64):
            piece = self.board.piece_at(square)
            if piece:
                r, c = 7 - (square // 8), square % 8
                color_char = 'w' if piece.color == chess.WHITE else 'b'
                piece_key = color_char + piece.symbol().upper() if piece.piece_type != chess.PAWN else color_char + 'p'
                map_symbol = {
                    'P': 'p', 'N': 'N', 'B': 'B', 'R': 'R', 'Q': 'Q', 'K': 'K'
                }
                piece_key = color_char + map_symbol[piece.symbol().upper()]
                
                if piece_key in IMAGES:
                    self.screen.blit(IMAGES[piece_key], pygame.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))

    def draw_text_overlay(self, text):
        s = pygame.Surface((WIDTH, HEIGHT))
        s.set_alpha(180)
        s.fill((0,0,0))
        self.screen.blit(s, (0,0))
        
        txt_surf = self.font.render(text, True, (255, 255, 255))
        txt_rect = txt_surf.get_rect(center=(WIDTH//2, HEIGHT//2))
        self.screen.blit(txt_surf, txt_rect)
        
        hint_surf = self.font_small.render("Press R to reload, ESC to return to menu", True, (200, 200, 200))
        hint_rect = hint_surf.get_rect(center=(WIDTH//2, HEIGHT//2 + 40))
        self.screen.blit(hint_surf, hint_rect)

    def get_ai_move_thread(self, agent):
        try:
            time.sleep(0.1) 
            move = agent.select_move(self.board.copy())
            self.ai_move_result = move 
        except Exception as e:
            print(f"\n[CRITICAL AI ERROR]: {e}")
            import traceback
            traceback.print_exc()
            self.ai_move_result = False
        finally:
            self.ai_thinking = False

    def trigger_ai_turn(self, agent_type):
        if self.ai_thinking: return
        
        agent = self.agents.get(agent_type)
        if not agent:
            print("Agent not found/loaded")
            return

        self.ai_thinking = True
        self.ai_move_result = None
        # Tạo luồng mới
        t = threading.Thread(target=self.get_ai_move_thread, args=(agent,))
        t.start()

    def run_game(self):
        running = True
        while running:
            # 1. Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return # Quay về menu
                    if event.key == pygame.K_r:
                        self.reset_game()

                if not self.game_over and not self.ai_thinking:
                    current_player_type = self.player_white_type if self.board.turn == chess.WHITE else self.player_black_type
                    
                    if current_player_type == "Human" and event.type == pygame.MOUSEBUTTONDOWN:
                        pos = pygame.mouse.get_pos()
                        c, r = pos[0] // SQ_SIZE, pos[1] // SQ_SIZE
                        # Convert Pygame coords to Chess Square Index
                        clicked_sq = (7 - r) * 8 + c
                        
                        if self.selected_square is None:
                            # Nếu click vào quân mình -> Chọn
                            piece = self.board.piece_at(clicked_sq)
                            if piece and piece.color == self.board.turn:
                                self.selected_square = clicked_sq
                                self.valid_moves = [m for m in self.board.legal_moves if m.from_square == clicked_sq]
                        else:
                            # Nếu đã chọn quân -> Click lần 2 là đi
                            # Tạo nước đi
                            move = chess.Move(self.selected_square, clicked_sq)
                            
                            # Xử lý phong cấp (Promotion): Mặc định phong Hậu
                            # Nếu quân là tốt, đi xuống hàng cuối -> Thêm flag promotion
                            piece = self.board.piece_at(self.selected_square)
                            if piece.piece_type == chess.PAWN:
                                if (piece.color == chess.WHITE and r == 0) or (piece.color == chess.BLACK and r == 7):
                                    move = chess.Move(self.selected_square, clicked_sq, promotion=chess.QUEEN)

                            # Kiểm tra hợp lệ
                            if move in self.board.legal_moves:
                                self.board.push(move)
                                self.last_move = move
                                self.selected_square = None
                                self.valid_moves = []
                            else:
                                # Nếu click vào quân khác của mình -> Đổi lựa chọn
                                piece = self.board.piece_at(clicked_sq)
                                if piece and piece.color == self.board.turn:
                                    self.selected_square = clicked_sq
                                    self.valid_moves = [m for m in self.board.legal_moves if m.from_square == clicked_sq]
                                else:
                                    # Click ra ngoài -> Bỏ chọn
                                    self.selected_square = None
                                    self.valid_moves = []

            # 2. Game Logic Check
            if self.board.is_game_over():
                self.game_over = True
                outcome = self.board.outcome()
                if outcome.winner == chess.WHITE: self.result_text = "White WIN!"
                elif outcome.winner == chess.BLACK: self.result_text = "Black WIN!"
                else: self.result_text = "DRAW!"

                self.last_game_moves = list(self.board.move_stack)
            
            # 3. AI Logic
            if not self.game_over:
                current_turn_white = (self.board.turn == chess.WHITE)
                current_type = self.player_white_type if current_turn_white else self.player_black_type
                
                if current_type != "Human":
                    if not self.ai_thinking and self.ai_move_result is None:
                        self.trigger_ai_turn(current_type)

                    if not self.ai_thinking and self.ai_move_result is not None:
                        move = self.ai_move_result

                        if move:
                            self.board.push(move)
                            self.last_move = move
                            self.ai_move_result = None
                        else:
                            print("AI gặp lỗi hoặc đầu hàng!")
                            self.game_over = True
                            self.result_text = "Error with AI / AI LOSE"


            # 4. Render
            self.draw_board_background()
            self.draw_highlights()
            self.draw_pieces()
            
            if self.game_over:
                self.draw_text_overlay(self.result_text)
                
            pygame.display.flip()
            self.clock.tick(MAX_FPS)

    def replay_last_game(self):
        if not self.last_game_moves:
            return

        temp_board = chess.Board()
        move_index = -1
        autoplay = True
        autoplay_delay = 0.7
        last_step_time = time.time()

        real_board = self.board
        real_last_move = self.last_move

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        autoplay = not autoplay
                    elif event.key == pygame.K_r:
                        temp_board = chess.Board()
                        move_index = -1
                        self.last_move = None
                    elif event.key == pygame.K_RIGHT:
                        if move_index < len(self.last_game_moves) - 1:
                            move_index += 1
                            temp_board.push(self.last_game_moves[move_index])
                    elif event.key == pygame.K_LEFT:
                        if move_index >= 0:
                            temp_board.pop()
                            move_index -= 1

            if autoplay and move_index < len(self.last_game_moves) - 1:
                now = time.time()
                if now - last_step_time >= autoplay_delay:
                    move_index += 1
                    temp_board.push(self.last_game_moves[move_index])
                    last_step_time = now

            self.last_move = self.last_game_moves[move_index] if move_index >= 0 else None

            self.board = temp_board

            self.draw_board_background()
            self.draw_highlights()
            self.draw_pieces()

            bg_height = 60
            overlay = pygame.Surface((WIDTH, bg_height))
            overlay.set_alpha(150)                    
            overlay.fill((255, 255, 255))
            self.screen.blit(overlay, (0, 0))

            title_surf = self.font.render("REPLAY MODE", True, (0, 0, 0))  
            hint_surf = self.font_small.render(
                "SPACE: Pause/Play  |  \u2190 / \u2192: Step Back/Forward  |  R: Restart  |  ESC: Exit",
                True,
                (0, 0, 0)  
            )

            self.screen.blit(title_surf, (10, 5))
            self.screen.blit(hint_surf, (10, 32))


            pygame.display.flip()
            self.clock.tick(60)

        self.board = real_board
        self.last_move = real_last_move


    def run_menu(self):
        options = ["Human", "Random", "Minimax", "MLP"] if HAS_MLP else ["Human", "Random", "Minimax"]
        
        while True:
            self.screen.fill((30, 30, 30))
            
            title = self.font.render("CHESS WITH AI", True, (255, 215, 0))
            self.screen.blit(title, (WIDTH//2 - title.get_width()//2, 50))
            
            info_w = self.font_small.render(f"White (Press 1 to choose): {self.player_white_type}", True, (200, 200, 200))
            info_b = self.font_small.render(f"Black (Press 2 to choose): {self.player_black_type}", True, (200, 200, 200))
            start_txt = self.font.render("Press Enter to start", True, (0, 255, 0))

            if self.last_game_moves:
                replay_txt = self.font_small.render("Press R to replay last game", True, (173, 216, 230))
            else:
                replay_txt = self.font_small.render("No game history yet", True, (100, 100, 100))
            
            self.screen.blit(info_w, (WIDTH//2 - info_w.get_width()//2, 200))
            self.screen.blit(info_b, (WIDTH//2 - info_b.get_width()//2, 250))
            self.screen.blit(start_txt, (WIDTH//2 - start_txt.get_width()//2, 400))
            self.screen.blit(replay_txt, (WIDTH//2 - replay_txt.get_width()//2, 440))
            
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        self.reset_game()
                        self.run_game()
                    
                    if event.key == pygame.K_1:
                        idx = options.index(self.player_white_type)
                        self.player_white_type = options[(idx + 1) % len(options)]
                    if event.key == pygame.K_2:
                        idx = options.index(self.player_black_type)
                        self.player_black_type = options[(idx + 1) % len(options)]

                    if event.key == pygame.K_r and self.last_game_moves:
                        self.replay_last_game()


def main():
    download_images() 
    pygame.init()
    pygame.display.set_caption("Chess AI")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    load_images()
    
    game = ChessGameGUI(screen)
    game.run_menu()

if __name__ == "__main__":
    main()