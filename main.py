#!/usr/bin/env python3
"""
Interactive chess game using Maia 2 ONNX model with opening book support.
The script provides a CLI loop where you input moves and Maia suggests the most human-like response.
"""

import chess
import chess.polyglot
import onnxruntime as ort
import numpy as np
import torch
import os
import random
import argparse

# Utility functions from maia2 package
def create_elo_dict():
    """Create ELO rating category dictionary"""
    interval = 100
    start = 1100
    end = 2000
    
    range_dict = {f"<{start}": 0}
    range_index = 1

    for lower_bound in range(start, end - 1, interval):
        upper_bound = lower_bound + interval
        range_dict[f"{lower_bound}-{upper_bound - 1}"] = range_index
        range_index += 1

    range_dict[f">={end}"] = range_index
    return range_dict


def map_to_category(elo, elo_dict):
    """Map ELO rating to category"""
    interval = 100
    start = 1100
    end = 2000
    
    if elo < start:
        return elo_dict[f"<{start}"]
    elif elo >= end:
        return elo_dict[f">={end}"]
    else:
        for lower_bound in range(start, end - 1, interval):
            upper_bound = lower_bound + interval
            if lower_bound <= elo < upper_bound:
                return elo_dict[f"{lower_bound}-{upper_bound - 1}"]


def board_to_tensor(board):
    """Convert chess board to tensor representation"""
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    num_piece_channels = 12  # 6 piece types * 2 colors
    additional_channels = 6  # 1 for player's turn, 4 for castling rights, 1 for en passant
    tensor = torch.zeros((num_piece_channels + additional_channels, 8, 8), dtype=torch.float32)

    piece_indices = {piece: i for i, piece in enumerate(piece_types)}

    # Fill tensor for each piece type
    for piece_type in piece_types:
        for color in [True, False]:  # True is White, False is Black
            piece_map = board.pieces(piece_type, color)
            index = piece_indices[piece_type] + (0 if color else 6)
            for square in piece_map:
                row, col = divmod(square, 8)
                tensor[index, row, col] = 1.0

    # Player's turn channel (White = 1, Black = 0)
    turn_channel = num_piece_channels
    if board.turn == chess.WHITE:
        tensor[turn_channel, :, :] = 1.0

    # Castling rights channels
    castling_rights = [board.has_kingside_castling_rights(chess.WHITE),
                       board.has_queenside_castling_rights(chess.WHITE),
                       board.has_kingside_castling_rights(chess.BLACK),
                       board.has_queenside_castling_rights(chess.BLACK)]
    for i, has_right in enumerate(castling_rights):
        if has_right:
            tensor[num_piece_channels + 1 + i, :, :] = 1.0

    # En passant target channel
    ep_channel = num_piece_channels + 5
    if board.ep_square is not None:
        row, col = divmod(board.ep_square, 8)
        tensor[ep_channel, row, col] = 1.0

    return tensor


def generate_pawn_promotions():
    """Generate all possible pawn promotion moves"""
    promotion_rows = {'white': '7'}
    promotion_pieces = ['q', 'r', 'b', 'n']
    promotions = []

    for color, row in promotion_rows.items():
        target_row = '8' if color == 'white' else '1'

        for file in 'abcdefgh':
            # Direct move to promotion
            for piece in promotion_pieces:
                promotions.append(f'{file}{row}{file}{target_row}{piece}')

            # Capturing moves to the left and right
            if file != 'a':
                left_file = chr(ord(file) - 1)
                for piece in promotion_pieces:
                    promotions.append(f'{file}{row}{left_file}{target_row}{piece}')

            if file != 'h':
                right_file = chr(ord(file) + 1)
                for piece in promotion_pieces:
                    promotions.append(f'{file}{row}{right_file}{target_row}{piece}')

    return promotions


def get_all_possible_moves():
    """Generate all possible chess moves"""
    all_moves = []

    for rank in range(8):
        for file in range(8): 
            square = chess.square(file, rank)
            
            board = chess.Board(None)
            board.set_piece_at(square, chess.Piece(chess.QUEEN, chess.WHITE))
            legal_moves = list(board.legal_moves)
            all_moves.extend(legal_moves)
            
            board = chess.Board(None)
            board.set_piece_at(square, chess.Piece(chess.KNIGHT, chess.WHITE))
            legal_moves = list(board.legal_moves)
            all_moves.extend(legal_moves)
    
    all_moves = [all_moves[i].uci() for i in range(len(all_moves))]
    pawn_promotions = generate_pawn_promotions()
    
    return all_moves + pawn_promotions


def mirror_square(square):
    """Mirror a square vertically"""
    file = square[0]
    rank = str(9 - int(square[1]))
    return file + rank


def mirror_move(move_uci):
    """Mirror a move vertically"""
    is_promotion = len(move_uci) > 4
    start_square = move_uci[:2]
    end_square = move_uci[2:4]
    promotion_piece = move_uci[4:] if is_promotion else ""

    mirrored_start = mirror_square(start_square)
    mirrored_end = mirror_square(end_square)

    return mirrored_start + mirrored_end + promotion_piece


def preprocessing(fen, elo_self, elo_oppo, elo_dict, all_moves_dict):
    """Preprocess board position for model inference"""
    if fen.split(' ')[1] == 'w':
        board = chess.Board(fen)
    elif fen.split(' ')[1] == 'b':
        board = chess.Board(fen).mirror()
    else:
        raise ValueError(f"Invalid fen: {fen}")
        
    board_input = board_to_tensor(board)
    
    elo_self = map_to_category(elo_self, elo_dict)
    elo_oppo = map_to_category(elo_oppo, elo_dict)
    
    legal_moves = torch.zeros(len(all_moves_dict))
    legal_moves_idx = torch.tensor([all_moves_dict[move.uci()] for move in board.legal_moves])
    legal_moves[legal_moves_idx] = 1
    
    return board_input, elo_self, elo_oppo, legal_moves


def get_book_move(opening_book, board):
    """Get a move from the opening book if available"""
    if opening_book is None:
        return None
    
    try:
        # Get all book entries for this position
        entries = list(opening_book.find_all(board))
        
        if not entries:
            return None
        
        # Choose move based on weights
        total_weight = sum(entry.weight for entry in entries)
        
        if total_weight == 0:
            # If all weights are 0, choose the first one
            return entries[0].move
        
        # Use weighted random selection
        rand_value = random.randint(0, total_weight - 1)
        cumulative = 0
        
        for entry in entries:
            cumulative += entry.weight
            if rand_value < cumulative:
                return entry.move
        
        return entries[0].move
    
    except Exception as e:
        print(f"Error reading opening book: {e}")
        return None


def get_maia_move(session, fen, elo_self, elo_oppo, all_moves_dict, all_moves_dict_reversed, elo_dict, board_for_san=None):
    """Get the most human-like move from Maia model"""
    
    board_input, elo_self_cat, elo_oppo_cat, legal_moves = preprocessing(
        fen, elo_self, elo_oppo, elo_dict, all_moves_dict
    )
    
    # Prepare inputs for ONNX model
    boards = board_input.unsqueeze(0).numpy().astype(np.float32)
    elos_self = np.array([elo_self_cat], dtype=np.int64)
    elos_oppo = np.array([elo_oppo_cat], dtype=np.int64)
    
    # Run inference
    outputs = session.run(None, {
        'boards': boards,
        'elo_self': elos_self,
        'elo_oppo': elos_oppo
    })
    
    logits_maia = torch.from_numpy(outputs[0])
    logits_value = torch.from_numpy(outputs[2])
    
    # Apply legal move mask and get probabilities
    legal_moves = legal_moves.unsqueeze(0)
    logits_maia_legal = logits_maia * legal_moves
    probs = logits_maia_legal.softmax(dim=-1).cpu().tolist()[0]
    
    # Calculate win probability
    win_prob = (logits_value / 2 + 0.5).clamp(0, 1).item()
    
    black_flag = False
    if fen.split(" ")[1] == "b":
        win_prob = 1 - win_prob
        black_flag = True
    
    # Get move probabilities (keep UCI internally)
    move_probs_uci = {}
    legal_move_indices = legal_moves.nonzero().flatten().cpu().numpy().tolist()
    
    for move_idx in legal_move_indices:
        move = all_moves_dict_reversed[move_idx]
        if black_flag:
            move = mirror_move(move)
        move_probs_uci[move] = round(probs[move_idx], 4)
    
    # Sort by probability
    move_probs_uci = dict(sorted(move_probs_uci.items(), key=lambda item: item[1], reverse=True))
    
    return move_probs_uci, win_prob


def parse_move(move_str, board, use_san):
    """Parse a move string in either SAN or UCI notation"""
    if use_san:
        try:
            # Handle castling
            if move_str.lower() in ['o-o', 'o-o-o']:
                move_str = move_str.upper()
            
            return board.parse_san(move_str)
        except:
            raise ValueError(f"Invalid algebraic notation: {move_str}")
    else:
        try:
            return chess.Move.from_uci(move_str)
        except:
            raise ValueError(f"Invalid UCI notation: {move_str}")


def format_move(move, board, use_san):
    """Format a move in either SAN or UCI notation"""
    if use_san:
        return board.san(move)
    else:
        return move.uci()


def find_files_by_extension(extension):
    """Find all files with given extension in current directory"""
    current_dir = os.getcwd()
    files = [f for f in os.listdir(current_dir) if f.endswith(extension) and os.path.isfile(f)]
    return files


def select_file(files, file_type):
    """Let user select from multiple files or return the single file"""
    if len(files) == 0:
        return None
    elif len(files) == 1:
        return files[0]
    else:
        print(f"\nMultiple {file_type} files found:")
        for i, f in enumerate(files, 1):
            print(f"  {i}. {f}")
        
        while True:
            try:
                choice = input(f"Select {file_type} file (1-{len(files)}): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(files):
                    return files[idx]
                else:
                    print(f"Please enter a number between 1 and {len(files)}")
            except ValueError:
                print("Please enter a valid number")


def main():
    """Main CLI loop for interactive chess with Maia"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Interactive chess game using Maia 2 ONNX model')
    parser.add_argument('--notation', choices=['san', 'uci'], default='san',
                        help='Move notation format: san (algebraic notation, default) or uci')
    args = parser.parse_args()
    
    use_san = args.notation == 'san'
    notation_name = "algebraic notation (SAN)" if use_san else "UCI notation"
    notation_example = "e4, Nf3, O-O" if use_san else "e2e4, g1f3, e1g1"
    
    # Initialize
    print("=" * 60)
    print("Maia 2 Interactive Chess CLI")
    print("=" * 60)
    
    # Search for ONNX model files
    print("\nSearching for ONNX model files...")
    onnx_files = find_files_by_extension('.onnx')
    
    if onnx_files:
        model_path = select_file(onnx_files, "ONNX model")
        print(f"Using model: {model_path}")
    else:
        print("No .onnx files found in current directory.")
        model_path = input("Enter path to ONNX model: ").strip()
    
    try:
        session = ort.InferenceSession(model_path)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Search for opening book files
    opening_book = None
    print("\nSearching for opening book files...")
    bin_files = find_files_by_extension('.bin')
    
    if bin_files:
        use_book = input(f"\nFound {len(bin_files)} opening book file(s). Use opening book? (y/n): ").strip().lower()
        
        if use_book == 'y':
            book_path = select_file(bin_files, "opening book")
            print(f"Using opening book: {book_path}")
            
            try:
                opening_book = chess.polyglot.open_reader(book_path)
                print("✓ Opening book loaded successfully!")
            except Exception as e:
                print(f"✗ Error loading opening book: {e}")
                print("Continuing without opening book...")
                opening_book = None
    else:
        print("No .bin files found in current directory.")
        use_book = input("Do you want to use an opening book? (y/n): ").strip().lower()
        
        if use_book == 'y':
            book_path = input("Enter path to opening book (.bin format): ").strip()
            
            if os.path.exists(book_path):
                try:
                    opening_book = chess.polyglot.open_reader(book_path)
                    print("✓ Opening book loaded successfully!")
                except Exception as e:
                    print(f"✗ Error loading opening book: {e}")
                    print("Continuing without opening book...")
                    opening_book = None
            else:
                print(f"✗ File not found: {book_path}")
                print("Continuing without opening book...")
    
    # Setup
    print("\nPreparing move dictionaries...")
    all_moves = get_all_possible_moves()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}
    all_moves_dict_reversed = {v: k for k, v in all_moves_dict.items()}
    elo_dict = create_elo_dict()
    print("✓ Ready!")
    
    # Get player settings
    print("\n" + "=" * 60)
    your_elo = int(input("Enter your ELO rating (e.g., 1500): "))
    maia_elo = int(input("Enter Maia's ELO rating (e.g., 1500): "))
    
    player_color = input("Do you want to play as White or Black? (w/b): ").strip().lower()
    
    # Initialize board
    board = chess.Board()
    
    print("\n" + "=" * 60)
    print("Game started!")
    print(f"Move notation: {notation_name}")
    print(f"Enter moves in {notation_name} (e.g., {notation_example})")
    print("Type 'quit' to exit, 'board' to show position")
    if opening_book:
        print("Opening book is active for Maia's moves")
    print("=" * 60 + "\n")
    
    # Game loop
    move_count = 0
    while not board.is_game_over():
        print(board)
        print(f"\nMove {move_count + 1}")
        print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
        
        is_player_turn = (board.turn == chess.WHITE and player_color == 'w') or \
                         (board.turn == chess.BLACK and player_color == 'b')
        
        if is_player_turn:
            # Player's turn
            while True:
                move_input = input("\nYour move: ").strip()
                
                if move_input.lower() == 'quit':
                    print("Game ended by player.")
                    if opening_book:
                        opening_book.close()
                    return
                elif move_input.lower() == 'board':
                    print(board)
                    continue
                
                try:
                    move = parse_move(move_input, board, use_san)
                    if move in board.legal_moves:
                        board.push(move)
                        break
                    else:
                        print("Illegal move! Try again.")
                except ValueError as e:
                    print(f"Invalid move! {e}")
                    print(f"Use {notation_name} (e.g., {notation_example})")
                except Exception as e:
                    print(f"Error parsing move: {e}")
        else:
            # Maia's turn
            print("\nMaia is thinking...")
            
            # Check opening book first
            book_move = get_book_move(opening_book, board)
            
            if book_move:
                print("📖 Using opening book move")
                move_notation = format_move(book_move, board, use_san)
                board.push(book_move)
                print(f"Maia plays: {move_notation} (from book)")
            else:
                # Use Maia model
                fen = board.fen()
                move_probs_uci, win_prob = get_maia_move(
                    session, fen, maia_elo, your_elo, 
                    all_moves_dict, all_moves_dict_reversed, elo_dict
                )
                
                # Get best move (UCI)
                best_move_uci = max(move_probs_uci, key=move_probs_uci.get)
                best_prob = move_probs_uci[best_move_uci]
                
                # Convert to chess.Move and format for display
                best_move_obj = chess.Move.from_uci(best_move_uci)
                move_notation = format_move(best_move_obj, board, use_san)
                
                print(f"Maia plays: {move_notation} (confidence: {best_prob:.1%})")
                print(f"Win probability: {win_prob:.1%}")
                
                # Show top 3 alternatives
                top_moves_uci = list(move_probs_uci.items())[:3]
                print("\nTop alternatives:")
                for i, (move_uci, prob) in enumerate(top_moves_uci, 1):
                    move_obj = chess.Move.from_uci(move_uci)
                    move_display = format_move(move_obj, board, use_san)
                    print(f"  {i}. {move_display} ({prob:.1%})")
                
                board.push(best_move_obj)
        
        move_count += 1
        print("\n" + "-" * 60 + "\n")
    
    # Game over
    print(board)
    print("\n" + "=" * 60)
    print("Game Over!")
    print(f"Result: {board.result()}")
    
    if board.is_checkmate():
        print("Checkmate!")
    elif board.is_stalemate():
        print("Stalemate!")
    elif board.is_insufficient_material():
        print("Draw by insufficient material!")
    elif board.is_fifty_moves():
        print("Draw by fifty-move rule!")
    elif board.is_repetition():
        print("Draw by repetition!")
    
    print("=" * 60)
    
    # Clean up
    if opening_book:
        opening_book.close()


if __name__ == "__main__":
    main()