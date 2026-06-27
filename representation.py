"""
Convert FEN strings to tensor representations for neural network input.
"""

import chess
import numpy as np
import torch

import json

class ChessDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path, use_attack_map=False):
        self.use_attack_map = use_attack_map

        raw_entires = []

        with open(jsonl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_entires.append(json.loads(line))

        seen_fens = set()
        self.data_entries = []
    
        # Process and filter the raw entries
        for entry in raw_entires:
            fen = entry['fen']
            if fen not in seen_fens:
                self.data_entries.append(entry)
                seen_fens.add(fen)

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx):
        entry = self.data_entries[idx]
        
        # Generate the features
        X_tensor = fen_to_tensor(entry['fen'], self.use_attack_map)
        
        # Extract the target label as a float32 tensor
        y_target = torch.tensor(entry['target'], dtype=torch.float32)
        
        return X_tensor, y_target

def fen_to_tensor(fen, use_attack_map=False):
    board = chess.Board(fen)
    channels = 22 if use_attack_map else 20
    tensor = np.zeros((channels, 8, 8), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            color_idx = 0 if piece.color == chess.WHITE else 1
            channel = (piece.piece_type - 1) + (color_idx * 6)
            rank, col = divmod(square, 8)
            row = 7 - rank
            tensor[channel, row, col] = 1.0
        
    tensor[12, :, :] = int(board.turn)  # Turn channel
    tensor[13, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0  # White kingside castling
    tensor[14, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0  # White queenside castling
    tensor[15, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0  # Black kingside castling
    tensor[16, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0  # Black queenside castling

    if board.ep_square is not None:
        ep_row, ep_col = divmod(board.ep_square, 8)
        tensor[17, ep_row, ep_col] = 1.0  # En passant square
    tensor[18, :, :] = board.halfmove_clock / 100.0  # Halfmove clock
    tensor[19, :, :] = min(board.fullmove_number, 200) / 200.0  # Fullmove number

    if use_attack_map:
        attack_map = np.zeros((2, 8, 8), dtype=np.float32)
        for square in chess.SQUARES:
            rank, col = divmod(square, 8)
            row = 7 - rank

            attack_map[0, row, col] = 1.0 if board.attackers(chess.WHITE, square) else 0.0
            attack_map[1, row, col] = 1.0 if board.attackers(chess.BLACK, square) else 0.0
        tensor[20:22, :, :] = attack_map

    return torch.tensor(tensor)

