"""
This script is to extract positions from lichess file
and parse into FEN format for further analysis.
"""

import zstandard as zstd
import io
import chess.pgn
import os
from dotenv import load_dotenv
import random

load_dotenv()

def extract_positions(positions):
    total_positions = 0
# decompress loop, lichess games are stored as .pgn.zst
    with open(lichess_db, "rb") as compressed_file:
        dctx = zstd.ZstdDecompressor()

        with dctx.stream_reader(compressed_file) as reader:
            buffered_reader = io.BufferedReader(reader, buffer_size=1024*1024)
            
            text_stream = io.TextIOWrapper(buffered_reader, encoding="utf-8")

            # loop to iterate through games
            while total_positions < positions:
                game = chess.pgn.read_game(text_stream)
                # random thinning (because a single move in chess usually barely affects eval, so we thin dataset for more info)
                next_sample = random.randint(4, 6)

                if game is None:
                    break
                
                # conditions to check if game is valid for analysis
                event_string = game.headers.get("Event", "")
                events_exclude=["Bullet", "Computer", "Puzzles", "Casual"]
                if any(event in event_string for event in events_exclude):
                    continue

                if game.headers.get("Termination") == "Abandoned":
                    continue

                if game.headers.get("Variant", "Standard") != "Standard":
                    continue

                white_elo = game.headers.get("WhiteElo")
                black_elo = game.headers.get("BlackElo")

                if not white_elo or not black_elo or int(white_elo) < 2200 or int(black_elo) < 2200:
                    continue

                board = game.board()

                # loop to iterate through moves
                for node in game.mainline():
                    board.push(node.move)

                    if next_sample > 1:
                        next_sample -= 1
                        continue
                    else:
                        next_sample = random.randint(4, 6)

                    fen = board.fen()
                    # write fen to file
                    f.write(f"{fen}\n")
                    total_positions += 1
                    
                    # progress check
                    if total_positions % 5000 == 0:
                        print(f"Progress: {total_positions}/{positions} positions extracted...", end="\r")


if __name__ == "__main__":
    lichess_db = os.getenv("lichess_db")
    if lichess_db is None:
        raise ValueError("Missing 'lichess_db' in .env file.")
    f = open("data/lichess_positions.txt", "w")
    positions_to_extract = 100000
    extract_positions(positions_to_extract)
    print()
    print("Finished extracting positions.")
    f.close()
