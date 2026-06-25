"""
This script evaluates chess positions using the Stockfish engine and saves the results in a JSON file. It reads FEN strings from a text file, evaluates each position, and writes the evaluation results to an output JSON file.
"""

import chess
import chess.engine
import json
import math

path = "engines/stockfish-ubuntu-x86-64-avx2"

def evaluate_positions():
    with chess.engine.SimpleEngine.popen_uci(path) as engine:
        f = open("data/output/lichess_positions.txt", "r")
        output_file = open("data/output/evaluated_positions.jsonl", "w")
        
        processed_lines = 0
        total_lines = 100000 # Assuming we know the total number of lines in advance; adjust as needed

        for line in f:
            fen = line.strip()
            if not fen:
                continue
                
            board = chess.Board(fen)
            info = engine.analyse(board, chess.engine.Limit(depth=8))
            wdl_score = info["score"].white()
            
            if wdl_score.is_mate():
                mate_moves = wdl_score.mate()
                if mate_moves > 0:
                    target = 0.9999
                else:
                    target = 0.0001
            else:
                cp = wdl_score.score()
                target = 1 / (1 + math.exp(-(cp / 600)))

            data_line = {
                "fen": fen,
                "target": target
            }
            output_file.write(json.dumps(data_line) + "\n")

            processed_lines += 1
            if processed_lines % 500 == 0 or processed_lines == total_lines:
                percent = (processed_lines / total_lines) * 100
                print(f"Progress: {processed_lines}/{total_lines} positions evaluated ({percent:.1f}%)", end="\r")
                
        f.close()
        output_file.close()
        print()


if __name__ == "__main__":
    evaluate_positions()
    print("Finished evaluating positions.")
    