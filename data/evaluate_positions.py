"""
This script parallelizes chess position evaluation using a pool of Stockfish engines.
It dynamically tracks lines processed across all cores for progress updates.
"""

import chess
import chess.engine
import json
import math
import os
import sys
from multiprocessing import Pool, cpu_count


ENGINE_PATH = "engines/stockfish-ubuntu-x86-64-avx2"
INPUT_PATH = "data/lichess_positions.txt"
OUTPUT_PATH = "data/output/evaluated_positions.jsonl"
BATCH_SIZE = 1000  # Number of FENs sent to a worker at a time

def worker_init():
    """
    Initializer executed once when each worker process starts.
    Gives each process its own dedicated, long-running Stockfish instance.
    """
    global worker_engine
    # Set threads=1 inside Stockfish so it doesn't fight other workers for cores
    worker_engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    worker_engine.configure({"Threads": 1})

def worker_cleanup():
    """Executed when the worker process shuts down."""
    global worker_engine
    if 'worker_engine' in globals():
        worker_engine.quit()

def evaluate_batch(args):
    """
    Processes a localized batch of FEN strings using the worker's dedicated engine.
    """
    batch_idx, fens = args
    evaluated_lines = []
    
    global worker_engine

    for fen in fens:
        fen = fen.strip()
        if not fen:
            continue
            
        try:
            board = chess.Board(fen)
            info = worker_engine.analyse(board, chess.engine.Limit(depth=8))
            wdl_score = info["score"].white()
            
            if wdl_score.is_mate():
                mate_moves = wdl_score.mate()
                target = 0.9999 if mate_moves > 0 else 0.0001
            else:
                cp = wdl_score.score()
                target = 1 / (1 + math.exp(-(cp / 600)))

            evaluated_lines.append(json.dumps({"fen": fen, "target": target}) + "\n")
        except Exception as e:
            continue
            
    return evaluated_lines

def fen_batch_generator(file_path, batch_size):
    """Streams FEN lines from disk and groups them into chunks on the fly."""
    current_batch = []
    batch_counter = 0
    with open(file_path, "r") as f:
        for line in f:
            current_batch.append(line)
            if len(current_batch) >= batch_size:
                yield (batch_counter, current_batch)
                batch_counter += 1
                current_batch = []
        if current_batch:
            yield (batch_counter, current_batch)

def main():
    print("🚀 Initializing Parallel Stockfish Evaluation Pool...")
    
    num_workers = max(1, cpu_count() - 1)
    print(f"Spawning {num_workers} parallel workers (1 Stockfish instance per core)...")

    processed_positions = 0
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    with open(OUTPUT_PATH, "w") as output_file:
        with Pool(processes=num_workers, initializer=worker_init) as pool:
        
            batches = pool.imap(evaluate_batch, fen_batch_generator(INPUT_PATH, BATCH_SIZE), chunksize=1)
            
            for evaluated_lines in batches:
                output_file.writelines(evaluated_lines)
                
                processed_positions += len(evaluated_lines)
                
                sys.stdout.write(f"\rProgress: {processed_positions:,} positions evaluated & written.")
                sys.stdout.flush()
                
            pool.apply(worker_cleanup)

    print(f"\n🎉 Success! Evaluation complete. File compiled at: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()