"""
Optimized parallel extraction for 10M+ positions.
Balances worker load, isolates scratch files, and tracks live position counts.
"""

import zstandard as zstd
import io
import chess.pgn
import os
import random
import shutil
import sys
import multiprocessing as mp
from dotenv import load_dotenv

load_dotenv()

def process_games_worker(worker_id, game_texts, chunk_id, scratch_dir):
    """
    Worker process: Parses its assigned chunk of raw text and 
    immediately appends the extracted FENs directly to a scratch file.
    """
    scratch_filename = os.path.join(scratch_dir, f"scratch_worker_{worker_id}_chunk_{chunk_id}.tmp")
    extracted_count = 0
    
    with open(scratch_filename, "w") as f_out:
        for text in game_texts:
            game = chess.pgn.read_game(io.StringIO(text))
            if game is None:
                continue
            
            # Game Filters
            event_string = game.headers.get("Event", "")
            events_exclude = ["Bullet", "Computer", "Puzzles", "Casual"]
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

            # Position Sampling
            board = game.board()
            next_sample = random.randint(4, 6)

            for node in game.mainline():
                board.push(node.move)

                if next_sample > 1:
                    next_sample -= 1
                    continue
                else:
                    next_sample = random.randint(4, 6)

                f_out.write(f"{board.fen()}\n")
                extracted_count += 1
                
    return scratch_filename, extracted_count


def extract_positions_parallel(lichess_db, final_output_path, target_positions, max_workers=8):
    # Set up dedicated scratch directory
    scratch_dir = "data/scratch"
    os.makedirs(scratch_dir, exist_ok=True)

    pool = mp.Pool(processes=max_workers)
    active_tasks = []
    
    chunk_size = 1500  
    current_chunk = []
    chunk_id = 0
    
    total_positions_tracked = 0
    scratch_files = []

    print(f"Starting extraction using {max_workers} worker cores...")

    with open(lichess_db, "rb") as compressed_file:
        dctx = zstd.ZstdDecompressor()

        with dctx.stream_reader(compressed_file) as reader:
            buffered_reader = io.BufferedReader(reader, buffer_size=4 * 1024 * 1024) 
            text_stream = io.TextIOWrapper(buffered_reader, encoding="utf-8")

            while total_positions_tracked < target_positions:
                game_text = ""
                while True:
                    line = text_stream.readline()
                    if not line:
                        break
                    game_text += line
                    if line.startswith("1. "): 
                        while True:
                            move_line = text_stream.readline()
                            game_text += move_line
                            if move_line.strip() == "":
                                break
                        break
                
                if not game_text:
                    break 
                
                current_chunk.append(game_text)
                
                if len(current_chunk) >= chunk_size:
                    worker_id = chunk_id % max_workers
                    
                    # Submit task
                    task = pool.apply_async(process_games_worker, (worker_id, current_chunk, chunk_id, scratch_dir))
                    active_tasks.append(task)
                    
                    chunk_id += 1
                    current_chunk = []
                    
                    # Throttling guardrail with clean in-place tracking
                    if len(active_tasks) > max_workers * 2:
                        oldest_task = active_tasks.pop(0)
                        scr_file, count = oldest_task.get() 
                        total_positions_tracked += count
                        if scr_file:
                            scratch_files.append(scr_file)
                        
                        # Print live status on a single updating line
                        sys.stdout.write(
                            f"\rChunks Submitted: {chunk_id} | Extracted Positions Tracked: ~{total_positions_tracked:,} / {target_positions:,}"
                        )
                        sys.stdout.flush()

            # Submit final remaining games
            if current_chunk:
                worker_id = chunk_id % max_workers
                task = pool.apply_async(process_games_worker, (worker_id, current_chunk, chunk_id, scratch_dir))
                active_tasks.append(task)

            print("\nEntire database file parsed into workers. Finalizing background processing...")
            pool.close()
            
            # Resolve remaining tasks
            for i, task in enumerate(active_tasks):
                scr_file, count = task.get()
                total_positions_tracked += count
                if scr_file:
                    scratch_files.append(scr_file)
                sys.stdout.write(f"\rResolving remaining cores: {i+1}/{len(active_tasks)} processed...")
                sys.stdout.flush()
                    
            pool.join()

    # --- STITCHING PHASE ---
    print(f"\n\nProcessing complete. Combining temporary chunks into final dataset...")
    
    positions_written = 0
    with open(final_output_path, "w") as f_final:
        for scr_file in scratch_files:
            if positions_written >= target_positions:
                break
                
            if os.path.exists(scr_file):
                with open(scr_file, "r") as f_temp:
                    for line in f_temp:
                        if positions_written < target_positions:
                            f_final.write(line)
                            positions_written += 1
                            if positions_written % 250000 == 0:
                                sys.stdout.write(f"\rAssembling Final File: {positions_written:,}/{target_positions:,} lines merged...")
                                sys.stdout.flush()
                        else:
                            break

    # Clean up the scratch folder entirely
    print("\n\nCleaning up temporary scratch folder directory...")
    if os.path.exists(scratch_dir):
        shutil.rmtree(scratch_dir)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True) 

    lichess_db = os.getenv("lichess_db")
    if lichess_db is None:
        raise ValueError("Missing 'lichess_db' in .env file.")
        
    os.makedirs("data", exist_ok=True)
    
    TARGET_WORKERS = 8 
    positions_to_extract = 10000000
    final_output = "data/lichess_positions.txt"

    extract_positions_parallel(
        lichess_db=lichess_db, 
        final_output_path=final_output, 
        target_positions=positions_to_extract,
        max_workers=TARGET_WORKERS
    )
        
    print("\nFinished extracting all positions safely.")