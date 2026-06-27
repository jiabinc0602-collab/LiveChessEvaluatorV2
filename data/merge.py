import os
import glob

"""
Script to stitch together temporary evaluation files into a single output file.
"""

scratch_dir = "data/scratch"
final_output = "data/lichess_positions.txt"
target_positions = 100000

positions_written = 0
scratch_files = glob.glob(os.path.join(scratch_dir, "*.tmp"))

print(f"Found {len(scratch_files)} temporary files. Stitching...")
with open(final_output, "w") as f_final:
    for scr_file in scratch_files:
        if positions_written >= target_positions:
            break
        with open(scr_file, "r") as f_temp:
            for line in f_temp:
                if positions_written < target_positions:
                    f_final.write(line)
                    positions_written += 1
                else:
                    break
        os.remove(scr_file)

print(f"Done! Successfully generated {final_output} with {positions_written:,} positions.")