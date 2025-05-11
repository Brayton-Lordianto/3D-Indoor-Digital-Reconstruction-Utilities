import os
import shutil
import argparse

def move_file_by_id_and_token(input_dir, output_dir, file_id, token="front"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    number = f"{file_id:04d}"

    file_name = f"{number}_{token}.png"
    src_path = os.path.join(input_dir, file_name)
    dest_path = os.path.join(output_dir, file_name)

    if os.path.exists(src_path):
        # shutil.copy(src_path, dest_path)
        shutil.move(src_path, dest_path)
        print(f"Copied: {src_path} -> {dest_path}")
    else:
        print(f"File not found: {src_path}")
        return -1

def bulk_move(input_dir, output_dir, token="front", jump=1):
    for i in range(2, 1000, jump):
        res = move_file_by_id_and_token(input_dir, output_dir, i, token)
        if res == -1: return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move a specific '<id>_<token>.png' file to output directory.")
    parser.add_argument("--input", help="Input directory containing PNG files.")
    parser.add_argument("--output", help="Output directory to move the file.")
    parser.add_argument("--token", default="front", help="Token of the file to move.")
    parser.add_argument("--jump", type=int, default=1, help="Jump value for file IDs.")
    args = parser.parse_args()
    bulk_move(args.input, args.output, args.token, args.jump)
    