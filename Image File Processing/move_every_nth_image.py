import os
import shutil
import re

def copy_every_nth_set(source_dir, dest_dir, n=5):
    """
    Copy every nth set of images from source_dir to dest_dir.
    A set consists of all angle views (back, front, left, right) for a given index.
    
    Parameters:
    - source_dir: Source directory containing all images
    - dest_dir: Destination directory for the copied subset
    - n: Copy every nth set (default: 5)
    """
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Get all files in the source directory
    all_files = sorted(os.listdir(source_dir))
    
    # Extract unique IDs (e.g., "0001", "0002", etc.)
    id_pattern = re.compile(r'(\d{4})_')
    unique_ids = set()
    
    for filename in all_files:
        match = id_pattern.search(filename)
        if match:
            unique_ids.add(match.group(1))
    
    # Sort the unique IDs
    sorted_ids = sorted(list(unique_ids))
    
    # Select every nth ID
    selected_ids = [sorted_ids[i] for i in range(0, len(sorted_ids), n)]
    
    print(f"Found {len(sorted_ids)} unique IDs")
    print(f"Selected {len(selected_ids)} IDs: {', '.join(selected_ids)}")
    
    # Copy files for selected IDs
    copied_count = 0
    for filename in all_files:
        for selected_id in selected_ids:
            if filename.startswith(selected_id):
                src_path = os.path.join(source_dir, filename)
                dst_path = os.path.join(dest_dir, filename)
                shutil.copy2(src_path, dst_path)
                copied_count += 1
                print(f"Copied: {filename}")
                break
    
    print(f"Total files copied: {copied_count}")

if __name__ == "__main__":
    # Change these paths to match your directories
    source_directory = "inpainted_images"
    destination_directory = "trimmed_images"
    
    # Copy every 5th set (0001, 0006, 0011, ...)
    copy_every_nth_set(source_directory, destination_directory, n=5)