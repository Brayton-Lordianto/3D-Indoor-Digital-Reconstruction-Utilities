import os
import cv2
import numpy as np
from glob import glob
from PIL import Image
import sys
'''
This program inpaints images using masks using OpenCV.
'''

def inpaint_images(image_dir, mask_dir, output_dir, inpaint_method='telea'):
    os.makedirs(output_dir, exist_ok=True)
    image_files = sorted(glob(os.path.join(image_dir, "*_back.png")) + 
                        glob(os.path.join(image_dir, "*_front.png")) + 
                        glob(os.path.join(image_dir, "*_left.png")) + 
                        glob(os.path.join(image_dir, "*_right.png")))
    
    for img_path in image_files:
        basename = os.path.basename(img_path)
        mask_path = os.path.join(mask_dir, basename + ".png")
        if not os.path.exists(mask_path):
            print(f"Mask not found for {basename}, skipping...")
            continue
        
        # Load image and mask
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        # Ensure mask is binary (255 for areas to inpaint, 0 for areas to keep)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # If mask is inverted (black=inpaint), invert it
        if np.mean(mask) > 127:  # If mask is mostly white, invert it
            mask = 255 - mask
        
        # Apply inpainting
        print(img.shape, mask.shape)
        if inpaint_method == 'telea':
            inpainted = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        else:  # 'ns' method
            inpainted = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
        output_path = os.path.join(output_dir, basename)
        cv2.imwrite(output_path, inpainted)
        print(f"Processed {basename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inpaint images using masks.")
    parser.add_argument("--image_dir", default="images", help="Directory containing original images.")
    parser.add_argument("--mask_dir", default="masks", help="Directory containing masks.")
    parser.add_argument("--output_dir", default="inpainted_images", help="Directory to save inpainted images.")
    args = parser.parse_args()