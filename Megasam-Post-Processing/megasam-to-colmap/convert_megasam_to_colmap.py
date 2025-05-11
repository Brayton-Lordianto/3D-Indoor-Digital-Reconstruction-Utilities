import numpy as np
import os
import argparse
import imageio
from scipy.spatial.transform import Rotation as R

def save_images(images, output_dir):
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    print("Saving images...")
    for idx, img in enumerate(images):
        filename = f"{idx+1:08d}.png"
        imageio.imwrite(os.path.join(images_dir, filename), img)
    print("Images saved to", images_dir)

def save_depths(depths, output_dir):
    depths_dir = os.path.join(output_dir, "depths")
    os.makedirs(depths_dir, exist_ok=True)

    print("Saving depth maps...")
    for idx, depth in enumerate(depths):
        filename = f"{idx+1:08d}.npy"
        np.save(os.path.join(depths_dir, filename), depth)
    print("Depth maps saved to", depths_dir)

def save_cameras(intrinsic, image_shape, output_dir):
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    H, W = image_shape

    cameras_path = os.path.join(output_dir, "cameras.txt")
    with open(cameras_path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {W} {H} {fx} {fy} {cx} {cy}\n")
    
    print("Camera intrinsics saved to", cameras_path)

def save_images_txt(cam_c2w, output_dir, num_images):
    images_path = os.path.join(output_dir, "images.txt")
    
    with open(images_path, "w") as f:
        f.write("# Image list with one line of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")

        for idx in range(num_images):
            c2w = cam_c2w[idx]
            w2c = np.linalg.inv(c2w)
            R_mat = w2c[:3, :3]
            t_vec = w2c[:3, 3]

            quat = R.from_matrix(R_mat).as_quat()  # x, y, z, w
            q_w = quat[3]
            q_xyz = quat[:3]

            f.write(f"{idx+1} {q_w} {q_xyz[0]} {q_xyz[1]} {q_xyz[2]} {t_vec[0]} {t_vec[1]} {t_vec[2]} 1 {idx+1:08d}.png\n")
            f.write("\n")
    
    print("Camera poses saved to", images_path)

def main(npz_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    data = np.load(npz_path)
    images = data["images"]  # (N, H, W, 3)
    depths = data["depths"]  # (N, H, W)
    intrinsic = data["intrinsic"]  # (3, 3)
    cam_c2w = data["cam_c2w"]  # (N, 4, 4)

    N, H, W, _ = images.shape

    save_images(images, output_dir)
    save_depths(depths, output_dir)
    save_cameras(intrinsic, (H, W), output_dir)
    save_images_txt(cam_c2w, output_dir, N)

    print("Conversion completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MegaSaM .npz output to COLMAP format for 2DGS.")
    parser.add_argument("--input", type=str, required=True, help="Path to MegaSaM .npz file.")
    parser.add_argument("--output", type=str, required=True, help="Output directory for COLMAP format.")

    args = parser.parse_args()
    main(args.input, args.output)
