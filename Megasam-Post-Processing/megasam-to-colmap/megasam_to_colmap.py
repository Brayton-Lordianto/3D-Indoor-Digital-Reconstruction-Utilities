#!/usr/bin/env python3
import numpy as np
import os
import struct
import cv2
import shutil
import argparse

def write_cameras_bin(camera_id_to_params, filepath):
    with open(filepath, "wb") as fid:
        fid.write(struct.pack("<Q", len(camera_id_to_params)))
        for camera_id, params in camera_id_to_params.items():
            model_id = params["model_id"]
            width = params["width"]
            height = params["height"]
            params_array = params["params"]
            fid.write(struct.pack("<I", camera_id))
            fid.write(struct.pack("<i", model_id))
            fid.write(struct.pack("<QQ", width, height))
            for param in params_array:
                fid.write(struct.pack("<d", param))

def write_images_bin(image_id_to_data, filepath):
    with open(filepath, "wb") as fid:
        fid.write(struct.pack("<Q", len(image_id_to_data)))
        for image_id, data in image_id_to_data.items():
            fid.write(struct.pack("<I", image_id))
            for q in data["qvec"]:
                fid.write(struct.pack("<d", q))
            for t in data["tvec"]:
                fid.write(struct.pack("<d", t))
            fid.write(struct.pack("<I", data["camera_id"]))
            name_str = data["name"].encode("utf-8")
            fid.write(struct.pack("<I", len(name_str) + 1))
            fid.write(name_str + b"\0")
            fid.write(struct.pack("<Q", len(data["xys"])))
            for xy, point3D_id in zip(data["xys"], data["point3D_ids"]):
                fid.write(struct.pack("<dd", xy[0], xy[1]))
                fid.write(struct.pack("<Q", point3D_id))

def write_points3D_bin(points3D, filepath):
    with open(filepath, "wb") as fid:
        fid.write(struct.pack("<Q", len(points3D)))
        for point3D_id, data in points3D.items():
            fid.write(struct.pack("<Q", point3D_id))
            for coord in data["xyz"]:
                fid.write(struct.pack("<d", coord))
            for color in data["rgb"]:
                fid.write(struct.pack("<B", color))
            fid.write(struct.pack("<d", data["error"]))
            fid.write(struct.pack("<I", len(data["track"])))
            for image_id, point2D_idx in data["track"]:
                fid.write(struct.pack("<II", image_id, point2D_idx))

def write_points3D_ply(points3D, filepath):
    with open(filepath, "w") as fid:
        fid.write("ply\n")
        fid.write("format ascii 1.0\n")
        fid.write(f"element vertex {len(points3D)}\n")
        fid.write("property float x\n")
        fid.write("property float y\n")
        fid.write("property float z\n")
        fid.write("property uchar red\n")
        fid.write("property uchar green\n")
        fid.write("property uchar blue\n")
        fid.write("end_header\n")
        for _, data in points3D.items():
            x, y, z = data["xyz"]
            r, g, b = data["rgb"]
            fid.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")

def convert_c2w_to_colmap_transform(c2w):
    from scipy.spatial.transform import Rotation as R
    rot = c2w[:3, :3]
    trans = c2w[:3, 3]
    r = R.from_matrix(rot)
    qvec = r.as_quat()
    qvec = np.roll(qvec, 1)
    return qvec, trans

def generate_sparse_sample_points(H, W, sample_rate=0.1):
    num_points = int(H * W * sample_rate)
    indices = np.random.choice(H * W, size=num_points, replace=False)
    y_indices = indices // W
    x_indices = indices % W
    return list(zip(x_indices, y_indices))

def megasam_to_colmap(npz_path, output_dir, image_source_dir=None):
    data = np.load(npz_path)
    images = data["images"]
    depths = data["depths"]
    poses = data["cam_c2w"]
    intrinsic = data["intrinsic"]
    colmap_dir = os.path.join(output_dir, "colmap")
    sparse_dir = os.path.join(colmap_dir, "sparse", "0")
    os.makedirs(sparse_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    num_frames, H, W = depths.shape
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    camera_id_to_params = {
        1: {
            "model_id": 1,
            "width": W,
            "height": H,
            "params": [fx, fy, cx, cy]
        }
    }
    write_cameras_bin(camera_id_to_params, os.path.join(sparse_dir, "cameras.bin"))
    image_id_to_data = {}
    points3D = {}
    point3D_id = 1
    sample_points = generate_sparse_sample_points(H, W, sample_rate=0.05)
    for i in range(num_frames):
        image_name = f"frame_{i:05d}.png"
        image_path = os.path.join(images_dir, image_name)
        if image_source_dir:
            src_path = os.path.join(image_source_dir, image_name)
            if os.path.exists(src_path):
                shutil.copy(src_path, image_path)
            else:
                img = images[i]
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
                cv2.imwrite(image_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        else:
            img = images[i]
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            cv2.imwrite(image_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        c2w = poses[i]
        qvec, tvec = convert_c2w_to_colmap_transform(c2w)
        image_id_to_data[i + 1] = {
            "qvec": qvec,
            "tvec": tvec,
            "camera_id": 1,
            "name": image_name,
            "xys": [],
            "point3D_ids": []
        }
        depth = depths[i]
        img_rgb = images[i]
        for point_idx, (x, y) in enumerate(sample_points):
            if depth[y, x] <= 0:
                continue
            z = depth[y, x]
            x_cam = (x - cx) * z / fx
            y_cam = (y - cy) * z / fy
            point_cam = np.array([x_cam, y_cam, z, 1.0])
            point_world = c2w @ point_cam
            points3D[point3D_id] = {
                "xyz": point_world[:3],
                "rgb": img_rgb[y, x].astype(np.uint8),
                "error": 0.0,
                "track": [(i + 1, point_idx)]
            }
            image_id_to_data[i + 1]["xys"].append([x, y])
            image_id_to_data[i + 1]["point3D_ids"].append(point3D_id)
            point3D_id += 1
    write_images_bin(image_id_to_data, os.path.join(sparse_dir, "images.bin"))
    write_points3D_bin(points3D, os.path.join(sparse_dir, "points3D.bin"))
    write_points3D_ply(points3D, os.path.join(sparse_dir, "points3D.ply"))
    with open(os.path.join(sparse_dir, "project.ini"), "w") as f:
        f.write("[General]\n")
        f.write("database_path=../../database.db\n")
        f.write("image_path=../../../images\n")
    with open(os.path.join(colmap_dir, "database.db"), "wb") as f:
        pass
    print(f"COLMAP conversion complete. Files written to {output_dir}")
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MegaSAM npz to COLMAP format")
    parser.add_argument("npz_path", type=str, help="Path to MegaSAM npz file")
    parser.add_argument("output_dir", type=str, help="Output directory for COLMAP files")
    parser.add_argument("--image_source", type=str, default=None, 
                        help="Optional directory containing source images")
    args = parser.parse_args()
    megasam_to_colmap(args.npz_path, args.output_dir, args.image_source)
