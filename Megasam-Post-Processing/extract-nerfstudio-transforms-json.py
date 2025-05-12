import numpy as np
import json
import os
from pathlib import Path

def megasam_to_nerfstudio(npz_path, output_dir, image_dir="images"):
     data = np.load(npz_path)
    images = data["images"]         # (T, H, W, 3)
    depths = data["depths"]         # (T, H, W)
    poses = data["cam_c2w"]         # (T, 4, 4)
    intrinsic = data["intrinsic"]   # (3, 3)
    
    os.makedirs(os.path.join(output_dir, image_dir), exist_ok=True)
    for i in range(len(images)):
        image_path = os.path.join(output_dir, image_dir, f"frame_{i:05d}.png")
        if not os.path.exists(image_path):
            # Convert to uint8 if needed
            if images[i].max() <= 1.0:
                img = (images[i] * 255).astype(np.uint8)
            else:
                img = images[i].astype(np.uint8)
            import cv2
            cv2.imwrite(image_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    H, W = images.shape[1:3]
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    # Create transforms.json
    transforms = {
        "fl_x": float(fx),
        "fl_y": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "w": int(W),
        "h": int(H),
        "aabb_scale": 16,
        "frames": [],
        "ply_file_path": "sparse_pc.ply"
    }
    
    # no distortion
    transforms["k1"] = 0.0
    transforms["k2"] = 0.0
    transforms["p1"] = 0.0
    transforms["p2"] = 0.0
    
    # Create frame entries
    for i in range(len(poses)):
        frame = {
            "file_path": f"./{image_dir}/frame_{i:05d}.png",
            "transform_matrix": poses[i].tolist()
        }
        transforms["frames"].append(frame)
    output_path = os.path.join(output_dir, "transforms.json")
    with open(output_path, 'w') as f:
        json.dump(transforms, f, indent=4)
    
    return output_path

# Complete conversion pipeline function that also generates the point cloud
def convert_megasam_to_nerfstudio(npz_path, output_dir, image_dir="images"):
    """
    Complete conversion from MegaSAM npz to Nerfstudio format including point cloud generation.
    
    Args:
        npz_path: Path to the MegaSAM npz file
        output_dir: Output directory for Nerfstudio files
        image_dir: Directory inside output_dir where images are stored
    """
    import open3d as o3d
    
    # Load MegaSAM data
    data = np.load(npz_path)    
    images = data["images"]         # (T, H, W, 3)
    depths = data["depths"]         # (T, H, W)
    poses = data["cam_c2w"]         # (T, 4, 4)
    intrinsic = data["intrinsic"]   # (3, 3)
    
    os.makedirs(os.path.join(output_dir, image_dir), exist_ok=True)
    transforms_path = megasam_to_nerfstudio(npz_path, output_dir, image_dir)
    print(f"Saved transforms.json to {transforms_path}")
    H, W = depths.shape[1:]
    # Build meshgrid for pixel coordinates
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    pixels = np.stack([xx, yy, np.ones_like(xx)], axis=-1).reshape(-1, 3).T  # (3, N)
    Kinv = np.linalg.inv(intrinsic)
    
    all_points = []
    all_colors = []
    
    # For each frame, create 3D points
    for i in range(len(images)):
        depth = depths[i].flatten()
        rgb = images[i].reshape(-1, 3) / 255.0  # normalize to [0, 1]
        pose = poses[i]
        
        pts_cam = Kinv @ pixels * depth
        pts_cam = np.vstack([pts_cam, np.ones((1, pts_cam.shape[1]))])  # (4, N)
        pts_world = (pose @ pts_cam)[:3].T  # (N, 3)
        valid = depth > 0
        all_points.append(pts_world[valid])
        all_colors.append(rgb[valid])
    
    # Merge everything
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
    ply_path = os.path.join(output_dir, "sparse_pc.ply")
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"Saved point cloud to {ply_path}")
    
    return transforms_path, ply_path

# Example usage:
transforms_path, ply_path = convert_megasam_to_nerfstudio("./fronts_sgd_cvd_hr.npz", "./cory_one_block")
# transforms_path, ply_path = convert_megasam_to_nerfstudio("./fronts_all_sgd_cvd_hr.npz", "./nerfstudio_dataset")