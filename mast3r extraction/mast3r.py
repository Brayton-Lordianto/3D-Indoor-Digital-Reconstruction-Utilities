'''
This is a modified version of Mast3r demo script. Run in the same directory as the original Mast3r demo script.
Will create output point cloud in the same directory as the input images. 
example script usage
python mast3r.py --input cory_images/_test/ --output_dir cory_images/test_output/ --model_name DUSt3R_ViTLarge_BaseDecoder_512_dpt --device "cuda"
NOTE line 284
'''


import math
import gradio
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "asmk")))
import numpy as np
import functools
import trimesh
import copy
from scipy.spatial.transform import Rotation
import tempfile
import shutil
import torch

from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess
from mast3r.image_pairs import make_pairs
from mast3r.retrieval.processor import Retriever

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.demo import get_args_parser as dust3r_get_args_parser

import matplotlib.pyplot as pl
from mast3r.model import AsymmetricMASt3R
from plyfile import PlyData, PlyElement
import pandas as pd
def save_pointcloud_to_ply(points, filename="output.ply"):
    vertices = np.array([(x, y, z) for x, y, z in points],
                        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertices, 'vertex')
    PlyData([el]).write(filename)

def save_pointcloud_as_ply_with_color(points, colors, ply_filename):
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)
            
        # Create vertex data with color
        vertices = np.empty(len(points), 
                           dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                  ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        
        # Assign coordinates and colors
        vertices['x'] = points[:, 0]
        vertices['y'] = points[:, 1]
        vertices['z'] = points[:, 2]
        vertices['red'] = colors[:, 0]
        vertices['green'] = colors[:, 1]
        vertices['blue'] = colors[:, 2]
        
        el = PlyElement.describe(vertices, 'vertex')
        PlyData([el]).write(ply_filename)


class SparseGAState:
    def __init__(self, sparse_ga, should_delete=False, cache_dir=None, outfile_name=None):
        self.sparse_ga = sparse_ga
        self.cache_dir = cache_dir
        self.outfile_name = outfile_name
        self.should_delete = should_delete

    def __del__(self):
        if not self.should_delete:
            return
        if self.cache_dir is not None and os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        self.cache_dir = None
        if self.outfile_name is not None and os.path.isfile(self.outfile_name):
            os.remove(self.outfile_name)
        self.outfile_name = None


def get_args_parser():
    parser = dust3r_get_args_parser()
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--gradio_delete_cache', default=None, type=int,
                        help='age/frequency at which gradio removes the file. If >0, matching cache is purged')
    parser.add_argument('--retrieval_model', default="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth", type=str, help="retrieval_model to be loaded")
    parser.add_argument('--input', required=True, help="Directory of images")
    parser.add_argument('--output_dir', default='output', help='Output directory')

    # actions = parser._actions
    # for action in actions:
    #     if action.dest == 'model_name':
    #         action.choices = ["MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"]
    # change defaults
    parser.prog = 'mast3r demo'
    return parser


def _convert_scene_output_to_glb(outfile, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, mask)]).reshape(-1, 3)
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)]).reshape(-1, 3)
        valid_msk = np.isfinite(pts.sum(axis=1))
        pct = trimesh.PointCloud(pts[valid_msk], colors=col[valid_msk])
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            pts3d_i = pts3d[i].reshape(imgs[i].shape)
            msk_i = mask[i] & np.isfinite(pts3d_i.sum(axis=-1))
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d_i, msk_i))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile


def get_3D_model_from_scene(silent, scene_state, min_conf_thr=2, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05, TSDF_thresh=0):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene_state is None:
        return None
    outfile = scene_state.outfile_name
    if outfile is None:
        return None

    # get optimized values from scene
    scene = scene_state.sparse_ga
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()

    # 3D pointcloud from depthmap, poses and intrinsics
    if TSDF_thresh > 0:
        tsdf = TSDFPostProcess(scene, TSDF_thresh=TSDF_thresh)
        pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=clean_depth))
    else:
        pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth))
    msk = to_numpy([c > min_conf_thr for c in confs])
    return _convert_scene_output_to_glb(outfile, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, silent=silent)


def get_reconstructed_scene(outdir, model, retrieval_model, device, silent, image_size, filelist, optim_level, lr1, niter1, lr2, niter2, min_conf_thr,
                            matching_conf_thr, as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
                            scenegraph_type, winsize, win_cyclic, refid, TSDF_thresh, shared_intrinsics, **kw):
    """
    from a list of images, run mast3r inference, sparse global aligner.
    then run get_3D_model_from_scene
    """
    imgs = load_images(filelist, size=image_size, verbose=not silent)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
        filelist = [filelist[0], filelist[0] + '_2']

    scene_graph_params = [scenegraph_type]
    if scenegraph_type in ["swin", "logwin"]:
        scene_graph_params.append(str(winsize))
    elif scenegraph_type == "oneref":
        scene_graph_params.append(str(refid))
    elif scenegraph_type == "retrieval":
        scene_graph_params.append(str(winsize))  # Na
        scene_graph_params.append(str(refid))  # k

    if scenegraph_type in ["swin", "logwin"] and not win_cyclic:
        scene_graph_params.append('noncyclic')
    scene_graph = '-'.join(scene_graph_params)

    sim_matrix = None
    if 'retrieval' in scenegraph_type:
        assert retrieval_model is not None
        retriever = Retriever(retrieval_model, backbone=model, device=device)
        with torch.no_grad():
            sim_matrix = retriever(filelist)

        # Cleanup
        del retriever
        torch.cuda.empty_cache()

    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True, sim_mat=sim_matrix)
    if optim_level == 'coarse':
        niter2 = 0
    # Sparse GA (forward mast3r -> matching -> 3D optim -> 2D refinement -> triangulation)
    cache_dir = os.path.join(outdir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    scene = sparse_global_alignment(filelist, pairs, cache_dir,
                                    model, lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2, device=device,
                                    opt_depth='depth' in optim_level, shared_intrinsics=shared_intrinsics,
                                    matching_conf_thr=matching_conf_thr, **kw)
    
    # save pointcloud to ply 
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    # 3D pointcloud from depthmap, poses and intrinsics
    if TSDF_thresh > 0:
        tsdf = TSDFPostProcess(scene, TSDF_thresh=TSDF_thresh)
        pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=clean_depth))
    else:
        pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth))

    confs2 = np.array(to_numpy([c for c in confs])) 
    print(confs2.max(), confs2.min(), confs2.mean(), confs2.std())
    # sys.exit(0)
    min_conf_thr = confs2.mean() + 0.5 * confs2.std() # gets most precise points
    mask = to_numpy([c > min_conf_thr for c in confs])
    # this is part of _convert_scene_output_to_glb
    imgs = rgbimg
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, mask)]).reshape(-1, 3)
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)]).reshape(-1, 3)
        valid_msk = np.isfinite(pts.sum(axis=1))
        pct = trimesh.PointCloud(pts[valid_msk], colors=col[valid_msk])
        pcfilename = outdir + "/test.ply"
        print("exporting pointcloud to", pcfilename)
        pct.export(pcfilename)
        scene.add_geometry(pct)
    
    
def main_demo(filelist, tmpdirname, model, retrieval_model, device, image_size, silent=False):
    silent = True 
    if not silent:
        print('Outputing stuff in', tmpdirname)

    recon_fun = get_reconstructed_scene
    model_from_scene_fun = get_3D_model_from_scene
    available_scenegraph_type = [("complete: all possible image pairs", "complete"),
                                 ("swin: sliding window", "swin"),
                                 ("logwin: sliding window with long range", "logwin"),
                                 ("oneref: match one image with all", "oneref")]
    if retrieval_model is not None:
        available_scenegraph_type.insert(1, ("retrieval: connect views based on similarity", "retrieval"))

    lr1 = 0.07  # Coarse learning rate
    niter1 = 300  # Coarse iterations
    lr2 = 0.01  # Fine learning rate
    niter2 = 300  # Fine iterations
    optim_level = 'refine+depth'  # Optimization level
    matching_conf_thr = 0.0  # Matching confidence threshold
    shared_intrinsics = False  # Use shared intrinsics
    # CHANGE SCENEGRAPH TYPE HERE
    scenegraph_type = 'complete'  # Scene graph type
    # scenegraph_type = 'retrieval'  # Scene graph type
    # Scene graph parameters
    winsize = 1  # Window size for scene graph  
    win_cyclic = False  # Cyclic sequence for scene graph
    refid = 0  # Reference id for scene graph
    # other settings
    min_conf_thr = 2  # Minimum confidence threshold
    cam_size = 0.2  # Camera size in output
    TSDF_thresh = 0.0  # TSDF threshold
    as_pointcloud = True  # Output as pointcloud
    mask_sky = False  # Mask sky
    clean_depth = True  # Clean depth maps
    transparent_cams = False  # Transparent cameras
    
    # do function
    get_reconstructed_scene(outdir=tmpdirname, model=model, retrieval_model=retrieval_model, device=device,
                            silent=silent, image_size=image_size, filelist=filelist, optim_level=optim_level,
                            lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2, min_conf_thr=min_conf_thr,
                            matching_conf_thr=matching_conf_thr, as_pointcloud=as_pointcloud,
                            mask_sky=mask_sky, clean_depth=clean_depth, transparent_cams=transparent_cams,
                            cam_size=cam_size, scenegraph_type=scenegraph_type, winsize=winsize,
                            win_cyclic=win_cyclic, refid=refid, TSDF_thresh=TSDF_thresh,
                            shared_intrinsics=shared_intrinsics)

if __name__ == '__main__':
    print("hi")
    parser = get_args_parser()
    args = parser.parse_args()
    device = torch.device(args.device)
    model_path = "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    model = AsymmetricMASt3R.from_pretrained(model_path).to(device)
    args.retrieval_model = "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"
    
    # Load images
    filelist = sorted([
        os.path.join(args.input, f) for f in os.listdir(args.input)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = os.path.join(args.output_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    main_demo(filelist, args.output_dir, model, args.retrieval_model, device, args.image_size) 
    
