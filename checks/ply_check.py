"""
Mesh NOC PLY render
"""

import os
import sys
import cv2
import numpy as np
from scipy import io as io
import torch
from models.textures import Texture
import pickle

from scipy.spatial.transform import Rotation as R

import trimesh
import argparse

from utils.render_utils import Render

from external.smplx.smplx import body_models
sys.path.insert(0, '../external/pyrender')
import pyrender


def main(opt):
    ply_start = '''ply
    format ascii 1.0
    element vertex {}
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header\n'''

    model = body_models.create(model_path='../3d_data/models', model_type='smpl', gender='male', ext='pkl')
    smpl = pickle.load(open('../3d_data/densepose_uv.pkl', 'rb'))
    faces = np.array(smpl['f_extended'], dtype=np.int64).reshape((-1, 3))
    uv_faceid = io.loadmat('../3d_data/DensePoseData/UV_data/UV_Processed.mat')['All_FaceIndices']
    uv = smpl['uv']

    # with open('../3d_data/nongrey_male_0110.jpg', 'rb') as file:
    texture = cv2.imread('../3d_data/nongrey_male_0110.jpg')

    model_pose = np.zeros((1, 69))
    global_rot = R.from_euler('zyx', [0, 45, 0], degrees=True)
    rot_1 = R.from_euler('zyx', [45, 0, 0], degrees=True)
    rot_2 = R.from_euler('zyx', [-45, 0, 0], degrees=True)

    # model_pose[0, 0:3] = np.array([1, 1, 1])
    model_pose[0, 0:3] = rot_1.as_rotvec()
    model_pose[0, 3:6] = rot_2.as_rotvec()
    model.body_pose.data = model.body_pose.new(model_pose)
    model.global_orient.data = model.global_orient.new(global_rot.as_rotvec().reshape((1, 3)))

    output = model(return_verts=True)
    vertices = output.vertices.detach().cpu().numpy().squeeze()

    smpl_vertices = np.array([vertices[i] for i in smpl['v_extended']])

    smpl_uv_visual = trimesh.visual.TextureVisuals(uv=uv, image=texture)

    # smpl_render = Render(width=opt.image_width, height=opt.image_height,
    #                      camera_distance=opt.camera_distance, pose_y=opt.global_y,
    #                      focal_length=opt.focal_length)
    smpl_render = Render(width=opt.image_width, height=opt.image_height, pose_y=opt.global_y)

    smpl_render.set_render(vertices=smpl_vertices, faces=faces, visual=smpl_uv_visual)

    smpl_norm_vertices = smpl_render.vertices

    smpl_render_uv = smpl_render.render_visual(flags=pyrender.RenderFlags.UV_RENDERING, face_id=uv_faceid)

    smpl_render.set_render(vertices=smpl_vertices, faces=faces)

    norm_vertices = smpl_render.vertices
    color = (norm_vertices * 255)
    concatenated_smpl = np.concatenate((norm_vertices, color), axis=1)
    ply_start = ply_start.format(norm_vertices.shape[0])

    with open(opt.save_loc, 'w') as write_file:
        write_file.write(ply_start)
        np.savetxt(write_file, concatenated_smpl, fmt=' '.join(['%0.8f'] * 3 + ['%d'] * 3))



def parse_args(args):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=32, help='# of samples of human poses')
    parser.add_argument('--n_views', type=int, default=32, help='# of global camera poses')
    parser.add_argument('--n_poses_on_gpu', type=int, default=32, help='# latentD sized vectors processed simulateneously')
    parser.add_argument('--camera_distance', type=float, default=3, help='distance from the camera in the camera space')
    parser.add_argument('--global_y', type=float, default=0, help='move the model in the up/down in the world space')
    parser.add_argument('--focal_length', type=float, default=1, help='focal length')
    parser.add_argument('--image_width', type=int, default=64, help='image width')
    parser.add_argument('--image_height', type=int, default=64, help='image height')
    parser.add_argument('--znear', type=float, default=0, help='near plane')
    parser.add_argument('--zfar', type=float, default=10, help='far plane')
    parser.add_argument('--save_loc', type=str, required=True, help='save_loc')
    return parser.parse_args(args)


if __name__ == '__main__':
    # opt = parse_args(sys.argv[1:])
    opt = parse_args([
        '--n_samples=10',
        '--camera_distance=2.8',
        '--global_y=0.0',
        '--focal_length=1.09375',
        '--image_width=340',
        '--image_height=340',
        '--znear=0.05',
        '--zfar=5.05',
        '--save_loc=../3d_data/smpl_NOC_vertices.ply'
    ])
    main(opt)