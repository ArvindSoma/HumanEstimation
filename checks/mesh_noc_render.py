"""
Mesh NOC Rendering
"""

import os
import sys
import cv2
import numpy as np
from scipy import io as io
import torch
from models.textures import Texture
import pickle

import trimesh
import argparse

from utils.render_utils import Render

from external.smplx.smplx import body_models
sys.path.insert(0, '../external/pyrender')
import pyrender


def main(opt):

    model = body_models.create(model_path='../3d_data/models', model_type='smpl', gender='male', ext='pkl')
    smpl = pickle.load(open('../3d_data/densepose_uv.pkl', 'rb'))
    faces = np.array(smpl['f_extended'], dtype=np.int64).reshape((-1, 3))
    uv_faceid = io.loadmat('../3d_data/DensePoseData/UV_data/UV_Processed.mat')['All_FaceIndices']
    uv = smpl['uv']

    # with open('../3d_data/nongrey_male_0110.jpg', 'rb') as file:
    texture = cv2.imread('../3d_data/nongrey_male_0110.jpg')

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



    smpl_render_norm = smpl_render.render_interpolate(vertices=uv).interpolated
    
    smpl_body_uv = smpl_render_uv[:, :, :2]
    
    smpl_body_class = smpl_render_uv[:, :, 2:3]
    
    smpl_uv_stack = np.array([]).reshape((0, opt.image_height, opt.image_width, 2))

    uv_vertices = (uv * 2) - 1
    uv_render = Render(width=opt.image_width, height=opt.image_height)

    aggregate_textures = np.array([], dtype='float64').reshape((0, 3, opt.image_height, opt.image_width))

    for idx in range(1, 4):
        face_select = faces[uv_faceid[:, 0] == idx]
        uv_visual = trimesh.visual.ColorVisuals(vertex_colors=uv)
        uv_render.set_render(vertices=uv_vertices, faces=face_select, visual=uv_visual, normalize=False)
        # out_view = uv_render.render_visual(flags=pyrender.RenderFlags.SKIP_CULL_FACES)
        out_view = uv_render.render_interpolate(vertices=smpl_norm_vertices).interpolated.transpose([2, 0, 1])
        aggregate_textures = np.concatenate([aggregate_textures, out_view.reshape((1,) + out_view.shape)])
        smpl_uv_stack = np.concatenate([smpl_uv_stack, (smpl_body_uv * (smpl_body_class.repeat([2], axis=-1) == idx)).reshape(
            (1,) + smpl_body_uv.shape)])

        cv2.imshow("Part body UV", np.concatenate([smpl_uv_stack[idx - 1], np.zeros(smpl_body_class.shape)], axis=-1))
        cv2.imshow("Part Texture", aggregate_textures[idx - 1].transpose([1, 2, 0]))
        cv2.waitKey(0)

    texture_map = torch.from_numpy(aggregate_textures)

    smpl_uv_stack = torch.from_numpy((smpl_uv_stack * 2) - 1)

    output_textured_uv = 0

    for idx in range(0, 3):
        output_textured_uv += torch.nn.functional.grid_sample(texture_map[idx: idx + 1], smpl_uv_stack[idx: idx + 1],
                                                              mode='bilinear', padding_mode='border')

    output_textured_uv = output_textured_uv[0].cpu().numpy().transpose([1, 2, 0])
    cv2.imshow("Resampled UV", output_textured_uv)
    cv2.imshow("Real Norm", smpl_render_norm)
    cv2.waitKey(0)


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
    parser.add_argument('--out_dir', type=str, required=True, help='directory to write results')
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
        '--out_dir=./smplx-uvs'
    ])
    main(opt)
