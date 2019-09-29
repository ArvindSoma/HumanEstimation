"""
Mesh Normalization
"""

import os
import sys
import cv2
import numpy as np
from scipy import io as io
import torch
import pickle

import trimesh
import argparse

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

    global_tr = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    # set up the rendering objects
    focal_length = opt.focal_length * opt.image_height
    # mesh_camera = pyrender.IntrinsicsCamera(focal_length, focal_length, opt.image_width / 2, opt.image_height / 2,
    #                                    opt.znear, opt.zfar)
    mesh_camera = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0, znear=0.05)
    camera = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0, znear=0.05)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.05],
        [0.0, 0.0, 0.0, 1.0]
    ])

    mesh_tr = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, opt.global_y + 0.11],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    mesh_camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, opt.camera_distance],
        [0.0, 0.0, 0.0, 1.0]
    ])

    render = pyrender.OffscreenRenderer(opt.image_width, opt.image_height)


    output = model(return_verts=True)
    vertices = output.vertices.detach().cpu().numpy().squeeze()

    mesh_verts = np.array([vertices[i] for i in smpl['v_extended']])
    visual_check = trimesh.visual.TextureVisuals(uv=uv, image=texture)

    tri_mesh_scene = trimesh.Trimesh(vertices=mesh_verts, faces=faces, visual=visual_check)

    mesh_body = pyrender.Mesh.from_trimesh(tri_mesh_scene)
    mesh_scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5], bg_color=[-1.0, -1.0, -1.0])

    mesh_scene.add(mesh_body, pose=mesh_tr)
    mesh_scene.add(mesh_camera, pose=mesh_camera_pose)

    rendered_uv, depth = render.render(scene=mesh_scene, flags=pyrender.RenderFlags.UV_RENDERING)
    rendered_uv = rendered_uv.copy()

    mask = rendered_uv[:, :, 2] != -1.
    temp_2 = rendered_uv[:, :, 2]
    temp_2[mask] = np.take(uv_faceid, temp_2[mask].astype('int'))
    rendered_uv[:, :, 2] = temp_2

    cv2.imshow('UV', rendered_uv)
    bounds = tri_mesh_scene.bounding_box_oriented.extents

    mesh_verts -= mesh_scene.centroid
    mesh_verts /= bounds
    # mesh_verts *= 2
    mesh_verts = mesh_verts + 1/2

    face_select = faces[uv_faceid[:, 0] == 1]

    # verts = np.concatenate((uv, np.ones(uv.shape[:2] + (1,))), axis=2)

    # uv[:, 2] = 1
    verts = (uv * 2) - 1
    visual = trimesh.visual.ColorVisuals(vertex_colors=uv)
    tri_mesh = trimesh.Trimesh(vertices=verts, faces=face_select, visual=visual)
    # tri_mesh

    mesh = pyrender.Mesh.from_trimesh(tri_mesh)

    # tri_mesh.show()

    scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5], bg_color=[-1.0, -1.0, -1.0])
    scene.add(mesh, pose=global_tr)
    scene.add(camera, pose=camera_pose)


    rendered_color_visual, depth = render.render(scene=scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
    # pyrender.Viewer(scene, render_flags={'cull_faces': False})
    cv2.imshow('Part UV', rendered_color_visual)
    # cv2.waitKey(0)

    rendered_interp, _ = render.render(scene=scene, flags=pyrender.RenderFlags.BARYCENTRIC_COORDINATES | pyrender.RenderFlags.SKIP_CULL_FACES)
    tri_id, _ = render.render(scene=scene, flags=pyrender.RenderFlags.TRIANGLE_ID_RENDERING | pyrender.RenderFlags.SKIP_CULL_FACES)

    vertex_stream = np.take(mesh_verts, face_select, axis=0)
    tri_id = tri_id[:, :, 0]

    rendered_interp = rendered_interp.reshape(rendered_interp.shape + (1,)).repeat([3], axis=-1)
    out_view = vertex_stream[tri_id.astype('int')] * rendered_interp
    out_view = out_view.sum(axis=-2)


    # rendered_uv[rendered_uv == -1] = 0
    # rendered_uv[:, :, 2] /= 255
    out_view[rendered_color_visual < 0] = 0

    # cv2.imwrite('../saves/checks/mesh_normalized_uv.jpg', (rendered_uv * 255).astype('uint8'))
    cv2.imshow('Coords', out_view)
    cv2.imwrite('../saves/checks/mesh_uv_render.jpg', (out_view * 255).astype('uint8'))
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
        '--global_y=0.15',
        '--focal_length=1.09375',
        '--image_width=340',
        '--image_height=340',
        '--znear=0.05',
        '--zfar=5.05',
        '--out_dir=./smplx-uvs'
    ])
    main(opt)

