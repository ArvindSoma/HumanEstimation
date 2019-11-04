import sys
import cv2
import numpy as np
from scipy import io as io
import torch
from models.textures import Texture
import pickle

import trimesh
import argparse

from utils.render_utils import Render, match_faces

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
    uv_vertices = (uv * 2) - 1
    # with open('../3d_data/nongrey_male_0110.jpg', 'rb') as file:
    texture = cv2.imread('../3d_data/nongrey_male_0110.jpg')

    output = model(return_verts=True)
    vertices = output.vertices.detach().cpu().numpy().squeeze()

    smpl_vertices = np.array([vertices[i] for i in smpl['v_extended']])

    smpl_uv_visual = trimesh.visual.TextureVisuals(uv=uv, image=texture)

    # smpl_render = Render(width=opt.image_width, height=opt.image_height, pose_y=opt.global_y)

    smpl_render = Render(width=opt.image_width, height=opt.image_height,
                         camera_distance=opt.camera_distance, pose_y=opt.global_y,
                         focal_length=opt.focal_length)

    smpl_render.set_render(vertices=smpl_vertices, faces=faces)

    smpl_norm_vertices = smpl_render.vertices

    smpl_uv = smpl_render.render_interpolate(vertices=uv, skip_cull=False)
    smpl_noc = smpl_render.render_interpolate(vertices=smpl_norm_vertices).interpolated

    smpl_class_id = match_faces(smpl_uv.triangle, uv_faceid)
    smpl_class_id = smpl_class_id.reshape(smpl_class_id.shape + (1,))
    smpl_class_id[smpl_class_id == -1] = 0

    smpl_uv_stack = np.array([]).reshape((0, opt.image_height, opt.image_width, 2))
    aggregate_textures = np.array([], dtype='float64').reshape((0, 3, opt.image_height, opt.image_width))

    uv_render = Render(width=opt.image_width, height=opt.image_height)
    store_aggregate = []

    for idx in range(1, 25):
        # face_select = faces[uv_faceid[:, 0] == idx, :]
        id_select = np.unique(np.hstack(
            np.where(faces == vert)[0] for face in faces[uv_faceid[:, 0] == idx, :] for vert in face))

        face_select = faces[id_select, :]

        uv_render.set_render(vertices=uv_vertices, faces=face_select, normalize=False)
        # out_view = uv_render.render_visual(flags=pyrender.RenderFlags.SKIP_CULL_FACES)
        out_view = np.flip(uv_render.render_interpolate(vertices=smpl_norm_vertices).interpolated.transpose([2, 0, 1]),
                           axis=1)
        aggregate_textures = np.concatenate([aggregate_textures, out_view.reshape((1,) + out_view.shape)])
        smpl_uv_stack = np.concatenate([smpl_uv_stack, (
                    smpl_uv.interpolated[:, :, :-1] * (smpl_class_id.repeat([2], axis=-1) == idx)).reshape(
            (1,) + smpl_uv.interpolated[:, :, :-1].shape)])

        store_aggregate.append(
            [aggregate_textures[idx - 1, :, j, i] for i in range(aggregate_textures[idx - 1].shape[1]) for j in
             range(aggregate_textures[idx - 1].shape[2])])

        # cv2.imshow("Part Texture", aggregate_textures[idx - 1].transpose([1, 2, 0]))
        # cv2.waitKey(0)
    store_aggregate = np.concatenate(store_aggregate, axis=0)
    print(store_aggregate.shape)
    ply_start = ply_start.format(store_aggregate.shape[0])
    color = (store_aggregate * 255).astype('uint8')
    concatenated_smpl = np.concatenate((store_aggregate, color), axis=1)
    # with open(opt.save_loc, 'w') as write_file:
    #     write_file.write(ply_start)
    #     np.savetxt(write_file, concatenated_smpl, fmt=' '.join(['%0.8f'] * 3 + ['%d'] * 3))

    texture_map = torch.from_numpy(aggregate_textures)

    smpl_uv_stack = torch.from_numpy((smpl_uv_stack * 2) - 1)

    store_aggregate = np.concatenate(store_aggregate, axis=0)

    output_textured_uv = 0

    for idx in range(0, 24):
        output_textured_uv += torch.nn.functional.grid_sample(texture_map[idx: idx + 1], smpl_uv_stack[idx: idx + 1],
                                                              mode='bilinear', padding_mode='border')

    output_textured_uv = output_textured_uv[0].cpu().numpy().transpose([1, 2, 0])
    cv2.imshow("Resampled UV", output_textured_uv)
    cv2.imwrite('../saves/checks/sampled_NOC_render.jpg', (output_textured_uv * 255).astype('uint8'))
    cv2.imshow("NOC", smpl_noc)
    cv2.imwrite('../saves/checks/NOC_render.jpg', (smpl_noc * 255).astype('uint8'))
    cv2.imshow("Rendered UV", smpl_uv.interpolated)
    cv2.imwrite('../saves/checks/UV_render.jpg', (smpl_uv.interpolated * 255).astype('uint8'))
    cv2.imshow("Rendered Class", smpl_class_id.astype('uint8'))
    print("Image mean: ", np.mean(cv2.subtract(output_textured_uv.astype('float64'), smpl_noc.astype('float64'))))
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
    parser.add_argument('--save_loc', type=str, required=True, help='save_loc')
    return parser.parse_args(args)


if __name__ == '__main__':
    # opt = parse_args(sys.argv[1:])
    opt = parse_args([
        '--n_samples=10',
        '--camera_distance=2.3',
        '--global_y=0.15',
        '--focal_length=1.09375',
        '--image_width=340',
        '--image_height=340',
        '--znear=0.05',
        '--zfar=5.05',
        '--out_dir=./smplx-uvs',
        '--save_loc=../3d_data/NOC_check.ply'
    ])
    main(opt)
