import os
import sys
import argparse
import pickle
import numpy as np
import cv2
import torch
from scipy import io as io
import tqdm

from pycocotools.coco import COCO
import pycocotools.mask as mask_util

import trimesh
from tqdm import tqdm

from utils.render_utils import Render, match_faces


from external.smplx.smplx import body_models
sys.path.insert(0, '../external/pyrender')


def GetDensePoseMask(Polys):
    MaskGen = np.zeros([256,256])
    for i in range(1,15):
        if(Polys[i-1]):
            current_mask = mask_util.decode(Polys[i-1])
            MaskGen[current_mask>0] = i
    return MaskGen


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

    coco_folder = os.environ['COCO']
    # save_annotation_file = opt.output
    #
    # if not os.path.exists(save_annotation_file):
    #     os.mkdir(save_annotation_file)
    # annotation_dict = {'minival': [COCO(coco_folder + '/annotations/densepose_coco_2014_minival.json'),
    #                                COCO(coco_folder + '/annotations/person_keypoints_val2014.json'),
    #                                'val2014'],
    #                    # 'train': [COCO(coco_folder + '/annotations/densepose_coco_2014_train.json'),
    #                    #           COCO(coco_folder + '/annotations/person_keypoints_train2014.json'),
    #                    #           'train2014'],
    #                    # 'valminusminival': [COCO(coco_folder + '/annotations/densepose_coco_2014_valminusminival.json'),
    #                    #                     COCO(coco_folder + '/annotations/person_keypoints_val2014.json'),
    #                    #                     'val2014'],
    #                    # 'test': [COCO(coco_folder + '/annotations/densepose_coco_2014_test.json'),
    #                    #          'test2014']
    #                    }

    #################################

    # SMPL prep
    coco = COCO(coco_folder + '/annotations/densepose_coco_2014_minival.json')
    model = body_models.create(model_path='../3d_data/models', model_type='smpl', gender='male', ext='pkl')
    smpl = pickle.load(open('../3d_data/densepose_uv.pkl', 'rb'))
    faces = np.array(smpl['f_extended'], dtype=np.int64).reshape((-1, 3))
    uv_faceid = io.loadmat('../3d_data/DensePoseData/UV_data/UV_Processed.mat')['All_FaceIndices']
    uv = smpl['uv']
    uv_vertices = (uv * 2) - 1
    # with open('../3d_data/nongrey_male_0110.jpg', 'rb') as file:
    # texture = cv2.imread('../3d_data/nongrey_male_0110.jpg')

    if not os.path.exists(opt.output):
        os.mkdir(opt.output)

    file_list = [os.path.join(opt.input, f) for f in os.listdir(opt.input) if
                 os.path.isfile(os.path.join(opt.input, f)) and f.endswith('.pkl')]

    output = model(return_verts=True)
    vertices = output.vertices.detach().cpu().numpy().squeeze()

    smpl_vertices = np.array([vertices[i] for i in smpl['v_extended']])

    smpl_render = Render(width=opt.image_width, height=opt.image_height)

    smpl_render.set_render(vertices=smpl_vertices, faces=faces)

    smpl_norm_vertices = smpl_render.vertices

    # smpl_uv = smpl_render.render_interpolate(vertices=uv, skip_cull=False)
    # smpl_noc = smpl_render.render_interpolate(vertices=smpl_norm_vertices).interpolated

    aggregate_textures = np.array([], dtype='float64').reshape((0, 3, opt.image_height, opt.image_width))

    uv_render = Render(width=opt.image_width, height=opt.image_height)

    kernel = np.ones((3, 3), np.uint8)

    filter_size = 9

    save_texture_dir = '../3d_data/textures'
    if not os.path.exists(save_texture_dir):
        os.mkdir(save_texture_dir)

    store_aggregate = []
    for idx in range(1, 25):
        # face_select = faces[uv_faceid[:, 0] == idx, :]
        id_select = np.unique(np.hstack(
            np.where(faces == vert)[0] for face in faces[uv_faceid[:, 0] == idx, :] for vert in face))

        face_select = faces[id_select, :]

        uv_render.set_render(vertices=uv_vertices, faces=face_select, normalize=False)

        out_view = np.flip(uv_render.render_interpolate(vertices=smpl_norm_vertices).interpolated.transpose([2, 0, 1]),
                           axis=1)

        out_view[out_view < 0] = 0
        # cv2.imshow("Out_view1", out_view.transpose([1, 2, 0]))
        cv2.imwrite(os.path.join(save_texture_dir, 'texture_{}.png'.format(idx)),
                    (out_view.transpose([1, 2, 0]) * 255).astype('uint8'))

        new_view = np.zeros_like(out_view)
        out_view = np.pad(out_view, pad_width=[(0, 0), (4, 4), (4, 4)], mode='constant')

        for mdx in range(out_view.shape[1] - filter_size + 1):
            for ndx in range(out_view.shape[2] - filter_size + 1):
                if (out_view[:,  mdx + 4, ndx + 4] > 0).any():
                    new_view[:, mdx, ndx] = out_view[:, mdx + 4, ndx + 4]
                else:
                    select_filter = out_view[:, mdx: mdx + filter_size, ndx: ndx + filter_size]
                    select_filter = select_filter[:, np.sum(select_filter, axis=0) > 0]
                    if select_filter.shape[1] == 0:
                        new_view[:, mdx, ndx] = 0
                    else:
                        new_view[:, mdx, ndx] = np.sum(select_filter, axis=1) / select_filter.shape[1]

        out_view = new_view
        cv2.imwrite(os.path.join(save_texture_dir, 'texture_extruded_{}.png'.format(idx)),
                    (out_view.transpose([1, 2, 0]) * 255).astype('uint8'))
        # cv2.imshow("Out_view2", out_view.transpose([1, 2, 0]))
        # cv2.waitKey(0)
        if out_view.max() > 1 or out_view.min() < 0:
            print('Error!!')
        aggregate_textures = np.concatenate([aggregate_textures, out_view.reshape((1,) + out_view.shape)])
        store_aggregate.append(
            [aggregate_textures[idx - 1, :, j, i] for i in range(aggregate_textures[idx - 1].shape[1]) for j in
             range(aggregate_textures[idx - 1].shape[2]) if aggregate_textures[idx - 1, :, j, i].all() != 0])

    # cv2.destroyWindow("Out_view")

    store_aggregate = np.concatenate(store_aggregate, axis=0)
    texture_map = torch.from_numpy(aggregate_textures)
    texture_map = texture_map.clamp(min=0, max=1)

    print('Shape of all vertices: ', store_aggregate.shape)
    print("SMPL textures loaded in memory.\n")

    #################################

    im_ids = coco.getImgIds()

    for im_id in tqdm(im_ids):
        im = coco.loadImgs(im_id)[0]
        file = im['file_name']
        with open(os.path.join(file), 'rb') as ifile:
            data = pickle.load(ifile)
        iuv_image = data['iuv']

        iuv_image[:, :, 1:] /= 255
        save_path = os.path.join(opt.output, os.path.splitext(os.path.basename(file))[0])
        uv_image = iuv_image[:, :, 1:]
        i_image = iuv_image[:, :, 0].astype('int32')
        # iuv_image = torch.from_numpy(iuv_image)

        output_noc = np.zeros_like(iuv_image)
        # # if iuv_image.shape[0] > 0:
        for jdx in range(0, 24):
            zero_uv = np.zeros_like(uv_image)
            if i_image[i_image == (jdx + 1)].shape[0] > 0:
                # print(jdx)
                zero_uv[i_image == (jdx + 1), :] = uv_image[i_image == (jdx + 1), :]

                zero_uv = zero_uv.reshape((1,) + zero_uv.shape)

                zero_uv = torch.from_numpy((zero_uv * 2) - 1)
                zero_uv = zero_uv.clamp(-1, 1)

                output_noc_temp = torch.nn.functional.grid_sample(input=texture_map[jdx: jdx + 1],
                                                                  grid=zero_uv,
                                                                  mode='bilinear', padding_mode='border')

                output_noc_temp = output_noc_temp.squeeze(0).cpu().numpy().transpose([1, 2, 0])

                output_noc[i_image == (jdx + 1), :] = output_noc_temp[i_image == (jdx + 1), :]
            # cv2.imshow('OUTPUT_NOC', (output_noc * 255).astype('uint8'))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        noc_dict = {'noc': output_noc, 'iuv': data['iuv'], 'inds': data['inds']}
        with open(os.path.join(opt.output, os.path.basename(file)), 'wb') as out_file:
            pickle.dump(noc_dict, out_file)


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
    parser.add_argument('--output', type=str, default='COCO-NOC', help='Location of COCO densepose to NOC dir')
    parser.add_argument('--input', type=str, default='COCO', help='Location of COCO densepose dir')
    parser.add_argument('--image_width', type=int, default=64, help='image width')
    parser.add_argument('--image_height', type=int, default=64, help='image height')

    return parser.parse_args(args)


if __name__ == '__main__':
    opt = parse_args(['--input=/home/arvindsoma/Documents/Datasets/COCO_2014/minival2014',
                      '--output=/home/arvindsoma/Documents/Datasets/COCO_2014/densepose_noc/val2014',
                      '--image_width=340',
                      '--image_height=340'])
    main(opt)
