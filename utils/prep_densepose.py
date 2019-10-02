import os
import sys
import argparse
import pickle
import numpy as np
import cv2
import torch
from scipy import io as io

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
    coco_folder = os.environ['COCO']
    save_annotation_file = opt.output

    if not os.path.exists(save_annotation_file):
        os.mkdir(save_annotation_file)
    annotation_dict = {'minival': [COCO(coco_folder + '/annotations/densepose_coco_2014_minival.json'),
                                   'val2014'],
                       'train': [COCO(coco_folder + '/annotations/densepose_coco_2014_train.json'),
                                 'train2014'],
                       'valminusminival': [COCO(coco_folder + '/annotations/densepose_coco_2014_valminusminival.json'),
                                           'val2014'],
                       'test': [COCO(coco_folder + '/annotations/densepose_coco_2014_test.json'),
                                'test2014']}

    #################################

    # SMPL prep

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

    smpl_render = Render(width=opt.image_width, height=opt.image_height)

    smpl_render.set_render(vertices=smpl_vertices, faces=faces)

    smpl_norm_vertices = smpl_render.vertices

    # smpl_uv = smpl_render.render_interpolate(vertices=uv, skip_cull=False)
    # smpl_noc = smpl_render.render_interpolate(vertices=smpl_norm_vertices).interpolated

    aggregate_textures = np.array([], dtype='float64').reshape((0, 3, opt.image_height, opt.image_width))

    uv_render = Render(width=opt.image_width, height=opt.image_height)

    for idx in range(1, 25):
        # face_select = faces[uv_faceid[:, 0] == idx, :]
        id_select = np.unique(np.hstack(
            np.where(faces == vert)[0] for face in faces[uv_faceid[:, 0] == idx, :] for vert in face))

        face_select = faces[id_select, :]

        uv_render.set_render(vertices=uv_vertices, faces=face_select, normalize=False)

        out_view = np.flip(uv_render.render_interpolate(vertices=smpl_norm_vertices).interpolated.transpose([2, 0, 1]),
                           axis=1)
        aggregate_textures = np.concatenate([aggregate_textures, out_view.reshape((1,) + out_view.shape)])

    texture_map = torch.from_numpy(aggregate_textures)

    print("SMPL textures loaded in memory.\n")

    #################################

    for key in annotation_dict:
        dp_coco = annotation_dict[key][0]
        parent_dir = annotation_dict[key][1]
        im_ids = dp_coco.getImgIds()
        len_ids = len(im_ids)
        key_list = []
        for idx, im_id in enumerate(tqdm(im_ids, desc="Key [{}] Progress".format(key), ncols=100)):
            im_dict = {}
            im = dp_coco.loadImgs(im_id)[0]
            im_name = os.path.join(coco_folder, parent_dir, im['file_name'])
            image = cv2.imread(im_name)
            im_dict['image'] = image

            ann_ids = dp_coco.getAnnIds(imgIds=im['id'])
            anns = dp_coco.loadAnns(ann_ids)

            im_dict['points'] = {}
            zero_im = np.zeros((image.shape[0], image.shape[1]))
            point_dict = im_dict['points']
            point_dict['xy'] = np.array([], dtype='int').reshape((0, 2))
            point_dict['iuv'] = np.array([]).reshape((0, 3))

            xy_mask = np.zeros((image.shape[0], image.shape[1], 1))
            zero_point_iuv = np.zeros_like(image)
            zero_point_uv = np.zeros((24, image.shape[0], image.shape[1], 2))
            for ann in anns:
                ann_dict = {}
                if 'dp_masks' in ann.keys():
                    bbr = (np.array(ann['bbox'])).astype('int')
                    mask = GetDensePoseMask(ann['dp_masks'])
                    x1, y1, x2, y2 = bbr[0], bbr[1], bbr[0] + bbr[2], bbr[1] + bbr[3]
                    x2 = min([x2, image.shape[1]])
                    y2 = min([y2, image.shape[0]])

                    mask_im = cv2.resize(mask, (int(x2 - x1), int(y2 - y1)), interpolation=cv2.INTER_NEAREST)
                    mask_bool = np.tile((mask_im == 0)[:, :, np.newaxis], [1, 1, 3])
                    zero_im[y1:y2, x1:x2] += mask_im

                    img_x = np.array(ann['dp_x']) / 255. * bbr[2] + x1    # Stretch the points to current box.
                    img_y = np.array(ann['dp_y']) / 255. * bbr[3] + y1    # Stretch the points to current box.
                    img_x = img_x.astype('int') - 1 * (img_x >= image.shape[1])
                    img_y = img_y.astype('int') - 1 * (img_y >= image.shape[0])


                    point_dict['xy'] = np.concatenate([point_dict['xy'], np.array([img_x, img_y]).T])

                    point_i = np.array(ann['dp_I']).astype('int')
                    point_u = np.array(ann['dp_U'])
                    point_v = np.array(ann['dp_V'])
                    point_dict['iuv'] = np.concatenate((point_dict['iuv'], np.array([point_i, point_u, point_v]).T))

                    zero_point_iuv[img_y, img_x, :] = np.array([point_i, point_u, point_v]).T

                    xy_mask[img_y, img_x, 0] = 1

                    zero_point_uv[point_i - 1, img_y, img_x] = np.array([point_u, point_v]).T

            uv_stack = torch.from_numpy((zero_point_uv * 2) - 1)

            output_noc = 0
            for idx in range(0, 24):
                output_noc += torch.nn.functional.grid_sample(texture_map[idx: idx + 1],
                                                                      uv_stack[idx: idx + 1],
                                                                      mode='bilinear', padding_mode='border')

            output_noc = output_noc[0].cpu().numpy().transpose([1, 2, 0])

            point_dict['noc'] = output_noc[point_dict['xy'][:, 1], point_dict['xy'][:, 0], :]

            key_list.append(im_dict)

            # cv2.imshow("Image", image)
            # cv2.imshow("IUV", (zero_point_iuv * 30).astype('uint8'))
            # cv2.imshow("NOC sampled", (output_noc * 255).astype('uint8'))
            # cv2.waitKey(0)

            # progress_bar(idx + 1, len_ids, prefix="Progress for {}:".format(key), suffix="Complete")

        save_file = os.path.join(save_annotation_file, '{}.pkl'.format(key))
        with open(save_file, 'wb') as write_file:
            pickle.dump(key_list, write_file, protocol=pickle.HIGHEST_PROTOCOL)



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
    parser.add_argument('--output', type=str, default='./data/new_annotation.pkl', help='# of samples of human poses')
    parser.add_argument('--image_width', type=int, default=64, help='image width')
    parser.add_argument('--image_height', type=int, default=64, help='image height')

    return parser.parse_args(args)


if __name__ == '__main__':
    opt = parse_args(['--output=../data/dp_annotation',
                      '--image_width=340',
                      '--image_height=340'])
    main(opt)
