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
from scipy.spatial.transform import Rotation as R
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
    save_annotation_file = opt.output

    if not os.path.exists(save_annotation_file):
        os.mkdir(save_annotation_file)
    annotation_dict = {'minival': [COCO(coco_folder + '/annotations/densepose_coco_2014_minival.json'),
                                   COCO(coco_folder + '/annotations/person_keypoints_val2014.json'),
                                   'val2014'],
                       'train': [COCO(coco_folder + '/annotations/densepose_coco_2014_train.json'),
                                 COCO(coco_folder + '/annotations/person_keypoints_train2014.json'),
                                 'train2014'],
                       'valminusminival': [COCO(coco_folder + '/annotations/densepose_coco_2014_valminusminival.json'),
                                           COCO(coco_folder + '/annotations/person_keypoints_val2014.json'),
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
    # texture = cv2.imread('../3d_data/nongrey_male_0110.jpg')

    seg_path = os.path.join(coco_folder, 'background')
    if not os.path.exists(seg_path):
        os.mkdir(seg_path)

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
    print('Shape of all vertices: ', store_aggregate.shape)
    ply_start = ply_start.format(store_aggregate.shape[0])
    color = (store_aggregate * 255).astype('uint8')
    concatenated_smpl = np.concatenate((store_aggregate, color), axis=1)
    with open('../3d_data/NOC_check.ply', 'w') as write_file:
        write_file.write(ply_start)
        np.savetxt(write_file, concatenated_smpl, fmt=' '.join(['%0.8f'] * 3 + ['%d'] * 3))

    texture_map = torch.from_numpy(aggregate_textures)
    texture_map = texture_map.clamp(min=0, max=1)

    print("SMPL textures loaded in memory.\n")

    #################################

    for key in annotation_dict:
        dp_coco = annotation_dict[key][0]
        person_coco = annotation_dict[key][1]
        parent_dir = annotation_dict[key][2]

        seg_key_path = os.path.join(seg_path, parent_dir)
        if not os.path.exists(seg_key_path):
            os.mkdir(seg_key_path)

        im_ids = dp_coco.getImgIds()
        # len_ids = len(im_ids)
        key_list = []
        for idx, im_id in enumerate(tqdm(im_ids, desc="Key [{}] Progress".format(key), ncols=100)):
            im_dict = {}
            im = dp_coco.loadImgs(im_id)[0]
            person_im = person_coco.loadImgs(im_id)[0]
            im_name = os.path.join(coco_folder, parent_dir, im['file_name'])
            image = cv2.imread(im_name)
            # im_dict['image'] = image

            im_dict['file_name'] = os.path.join(parent_dir, im['file_name'])

            ann_ids = dp_coco.getAnnIds(imgIds=im['id'])
            anns = dp_coco.loadAnns(ann_ids)
            person_anns = person_coco.loadAnns(ann_ids)

            im_dict['points'] = {}
            zero_im = np.zeros((image.shape[0], image.shape[1]))

            person_seg = np.zeros((image.shape[0], image.shape[1]))
            point_dict = im_dict['points']
            point_dict['yx'] = np.array([]).reshape((0, 2))
            point_dict['iuv'] = np.array([]).reshape((0, 3))

            # xy_mask = np.zeros((image.shape[0], image.shape[1], 1))
            # zero_point_iuv = np.zeros_like(image)
            # zero_point_uv = np.zeros((24, image.shape[0], image.shape[1], 2))
            # index_count = np.array([0] * 24)
            # iuv_values = np.zeros((24, 2000, 2))

            for person_ann in person_anns:
                person_seg += person_coco.annToMask(person_ann)

            for ann in anns:

                if 'dp_masks' in ann.keys():

                    bbr = np.array(ann['bbox'])
                    # print(bbr)
                    mask = GetDensePoseMask(ann['dp_masks'])
                    x1, y1, x2, y2 = bbr[0], bbr[1], bbr[0] + bbr[2], bbr[1] + bbr[3]
                    x2 = min([x2, image.shape[1]])
                    y2 = min([y2, image.shape[0]])

                    mask_im = cv2.resize(mask, (int(x2 - x1), int(y2 - y1)), interpolation=cv2.INTER_NEAREST)
                    # mask_bool = np.tile((mask_im == 0)[:, :, np.newaxis], [1, 1, 3])
                    # zero_im[y1:y2, x1:x2] += mask_im
                    zero_im[int(y1): (int(y1) + int(y2 - y1)), int(x1): (int(x1) + int(x2 - x1))] += mask_im

                    # Stretch the points to current box.
                    img_x = np.array(ann['dp_x']) / 255. * bbr[2] + x1
                    img_y = np.array(ann['dp_y']) / 255. * bbr[3] + y1
                    # img_x = img_x.astype('int') - 1 * (img_x >= image.shape[1])
                    # img_x = img_x - 1 * (img_x >= image.shape[1])
                    # img_y = img_y.astype('int') - 1 * (img_y >= image.shape[0])
                    img_x = img_x.clip(0, image.shape[1] - 1)
                    img_y = img_y.clip(0, image.shape[0] - 1)
                    # if img_x.any() == 0 or img_y.any() == 0:
                    #     print ("It reached 0!!")

                    point_dict['yx'] = np.concatenate([point_dict['yx'], np.array([img_y, img_x]).T])

                    point_i = np.array(ann['dp_I']).astype('int')
                    point_u = np.array(ann['dp_U'])
                    point_v = np.array(ann['dp_V'])
                    point_dict['iuv'] = np.concatenate((point_dict['iuv'], np.array([point_i, point_u, point_v]).T))

                    # zero_point_iuv[img_y, img_x, :] = np.array([point_i, point_u, point_v]).T
                    # xy_mask[img_y, img_x, 0] = 1
                    # zero_point_uv[point_i - 1, img_y, img_x] = np.array([point_u, point_v]).T
                    #
                    # iuv_values[point_i.astype('int') - 1, index_count[point_i.astype('int') - 1], :] = np.array(
                    #     [point_u, point_v]).T
                    # index_count[point_i.astype('int') - 1] += 1


            # uv_stack = torch.from_numpy((zero_point_uv * 2) - 1)
            # uv_stack = uv_stack.clamp(min=-1, max=1)

            # output_noc = 0
            # for jdx in range(0, 24):
            #     output_noc += torch.nn.functional.grid_sample(texture_map[jdx: jdx + 1],
            #                                                           uv_stack[jdx: jdx + 1],
            #                                                           mode='bilinear', padding_mode='border')
            #
            # output_noc = output_noc[0].cpu().numpy().transpose([1, 2, 0])

            output_noc = np.zeros_like(point_dict['iuv'])
            # if point_dict['iuv'].shape[0] > 0:
            for jdx in range(0, 24):
                zero_point_dict = np.zeros((point_dict['iuv'].shape[0], ) + (2,))
                zero_point_dict[point_dict['iuv'][:, 0] == (jdx + 1), :] = point_dict['iuv'][
                                                                           point_dict['iuv'][:, 0] == (jdx + 1), 1:]

                zero_point_dict = zero_point_dict.reshape((1, 1,) + zero_point_dict.shape)

                zero_point_dict = torch.from_numpy((zero_point_dict * 2) - 1)
                # zero_point_dict = zero_point_dict.clamp(-1, 1)

                output_noc_temp = torch.nn.functional.grid_sample(texture_map[jdx: jdx + 1],
                                                                  zero_point_dict,
                                                                  mode='bilinear', padding_mode='border')

                output_noc_temp = output_noc_temp.cpu().numpy().transpose([0, 2, 3, 1]).reshape(output_noc.shape)

                output_noc[point_dict['iuv'][:, 0] == (jdx + 1), :] = output_noc_temp[
                                                                      point_dict['iuv'][:, 0] == (jdx + 1), :]

            zero_im = zero_im + person_seg

            zero_im = (zero_im > 0).astype('float32')
            zero_im = cv2.dilate(zero_im, kernel, iterations=1)

            cv2.imwrite(os.path.join(seg_key_path,  im['file_name']), (zero_im == 0.0).astype('uint8'))
            if point_dict['yx'].shape[0] > 0:
                if point_dict['yx'].min() == 0:
                    print('Min 0!')
                uniques, counts = np.unique(point_dict['yx'], axis=0, return_counts=True)
                if counts.max() > 1:
                    print("Lot of YXs!.")

            # point_dict['noc'] = output_noc[point_dict['yx'][:, 0], point_dict['yx'][:, 1], :]
            point_dict['noc'] = output_noc
            # if point_dict['noc'].shape[0] > 0:
            #     if point_dict['noc'].min() < smpl_norm_vertices.min() or point_dict['noc'].max() > smpl_norm_vertices.max():
            #         print("Error at idx {}.".format(idx))
            # print(np.min(point_dict['yx']))
            key_list.append(im_dict)

            # cv2.imshow("Image", image)
            # cv2.imshow("Background Image", (zero_im == 0.0).astype('uint8') * 255)
            # cv2.imshow("IUV", (zero_point_iuv * 30).astype('uint8'))
            # cv2.imshow("NOC sampled", (output_noc * 255).astype('uint8'))
            #
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
