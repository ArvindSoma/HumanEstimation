import os
import sys
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy import io as io
import torch
from models.textures import Texture
import pickle

import trimesh
import argparse
from utils.render_utils import look_at

sys.path.insert(0, '../external/pyrender')
import pyrender


import os
import sys
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy import io as io
import torch
from models.textures import Texture
import pickle

import trimesh
import argparse
from utils.render_utils import look_at

from external.smplx.smplx import body_models
sys.path.insert(0, '../external/pyrender')
import pyrender


def main(opt):
    if not os.path.exists(opt.output):
        os.mkdir(opt.output)

    file_size = len([file for file in os.listdir(opt.input) if os.path.isfile(os.path.join(opt.input, file))]) // 3

    camera_locations = np.array([[0.5, 0.5, 10.5],
                                 [10.0, 0.5, 4.5],
                                 [-10.0, 0.5, 4.5],
                                 [0.5, 0.5, -10.5],
                                 [10, 10, 10]])

    for idx in range(file_size):
        file_list = ["../3d_data/smpl_NOC_vertices.ply", "Ground_truth_{}.ply".format(idx), "Output_{}.ply".format(idx)]
        camera_list = []
        for jdx, file in enumerate(file_list):
            render_list = []
            if jdx == 0:
                path = file
            else:
                path = os.path.join(opt.input, file)
            tri_mesh = trimesh.load_mesh(path, file_type='ply')
            colors = np.hstack(
                (tri_mesh.colors.copy()[0], np.ones((tri_mesh.colors.copy()[0].shape[0], 1)) * 255)) / 255
            tri_mesh = trimesh.points.PointCloud(vertices=tri_mesh.vertices, colors=colors)
            pts = tri_mesh.vertices.copy()

            for cam_pose in camera_locations:
                camera = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0, znear=0.05)
                scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5])

                camera_pose = look_at(cam_pos_world=cam_pose, to_pos_world=np.array([0.5, 0.5, 0.5]))
                scene.add(camera, pose=camera_pose)
                if jdx == 0:
                    scene.add(pyrender.Mesh.from_points(pts, colors=colors))
                else:
                    for kdx, pose in enumerate(pts):
                        spr_tri = trimesh.creation.uv_sphere(radius=0.01)
                        spr_tri.visual.vertex_colors = colors[kdx, :]
                        # print(colors[kdx] / 255)
                        poses = np.eye(4)
                        poses[:3, 3] = pose
                        spr_py = pyrender.Mesh.from_trimesh(spr_tri)
                        scene.add(spr_py, pose=poses)

                render = pyrender.OffscreenRenderer(340, 340)
                rendered, _ = render.render(scene=scene, flags=pyrender.RenderFlags.RGBA)
                rendered = np.pad(array=rendered, pad_width=((5, 5), (5, 5), (0, 0)), mode='constant')
                render_list.append(rendered)

            render_list = np.hstack(render_list)

            camera_list.append(render_list)

        final_render = np.concatenate(camera_list, axis=0)

        # cv2.imshow("Final results", final_render)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        cv2.imwrite(os.path.join(opt.output, "Results_{}.png".format(idx)), final_render)

    return True


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
    parser.add_argument('--input', type=str, default='../3d_data', help='Location of GT and output data')
    parser.add_argument('--output', type=str, default='../3d_data/outputs_visualization', help='Location of viz dir')

    return parser.parse_args(args)


if __name__ == "__main__":
    opt = parse_args(['--input=../3d_data/sparse_sampled_test_Res50UNet_Dropout_2Head_fixed_points_1',
                      '--output=../3d_data/outputs_sparse_sampled_test_Res50UNet_Dropout_2Head_fixed_points_1'])
    main(opt)
