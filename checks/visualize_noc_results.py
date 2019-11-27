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
    # if not os.path.exists(opt.output):
    #     os.mkdir(opt.output)
    pcd = o3d.io.read_point_cloud("../3d_data/sparse_sampled_test_Res50UNet_Dropout_2Head_fixed_points_1/Output_0.ply")
    o3d.display([pcd])
    pcd.estimate_normals()

    # estimate radius for rolling ball
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,
                                                                           o3d.utility.DoubleVector([radius, radius * 2]))
    # tri_mesh = trimesh.points.PointCloud(vertices=np.asarray(mesh.vertices), np.asarray(mesh.triangles),
    #                            vertex_normals=np.asarray(mesh.vertex_normals))
    tri_mesh = trimesh.load_mesh(
        '../3d_data/sparse_sampled_test_Res50UNet_Dropout_2Head_fixed_points_1/Output_0.ply', file_type='ply')
    # tri_mesh.colors = tri_mesh.colors / 255
    colors = np.hstack((tri_mesh.colors.copy()[0], np.ones((tri_mesh.colors.copy()[0].shape[0], 1)) * 255)) / 255
    tri_mesh = trimesh.points.PointCloud(vertices=tri_mesh.vertices, colors=colors)

    # tri_mesh.show()
    # sm = trimesh.creation.uv_sphere(radius=0.1)
    # sm.visual.vertex_colors = [1.0, 0.0, 0.0]

    pts = tri_mesh.vertices.copy()
    # colors = tri_mesh.colors.copy()[0]
    pymesh = pyrender.Mesh.from_points(points=pts, colors=colors)

    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.5],
        [0.0, 1.0, 0.0, 0.5],
        [0.0, 0.0, 1.0, 10.5],
        [0.0, 0.0, 0.0, 1.0]
    ])

    mesh_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    camera = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0, znear=0.05)

    scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5])

    scene.add(camera, pose=camera_pose)
    # scene.add(pymesh, pose=mesh_pose)

    # m = trimesh.creation.uv_sphere(radius=0.01)
    # m.visual.vertex_colors = np.array([1, 1, 1])
    # poses = np.tile(np.eye(4), (len(pts), 1, 1))
    # poses[:, :3, 3] = pts
    # mn = pyrender.Mesh.from_trimesh(m, poses=poses)
    # scene.add(mn, pose=mesh_pose)
    # sm = [pyrender.Mesh.from_trimesh(trimesh.creation.uv_sphere(radius=0.1)) for pose in pts]
    # for idx, mesh in enumerate(sm):
    #     mesh.vertex_colors = colors[idx]
    #
    # [scene.add(mesh, pose=pts[idx]) for idx, mesh in enumerate(sm)]

    for idx, pose in enumerate(pts):
        spr_tri = trimesh.creation.uv_sphere(radius=0.01)
        spr_tri.visual.vertex_colors = colors[idx] * 255
        print(colors[idx] / 255)
        poses = np.eye(4)
        poses[:3, 3] = pose
        spr_py = pyrender.Mesh.from_trimesh(spr_tri)
        scene.add(spr_py, pose=poses)

    # sm = [pyrender.Mesh.from_trimesh(mesh, poses=pts[idx])]

    pyrender.Viewer(scene=scene, flags=pyrender.RenderFlags.NOT_RENDER_GEOM | pyrender.RenderFlags.RGBA,
                    render_flags={'point_size': 10.0},
                    viewer_flags={'rotate_axis': [0.5, 0.5, 0.5],
                                  'view_center': [0.5, 0.5, 0.5],
                                  'use_raymond_lighting': True
                                  })

    render = pyrender.OffscreenRenderer(640, 640)
    color1, depth = render.render(scene=scene, flags=pyrender.RenderFlags.RGBA)

    scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5])
    # rot = R.from_euler(seq='zxy', angles=[0, 0, 90]).as_dcm()
    # camera_pose[:3, :3] = rot
    # camera_pose[0, 3] = 10.5
    # camera_pose[1, 3] = 0.5
    # camera_pose[2, 3] = -4.5
    camera_pose = look_at(np.array([10, 0.5, 4.5]), np.array([0.5, 0.5, 0.5]))
    camera = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0, znear=0.05)
    scene.add(camera, pose=camera_pose)

    for idx, pose in enumerate(pts):
        spr_tri = trimesh.creation.uv_sphere(radius=0.01)
        spr_tri.visual.vertex_colors = colors[idx] * 255
        # print(colors[idx] / 255)
        poses = np.eye(4)
        poses[:3, 3] = pose
        spr_py = pyrender.Mesh.from_trimesh(spr_tri)
        scene.add(spr_py, pose=poses)

    pyrender.Viewer(scene=scene, flags=pyrender.RenderFlags.RGBA,
                    render_flags={'point_size': 10.0},
                    viewer_flags={'rotate_axis': [0.5, 0.5, 0.5],
                                  'view_center': [0.5, 0.5, 0.5],
                                  'use_raymond_lighting': True})

    color2, depth = render.render(scene=scene, flags=pyrender.RenderFlags.RGBA)
    cv2.imshow("image1", color1)
    cv2.imshow("image2", color2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # mesh = trimesh.load(opt.input + '/sparse_sampled_test_Res50UNet_Dropout_2Head_fixed_points_1/Ground_truth_0.ply',
    #                     file_type='ply')

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
    opt = parse_args(['--input=../3d_data',
                      '--output=../3d_data/outputs_visualization'])
    main(opt)
