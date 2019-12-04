import os
import sys
from PIL import Image
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
from io import BytesIO
import base64

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
    scale = 1.75

    camera_locations = np.array([[0.5, 0.5, 10.5],
                                 [10.0, 0.5, 4.5],
                                 [-10.0, 0.5, 4.5],
                                 # [0.5, 0.5, -10.5],
                                 [10, 10, 10]])

    camera = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0, znear=0.05)
    scene = pyrender.Scene()
    smpl_model = "../3d_data/smpl_NOC_vertices.ply"
    tri_mesh = trimesh.load_mesh(smpl_model, file_type="ply")
    smpl_vertices = np.array(tri_mesh.vertices)
    smpl_vertices = scale * (smpl_vertices - 0.5) + 0.5
    smpl_py_mesh = pyrender.Mesh.from_points(smpl_vertices, tri_mesh.colors.copy()[0])
    smpl_node = pyrender.Node(mesh=smpl_py_mesh)

    point_size = 5.0
    delta = 0.015
    colors = [0, 0, 0]
    for idx in range(0, file_size):
        file_list = ["Ground_truth_{}.ply".format(idx), "Output_{}.ply".format(idx)]
        camera_list = []
        keep_mesh = 0
        keep_mesh_2 = 0
        old_vertices = 0
        line_segments = []
        line_segments_2 = []
        for jdx, file in enumerate(file_list):
            render_list = []

            path = os.path.join(opt.input, file)

            tri_mesh = trimesh.load_mesh(path, file_type='ply')
            colors = np.hstack(
                (tri_mesh.colors.copy()[0], np.ones((tri_mesh.colors.copy()[0].shape[0], 1)) * 255)) / 255
            colors = colors[:, [2, 1, 0, 3]]
            mesh_vertices = scale * (np.array(tri_mesh.vertices) - 0.5) + 0.5
            tri_mesh = trimesh.points.PointCloud(vertices=mesh_vertices, colors=colors)
            pts = tri_mesh.vertices.copy()


            if jdx == 0:
                old_vertices = np.array(pts)
                line_segments.append(np.array(pts) + np.array([0, delta, 0]))
                line_segments_2.append(np.array(pts) - np.array([0, delta, 0]))
                line_segments_2.append(np.array(pts) + np.array([0, delta, 0]))

            elif jdx == 1:
                color_diff = np.linalg.norm(old_vertices - np.array(pts), axis=1)
                line_segments.append(np.array(pts) - np.array([0, delta, 0]))
                line_segments.append(np.array(pts) + np.array([0, delta, 0]))
                line_segments_2.append(np.array(pts) - np.array([0, delta, 0]))
                # color_diff = np.abs(old_vertices - np.array(pts))
                # error_r = np.fabs(np.subtract(old_vertices[:, 0], pts[:, 0]))
                # error_g = np.fabs(np.subtract(old_vertices[:, 1], pts[:, 1]))
                # error_b = np.fabs(np.subtract(old_vertices[:, 2], pts[:, 2]))
                heat_map = None
                heat_map = cv2.normalize(color_diff, heat_map, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                         dtype=cv2.CV_8U)
                # heatmap[heatmap > 50] = 255

                colors = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET).reshape((old_vertices.shape[0], 3))

            for cam_pose in camera_locations:
                camera_pose = look_at(cam_pos_world=cam_pose, to_pos_world=np.array([0.5, 0.5, 0.5]))
                camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
                scene.add_node(node=camera_node)
                # if jdx == 0:
                render = pyrender.OffscreenRenderer(340, 340, point_size=1.0)
                scene.add_node(node=smpl_node)
                rendered_smpl, _ = render.render(scene=scene, flags=pyrender.RenderFlags.NONE)
                scene.remove_node(node=smpl_node)
                py_mesh = pyrender.Mesh.from_points(pts, colors=colors)
                mesh_node = pyrender.Node(mesh=py_mesh)
                if jdx == 1:
                    keep_mesh_2 = mesh_node
                else:
                    keep_mesh = mesh_node
                scene.add_node(node=mesh_node)
                # else:
                #     for kdx, pose in enumerate(pts):
                #         spr_tri = trimesh.creation.uv_sphere(radius=0.01)
                #         spr_tri.visual.vertex_colors = (colors[kdx, :] )
                #         # spr_tri.visual.vertex_colors[:, 3] = 255
                #         # spr_tri.visual.vertex_colors = np.array([255, 0, 0])
                #         # print(colors[kdx] / 255)
                #         poses = np.eye(4)
                #         poses[:3, 3] = pose
                #         spr_py = pyrender.Mesh.from_trimesh(spr_tri)
                #         scene.add(spr_py, pose=poses)
                render = pyrender.OffscreenRenderer(340, 340, point_size=point_size)
                rendered, _ = render.render(scene=scene, flags=pyrender.RenderFlags.NONE)
                rendered = rendered.copy()
                scene.remove_node(mesh_node)
                scene.remove_node(camera_node)
                rendered[rendered[:, :] == np.array([255, 255, 255])] = rendered_smpl[rendered[:, :] == np.array([255, 255, 255])]

                rendered = np.pad(array=rendered, pad_width=((5, 5), (5, 5), (0, 0)), mode='constant')
                render_list.append(rendered)

            # scene.clear()

            # line

            render_list = np.hstack(render_list)

            camera_list.append(render_list)

        line_render = []

        line_segments = np.concatenate(line_segments, axis=0)
        line_segments_2 = np.concatenate(line_segments_2, axis=0)

        faces = np.array([[val, val + old_vertices.shape[0], val + old_vertices.shape[0] * 2] for val in
                          range(old_vertices.shape[0])])

        # colors = np.array([colors[id].tolist() if id < old_vertices.shape[0] else [0, 0, 0] for id in
        #                    range(old_vertices.shape[0] * 3)])
        # colors = np.repeat(colors, 3, axis=0)
        colors = np.zeros((old_vertices.shape[0] * 3, 3))
        tri_mesh = trimesh.Trimesh(vertices=line_segments, faces=faces, vertex_colors=colors)
        tri_mesh_2 = trimesh.Trimesh(vertices=line_segments_2, faces=faces, vertex_colors=colors)
        py_mesh = pyrender.Mesh.from_trimesh(tri_mesh)
        py_mesh_2 = pyrender.Mesh.from_trimesh(tri_mesh_2)
        mesh_node = pyrender.Node(mesh=py_mesh)
        mesh_node_2 = pyrender.Node(mesh=py_mesh_2)
        # scene.add_node(mesh_node)
        # tri_scene = tri_mesh.scene()
        for cam_pose in camera_locations:
            # camera_old, _geometry = tri_scene.graph[tri_scene.camera.name]
            # camera_new = np.dot(camera_old,
            #                     look_at(cam_pos_world=cam_pose, to_pos_world=np.array([0.5, 0.5, 0.5])))
            # tri_scene.graph[tri_scene.camera.name] = camera_new
            # rendered = tri_scene.save_image(resolution=[340, 340])
            # rendered = Image.open(BytesIO(base64.b64decode(rendered)))

            camera_pose = look_at(cam_pos_world=cam_pose, to_pos_world=np.array([0.5, 0.5, 0.5]))
            camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
            scene.add_node(node=camera_node)
            # if jdx == 0:
            render = pyrender.OffscreenRenderer(340, 340, point_size=1.0)
            scene.add_node(node=smpl_node)
            rendered_smpl, _ = render.render(scene=scene, flags=pyrender.RenderFlags.NONE)
            scene.remove_node(node=smpl_node)

            scene.add_node(node=mesh_node)
            scene.add_node(node=mesh_node_2)
            scene.add_node(node=keep_mesh)
            scene.add_node(node=keep_mesh_2)

            render = pyrender.OffscreenRenderer(340, 340, point_size=point_size)
            rendered, _ = render.render(scene=scene,
                                        flags=pyrender.RenderFlags.NONE | pyrender.RenderFlags.SKIP_CULL_FACES)
            rendered = rendered.copy()
            scene.remove_node(mesh_node)
            scene.remove_node(mesh_node_2)
            scene.remove_node(keep_mesh)
            scene.remove_node(keep_mesh_2)
            scene.remove_node(camera_node)

            rendered[rendered[:, :] == np.array([255, 255, 255])] = rendered_smpl[
                rendered[:, :] == np.array([255, 255, 255])]

            rendered = np.pad(array=rendered, pad_width=((5, 5), (5, 5), (0, 0)), mode='constant')
            line_render.append(rendered)

        render_list = np.hstack(line_render)

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
    opt = parse_args(['--input=../3d_data/sparse_sampled_test_Res18UNet_3Decoder_Dropout_2Head_Coeff_100',
                      '--output=../3d_data/outputs_sparse_sampled_test_Res18UNet_3Decoder_Dropout_2Head_Coeff_100'])
    main(opt)
