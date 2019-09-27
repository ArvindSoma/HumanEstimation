import os
import sys

import numpy as np

import trimesh

sys.path.insert(0, '../external/pyrender')
import pyrender


class Render:
    def __init__(self, width, height, camera_distance=0.05, pose_y=0.0, focal_length=None):

        self.camera = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0, znear=0.05)

        if focal_length:
            focal_length = focal_length * height
            self.camera = pyrender.IntrinsicsCamera(focal_length, focal_length, width / 2, height / 2,
                                                    0.05, 5.05)
        self.width = width
        self.height = height
        self.global_tr = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, pose_y],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        self.camera_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, camera_distance],
            [0.0, 0.0, 0.0, 1.0]
        ])

        self.tri_mesh = self.py_mesh = self.vertices = self.faces = self.render = self.py_scene = None

    def set_render(self, vertices, faces, visual, normalize=True):

        self.tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual)
        self.render = pyrender.OffscreenRenderer(self.width, self.height)
        self.py_mesh = pyrender.Mesh.from_trimesh(self.tri_mesh)
        self.py_scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5], bg_color=[-1.0, -1.0, -1.0])
        self.py_scene.add(self.py_mesh, pose=self.global_tr)
        self.py_scene.add(self.camera, pose=self.camera_pose)

        self.vertices = vertices
        self.faces = faces
        if normalize:
            self.vertices -= self.tri_mesh.centroid
            bounds = self.tri_mesh.bounding_box_oriented.extents

            self.vertices /= bounds
            self.vertices += 1/2

    def render_visual(self, flags, face_id=None):

        rendered_color_visual, _ = self.render.render(scene=self.py_scene, flags=flags)
        if face_id is not None:
            mask = rendered_color_visual[:, :, 2] != -1.
            temp_2 = rendered_color_visual[:, :, 2]
            temp_2[mask] = np.take(face_id, temp_2[mask].astype('int'))
            rendered_color_visual[:, :, 2] = temp_2
        return rendered_color_visual

    def render_interpolate(self, vertices):

        rendered_color_visual = self.render_visual(flags=pyrender.RenderFlags.SKIP_CULL_FACES)

        rendered_interp = self.render_visual(flags=pyrender.RenderFlags.BARYCENTRIC_COORDINATES + pyrender.RenderFlags.SKIP_CULL_FACES)

        tri_id = self.render_visual(flags=pyrender.RenderFlags.TRIANGLE_ID_RENDERING + pyrender.RenderFlags.SKIP_CULL_FACES)

        # rendered_interp, _ = self.render.render(scene=self.py_scene,
        #                                         flags=pyrender.RenderFlags.BARYCENTRIC_COORDINATES | pyrender.RenderFlags.SKIP_CULL_FACES)
        # tri_id, _ = self.render.render(scene=self.py_scene,
        #                                flags=pyrender.RenderFlags.TRIANGLE_ID_RENDERING | pyrender.RenderFlags.SKIP_CULL_FACES)
        vertex_stream = np.take(vertices, self.faces, axis=0)

        tri_id = tri_id[:, :, 0]

        rendered_interp = rendered_interp.reshape(rendered_interp.shape + (1,)).repeat([3], axis=-1)
        out_view = vertex_stream[tri_id.astype('int')] * rendered_interp
        out_view = out_view.sum(axis=-2)

        # out_view[rendered_color_visual < 0] = 0

        return out_view

