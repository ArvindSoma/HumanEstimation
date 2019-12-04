import numpy as np
import trimesh
import pyrender


def main():
    points = np.random.random((100, 3, 2))
    colors = np.random.random((100, 3, 2))

    py_mesh = pyrender.Mesh.from_points(points=points, colors=colors)

    scene = pyrender.Scene()

    scene.add(py_mesh, pose=np.eye(4))

    pyrender.Viewer(scene=scene)

    return True


if __name__ == "__main__":
    main()
