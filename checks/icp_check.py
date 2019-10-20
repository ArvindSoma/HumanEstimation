import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import acos, atan2, cos, pi, sin
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as R



def R_axis_angle(matrix, axis, angle):
    """Generate the rotation matrix from the axis-angle notation.
    Conversion equations
    ====================
    From Wikipedia (http://en.wikipedia.org/wiki/Rotation_matrix), the conversion is given by::
        c = cos(angle); s = sin(angle); C = 1-c
        xs = x*s;   ys = y*s;   zs = z*s
        xC = x*C;   yC = y*C;   zC = z*C
        xyC = x*yC; yzC = y*zC; zxC = z*xC
        [ x*xC+c   xyC-zs   zxC+ys ]
        [ xyC+zs   y*yC+c   yzC-xs ]
        [ zxC-ys   yzC+xs   z*zC+c ]
    @param matrix:  The 3x3 rotation matrix to update.
    @type matrix:   3x3 numpy array
    @param axis:    The 3D rotation axis.
    @type axis:     numpy array, len 3
    @param angle:   The rotation angle.
    @type angle:    float
    """

    # Trig factors.
    ca = cos(angle)
    sa = sin(angle)
    C = 1 - ca

    # Depack the axis.
    x, y, z = axis

    # Multiplications (to remove duplicate calculations).
    xs = x*sa
    ys = y*sa
    zs = z*sa
    xC = x*C
    yC = y*C
    zC = z*C
    xyC = x*yC
    yzC = y*zC
    zxC = z*xC

    # Update the rotation matrix.
    matrix[0, 0] = x*xC + ca
    matrix[0, 1] = xyC - zs
    matrix[0, 2] = zxC + ys
    matrix[1, 0] = xyC + zs
    matrix[1, 1] = y*yC + ca
    matrix[1, 2] = yzC - xs
    matrix[2, 0] = zxC - ys
    matrix[2, 1] = yzC + xs
    matrix[2, 2] = z*zC + ca



def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=20, tolerance=1e-3):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]
    distances = 100
    iters = 200

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        print("Tolerance Check", np.abs(prev_error - mean_error))
        if np.sum(mean_error) < tolerance:
            break
        prev_error = mean_error
        iters = i

    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)

    return T, distances, iters


def main():
    point = np.array([[0.0, 0.0, 0.0],
                      [0.0, 0.4, 0.0],
                      [0.4, 0.0, 0.0],
                      [0.4, 0.4, 0.0],
                      [0.0, 0.0, 0.4],
                      [0.0, 0.4, 0.4],
                      [0.4, 0.0, 0.4],
                      [0.4, 0.4, 0.4]
                      ])

    transform = np.array([[1, 0, 0, 4],
                          [0, 1, 0, 4],
                          [0, 0, 1, 2],
                          [0, 0, 0, 1]])

    transform[:3, :3] = R.from_quat([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)]).as_dcm()

    initialization = np.array([[1, 0, 0, 0.1],
                               [0, 1, 0, 0.5],
                               [0, 0, 1, 0.03],
                               [0, 0, 0, 1]])

    colors = np.random.randint(low=0, high=255, size=(8, 3))
    fig = plt.figure()
    ax = fig.add_subplot(311, projection='3d')
    ax.scatter(point[:, 0], point[:, 1], point[:, 2], c=colors/255)
    ax = fig.add_subplot(312, projection='3d')
    transformed_point = np.dot(transform[:3, :3], point.T).T + transform[:3, 3:4].T
    ax.scatter(transformed_point[:, 0], transformed_point[:, 1], transformed_point[:, 2], c=colors/255)

    transformation, distances, iters = icp(A=point, B=transformed_point,
                                           init_pose=initialization)

    print("Predicted transformation:\n", transformation)
    print("Predicted distances:\n", distances)
    print("Iterations to converge:\n", iters)

    predicted_transform = np.dot(transformation[:3, :3], point.T).T + transformation[:3, 3:4].T
    ax = fig.add_subplot(313, projection='3d')
    ax.scatter(predicted_transform[:, 0], predicted_transform[:, 1], predicted_transform[:, 2], c=colors / 255)

    plt.show()



    # ang = np.linspace(-np.pi / 2, np.pi / 2, 320)
    # a = np.array([ang, np.sin(ang)])
    #
    # th = np.pi / 2
    # rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    # b = np.dot(rot, a) + np.array([[0.2], [0.3]])
    # b = np.dot(rot, a) + np.array([[0.2], [0.3]])
    #
    # # Run the icp
    # M2 = icp(a, b, [0.01, 0.01, 2], 30)
    #
    # # Plot the result
    # src = np.array([a.T]).astype(np.float32)
    # res = cv2.transform(src, M2)
    # plt.figure()
    # plt.plot(b[0], b[1], 'b.')
    # plt.plot(res[0].T[0], res[0].T[1], 'r.')
    # plt.plot(a[0], a[1], 'g.')
    # plt.show()


if __name__ == "__main__":
    main()