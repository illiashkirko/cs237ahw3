#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt

# For reproducible random results.
np.random.seed(789)

# The following utility functions are provided for your convenience.
# You may use them in your solution, or you may write your own.
# Do not alter them, and please read the docstrings carefully.
K = np.array([[471.14628085,  -4.94771211, 295.13159043], [0., 436.71106326, 240.96582594], [0., 0., 1.]])
SQUARE_SIZE = 0.0205


# ## Problem 1.1

# In[15]:


# Read the image from file
img_path = "p1_data/checkerboard.png"
image = cv2.imread(img_path, 0)

# Plot Corners on top of image
fig = plt.figure()
ax = plt.subplot()
ax.imshow(image, cmap="gray")
ax.set_ylabel("v")
ax.set_xlabel("u")

# 1) Find the corner locations of the chessboard in image space.
#    Hint: Use the function cv2.findChessboardCorners(image, (num_corners_x, num_corners_y))
#    Hint: findChessboardCorners outputs pixel coordinates!
# 2) Plot their location in image space over top of the image.
#    Hint: use ax.plot(...) -- should be one line!
#    Hint: This should only be 1 line of code!

ncorners_y = 7
ncorners_x = 9
# --------- YOUR CODE STARTS HERE ---------------
result, corners = cv2.findChessboardCorners(image, (ncorners_x, ncorners_y))
ax.plot(corners[:, 0, 0], corners[:, 0, 1], 'o')
# --------- YOUR CODE ENDS HERE -----------------


# ## Problem 1.2

# In[137]:


# 1)  Compute the homography, H.
#     1.a) Use the SQUARE_SIZE, and the dimensions of the grid above to
#             to create a the array P = [[X_0, Y_0, 1], .... [X_n, Y_n, 1]].
#             P should have shape (63, 3).
#          Hint: Use np.meshgrid to generate the X, Y coordinates, and
#             array.reshape and np.hstack to form the final matrix.
#     1.b) Form the M matrix outlined in the write-up!
#     1.c) Use np.linalg.svd(M) to solve for the nullspace of M and recover H.
#          Hint: np.linalg.svd(M) returns V.T.
#          Hint: Use <array>.reshape(3, 3) to give H the proper dimensions.
# 2)  Compute the camera pose rotation, R, and translation, t.
#     2.a) Compute KinvH = inv(K) @ H.
#          Hint: Use np.linalg.inv().
#     2.b) Normalize the values of KinvH to it's first column.
#          Hint: Use np.linalg.norm() and the / operator.
#     2.c) Compute r0, r1, r2, and t as outlined in the write-up.
#          Hint: Use np.cross().
#     2.d) Form R from r0, r1, and r2.
#          Hint: Use np.column_stack()

# --------- YOUR CODE STARTS HERE ---------------
# 1.a
nx, ny = (9, 7)
x = np.linspace(0, (nx-1) * SQUARE_SIZE, nx)
y = np.linspace(0, (ny-1) * SQUARE_SIZE, ny)

xv, yv = np.meshgrid(x, y)

xv = np.reshape(xv, (nx * ny, 1))
yv = np.reshape(yv, (nx * ny, 1))
zv = np.ones((63, 1))
P = np.hstack((xv, yv, zv))

# 1.b
M = np.empty((0, 9))
for i in range(0, 63):
    X, Y = (P[i, 0], P[i, 1])
    u, v = (corners[i, 0, 0], corners[i, 0, 1])
    Mi = np.array([[-X, -Y, -1, 0, 0, 0, u * X, u * Y, u], [0, 0, 0, -X, -Y, -1, v * X, v * Y, v]])
    M = np.append(M, Mi, axis=0)

# 1.c
U, S, Vt = np.linalg.svd(M)
H = Vt[-1].reshape(3, 3)

# 2.a
KinvH = np.linalg.inv(K) @ H
# 2.b
l = np.linalg.norm(KinvH[:, 0])
KinvH_normalized = KinvH / l
# 2.c
r0 = KinvH_normalized[:,0]
r1 = KinvH_normalized[:,1]
r2 = np.cross(r0, r1)
# 2.d
R = np.column_stack((r0, r1, r2))
t = KinvH_normalized[:,2]

print(np.round(R, 7))
print(np.round(t, 4))
# --------- YOUR CODE ENDS HERE -----------------


# ## Problem 1.3

# In[138]:


# 1) Finish the transform_world_to_camera function which uses
#       the given K, and the R and t that you calculated!.
#    Hint: You'll need to use some equations from the write-up.

def transform_world_to_camera(K, R, t, world_coords):
    """
    Args:
        K: np.array with shape (3, 3), camera intrinsics matrix.
        R: np.array with shape (3, 3), camera rotation.
        t: np.array with shape (3, ) or (3, 1), camera translation.
        world_coords: np.array with shape (N, 3), cartesian coordinates (X, Y, Z)
            in world frame to transform into camera pixel space.
    Return:
        uv: np.array with shape (N, 2), with (u, v) coordinates of that are
            the projections of the the world_coords on the image plane.
    """
    # --------- YOUR CODE STARTS HERE ---------------
    N, _ = world_coords.shape
    ones = np.ones((N, 1))
    homogeneous_world = np.hstack((world_coords, ones))
    R_t = np.hstack((R, t.reshape(3, 1)))
    uv = K @ R_t @ homogeneous_world.T
    uv[:2, :] /= uv[2, :]
    uv = uv[:2]
    uv = uv.T
    # --------- YOUR CODE ENDS HERE -----------------
    return uv

ax = plt.subplot()
ax.imshow(image, cmap="gray")
ax.set_xlabel("u")
ax.set_ylabel("v")

# 2) Project the global coordinates of the corners that you calculated
#       earlier back onto the image using transform_world_to_camera.
#    Hint: The input world_coords should have shape (63, 3)
#    Hint: Use the Xs and Ys you got earlier and set Zs = 0.
#    Hint: You may need to change some of the variable names at the bottom
#       of this cell to match the names that you used above!

# --------- YOUR CODE STARTS HERE ---------------
zeros = np.zeros((63, 1))
world_coords = np.hstack((xv, yv, zeros))
uv = transform_world_to_camera(K, R, t, world_coords)

ax.plot(uv[:, 0], uv[:, 1], 'o')
# --------- YOUR CODE ENDS HERE -----------------


# ## Project Anything!
# Try using your the camera pose you computed to project our secret point cloud (defined in world coordinates) onto the camera frame!

# In[141]:


ax = plt.subplot()
ax.imshow(image, cmap="gray")
ax.set_xlabel("u")
ax.set_ylabel("v")

# Load the secrect point cloud from file!
point_cloud_secret = np.load("p1_data/secret.npy")

# NOTE: Uncomment the lines below when you've finished everything else!
uv_secret = transform_world_to_camera(K, R, t, point_cloud_secret)
ax.scatter(uv_secret[:, 0], uv_secret[:, 1], s=0.5, c="b")


# In[ ]:




