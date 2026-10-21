import numpy as np
import cv2


K = np.loadtxt("./K.txt")
poses = np.loadtxt("./poses.txt")
img1 =  cv2.imread("./img_0001.jpg",cv2.IMREAD_GRAYSCALE)
img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
# s = float(input("Tỉ lệ scale [0,3) : "))
# img1 = cv2.resize(img1, (0, 0), fx=s, fy=s)

img2 =  cv2.imread("./img_0001_dist.jpg")

size_x = np.linspace(-1,9,11)
size_y = np.linspace(-1,6,8)
points_x, points_y = np.meshgrid(size_x, size_y)


norm = np.linalg.norm(poses[:3])
rot_vec =poses[:3]/norm

kx, ky, kz = rot_vec

k_matrix = np.array([
    [0, -kz, ky],
    [kz, 0, -kx],
    [-ky, kx, 0]
])

R = np.eye(3) + np.sin(norm) * k_matrix + (1 - np.cos(norm)) * k_matrix @ k_matrix
Rot_trans = np.column_stack((R, poses[3:]))

# W = np.zeros((4, 8))
#
# W[0, :] = [0.16, 0.28, 0.16, 0.28, 0.16, 0.28, 0.16, 0.28]
# W[1, :] = [0.08, 0.08, 0.20, 0.20, 0.08, 0.08, 0.20, 0.20]
# W[2, :] = [0, 0, 0, 0, -0.12, -0.12, -0.12, -0.12]
# # W[0, :] = points_x
#
# W[3, :] = 1.0
W = np.zeros((4, 88))

W[0, :] = points_x.flatten() * 0.04
W[1, :] = points_y.flatten() * 0.04
W[2, :] = 0
W[3, :] = 1.0

vec = K @  Rot_trans @ W
u = vec[0, :] / vec[2, :]
v = vec[1, :] / vec[2, :]

k1 = -1.6774e-06
k2 = 2.5847e-12
r_exp2 = (u - K[0,2])**2 + (v - K[1,2])**2

# points = (np.column_stack((u, v))*s).astype(int)
points = np.column_stack((u, v)).astype(int)
# for i in range(points.shape[0]):
#     u_distorted = (u[i]-K[0,2]) * (1 + k1*r_exp2[i] + k2*r_exp2[i]**2) + K[0,2]
#     v_distorted = (v[i]-K[1,2]) * (1 + k1*r_exp2[i] + k2*r_exp2[i]**2) + K[1,2]
#     points[i, 0] = u_distorted
#     points[i, 1] = v_distorted

u_distorted = (u-K[0,2]) * (1 + k1*r_exp2 + k2*r_exp2**2) + K[0,2]
v_distorted = (v-K[1,2]) * (1 + k1*r_exp2 + k2*r_exp2**2) + K[1,2]
points[ : , 0] = u_distorted
points[ : , 1] = v_distorted
print(points[10])
# edges = [
#     (0,1),(0,2),(1,3),(2,3),
#     (0,4),(1,5),(2,6),(3,7),
#     (4,6),(6,7),(4,5),(5,7)
#     # (4,3),(5,2),(6,1),(7,0)
# ]
#
# for i, j in edges:
#     cv2.line(img1, points[i], points[j], (0, 0, 255), 2)
#     cv2.circle(img1, points[i], 5, (0, 0, 255), -1)
#     cv2.circle(img1, points[j], 5, (0, 0, 255), -1)



for i in range(points.shape[0]):
    cv2.circle(img2, points[i, :], 5, (0, 0, 255), -1)

cv2.imshow("img", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()