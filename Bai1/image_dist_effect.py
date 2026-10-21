import cv2
import numpy as np

K = np.loadtxt("./K.txt")
img2 = cv2.imread("./img_0001_dist.jpg")
h, w = img2.shape[:2]
u, v = np.meshgrid(np.arange(w), np.arange(h))
r_exp2 = (u - K[0, 2]) ** 2 + (v - K[1, 2]) ** 2

k1 = -1.6774e-06
k2 = 2.5847e-12

d = 1 + k1 * r_exp2 + k2 * (r_exp2 ** 2)
u_u = ((u - K[0, 2]) * d + K[0, 2])
v_u = ((v - K[1, 2]) * d + K[1, 2])
u_u = np.round(u_u).astype(int)
v_u = np.round(v_u).astype(int)

img2 = img2[v_u, u_u]

# img_undistorted = cv2.remap(
#     img2, map_x, map_y, interpolation=cv2.INTER_LINEAR
# )
#
# cv2.imshow("Undistorted Image", img_undistorted)
# cv2.imshow("Distort Image", img2)
cv2.imshow("Anh sau khi gan truc tiep", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()