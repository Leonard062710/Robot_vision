import cv2
import numpy as np

K = np.loadtxt("./K.txt")
cdn = np.loadtxt("./p_W_corners.txt")
pW_corners = np.float32(cdn/100)

img1 =  cv2.imread("./img_0001.jpg")
detect_corn = np.loadtxt("./detected_corners.txt")

j = detect_corn[0]
K_inv = np.linalg.inv(K)
num_points = len(pW_corners)
k = np.zeros((12,2))
Q = np.zeros((2 * num_points, 12))
for i in range(0,12):
    X, Y, Z = pW_corners[i]
    u_n = j[2 * i]
    v_n = j[2 * i + 1]
    k[i] = np.column_stack((u_n,v_n))
    vect = np.array([[u_n],[v_n],[1]])
    vect_c = K_inv @ vect
    x_n = vect_c[0,0]/vect_c[2,0]
    y_n = vect_c[1,0]/vect_c[2,0]
    Q[2 * i] = np.array([X, Y, Z, 1, 0, 0, 0, 0, -X * x_n, -Y * x_n, -Z * x_n, -x_n])
    Q[2 * i + 1] = np.array([0, 0, 0, 0, X, Y, Z, 1, -X * y_n, -Y * y_n, -Z * y_n, -y_n])

U,S,VT = np.linalg.svd(Q)
V = VT.T
M = V[:,-1].reshape(3,4) * -1

# Hiệu chuẩn M
R = M[:3,:3]
U_R, _, VT_R = np.linalg.svd(R)
R_calib = np.dot(U_R, VT_R)
R_numerator = np.linalg.norm(R_calib)
R_denominator = np.linalg.norm(R)
alpha = R_numerator / R_denominator

t_calib = alpha * M[:,3]
M_calib = np.zeros((3,4))
M_calib[:3,:3] = R_calib
M_calib[:3,3] = t_calib
#Khởi tạo ma trận T
T = np.zeros((4, 4))
T[:3, :3] = M_calib[:3, :3]
T[:3, 3] = M_calib[:3, 3]
T[3, 3] = 1
T = np.linalg.inv(T)

World = np.zeros((4,12))
World[:3,] = pW_corners.T
World[-1,:] = 1
points = K @ M_calib @ World
points = points.T
pix_u = points[:,0]/points[:,2]
pix_v = points[:,1]/points[:,2]
p = np.column_stack((pix_u.astype(int), pix_v.astype(int)))

#Vẽ và so sánh
for i2 in range(k.shape[0]):
    cv2.circle(img1, k[i2,:].astype(int), 6, (200, 200, 0), -1)
for l in range(p.shape[0]):
    x, y = p[l, :].astype(int)
    cv2.rectangle(img1, (x - 1, y - 5), (x + 1, y + 5),
                      (0, 100, 240), -1)
    cv2.rectangle(img1, (x - 5, y - 1), (x + 5, y + 1),
                      (0, 100, 240), -1)
# print(k,'\n')
# print(p)
# cv2.imshow("Image", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
