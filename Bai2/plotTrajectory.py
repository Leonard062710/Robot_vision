from detect_images import detect_corn, pW_corners, K
import cv2
import matplotlib.pyplot as plt
import numpy as np
from represent_of_the_camera_position import drawCamera

K_inv = np.linalg.inv(K)
num_points = len(pW_corners)
M_list = []
R_list = []
t_list = []
for j in detect_corn:
    Q = np.zeros((2 * num_points, 12))
    for i in range(0, 12):
        X, Y, Z = pW_corners[i]
        u_n = j[2 * i]
        v_n = j[2 * i + 1]
        vect = np.array([[u_n], [v_n], [1]])
        vect_c = K_inv @ vect
        x_n = vect_c[0, 0] / vect_c[2, 0]
        y_n = vect_c[1, 0] / vect_c[2, 0]
        Q[2 * i] = np.array([X, Y, Z, 1, 0, 0, 0, 0, -X * x_n, -Y * x_n, -Z * x_n, -x_n])
        Q[2 * i + 1] = np.array([0, 0, 0, 0, X, Y, Z, 1, -X * y_n, -Y * y_n, -Z * y_n, -y_n])

    U, S, VT = np.linalg.svd(Q)
    V = VT.T
    M = V[:, -1].reshape(3, 4)

    # Hiệu chuẩn M
    R = M[:3, :3]
    if np.linalg.det(R) < 0:
        M = -M
        R = M[:3, :3]
    U_R, _, VT_R = np.linalg.svd(R)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(U_R @ VT_R)
    R_calib = U_R @ correction @ VT_R
    R_numerator = np.linalg.norm(R_calib)
    R_denominator = np.linalg.norm(R)
    alpha = R_numerator / R_denominator

    t_calib = alpha * M[:, 3]
    M_calib = np.zeros((3, 4))
    M_calib[:3, :3] = R_calib
    M_calib[:3, 3] = t_calib
    M_list.append(M_calib)
points = np.stack(M_list)
for k in range(len(points)):
    R = points[k, :, :3]
    R_list.append(R)
    t = points[k, :, 3]
    t_list.append(t)
R_list = np.stack(R_list)
t_list = np.stack(t_list)

def rotation_to_quaternion(R):
    r00, r01, r02 = R[0]
    r10, r11, r12 = R[1]
    r20, r21, r22 = R[2]

    trace = r00 + r11 + r22

    if trace > 0:
        S = 2 * np.sqrt(trace + 1)
        w = 0.25 * S
        x = (r21 - r12) / S
        y = (r02 - r20) / S
        z = (r10 - r01) / S
    elif r00 > r11 and r00 > r22:
        S = 2 * np.sqrt(1 + r00 - r11 - r22)
        w = (r21 - r12) / S
        x = 0.25 * S
        y = (r01 + r10) / S
        z = (r02 + r20) / S
    elif r11 > r22:
        S = 2 * np.sqrt(1 + r11 - r00 - r22)
        w = (r02 - r20) / S
        x = (r01 + r10) / S
        y = 0.25 * S
        z = (r12 + r21) / S
    else:
        S = 2 * np.sqrt(1 + r22 - r00 - r11)
        w = (r10 - r01) / S
        x = (r02 + r20) / S
        y = (r12 + r21) / S
        z = 0.25 * S
    return np.array([x, y, z, w])


quaternion_list = np.array([rotation_to_quaternion(R) for R in R_list])

def quaternion_to_rotation(q):
    x, y, z, w = np.asarray(q, dtype=float)
    q = np.array([x, y, z, w])
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Quaternion must be non-zero")
    x, y, z, w = q / norm
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def plotTrajectory3D(fps, filename, translations, quaternions, pW_corners):
    translations = np.asarray(translations, dtype=float)
    quaternions = np.asarray(quaternions, dtype=float)
    pW_corners = np.asarray(pW_corners, dtype=float)

    all_points = np.vstack((translations, pW_corners))
    minimum = all_points.min(axis=0)
    maximum = all_points.max(axis=0)
    spans = maximum - minimum
    largest_span = max(float(spans.max()), 1.0)
    margin = largest_span * 0.1
    arrow_length = largest_span * 0.2

    figure = plt.figure(figsize=(8, 8), dpi=100)
    axis = figure.add_subplot(111, projection="3d")
    figure.tight_layout()
    figure.canvas.draw()
    width, height = figure.canvas.get_width_height()
    writer = cv2.VideoWriter(
        str(filename), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )

    plt.show(block=False)
    try:
        for i, (position, quaternion) in enumerate(zip(translations, quaternions)):
            axis.clear()
            axis.scatter(*pW_corners.T, color="black", marker="o")
            planes = (
                ([0, 1, 3, 2, 0], "tab:blue"),
                ([4, 5, 7, 6, 4], "tab:orange"),
                ([8, 10, 11, 9, 8], "tab:green"),
            )
            for corner_indices, color in planes:
                plane = pW_corners[corner_indices]
                axis.plot(*plane.T, color=color, linewidth=2)
            axis.plot(*translations[:i + 1].T, color="orange")
            axis.scatter(*position, color="blue")
            rotation = quaternion_to_rotation(quaternion)
            drawCamera(axis, position, rotation, arrow_length, 0.15)

            axis.set_xlim(maximum[0] + margin, minimum[0] - margin)
            axis.set_ylim(minimum[1] - margin, maximum[1] + margin)
            axis.set_zlim(maximum[2] + margin, minimum[2] - margin)
            axis.set_box_aspect(np.maximum(spans, largest_span * 0.05))
            axis.set_xlabel("X")
            axis.set_ylabel("Y")
            axis.set_zlabel("Z")

            figure.canvas.draw()
            rgb = np.asarray(figure.canvas.buffer_rgba())[:, :, :3]
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            plt.pause(1.0 / fps)
    finally:
        writer.release()
    plt.show()
    plt.close(figure)


camera_translations = np.array([-R.T @ t for R, t in zip(R_list, t_list)])
camera_quaternions = np.array([rotation_to_quaternion(R.T) for R in R_list])
plotTrajectory3D(30, "Trajectory.mp4", camera_translations, camera_quaternions, pW_corners)
