from detect_images import T, img1, pW_corners, j
import numpy as np
import cv2
import matplotlib.pyplot as plt


def drawCamera(ax,pos,dir,length_scl = 1,head_size = 0.2) :
  dir_norm = np.array(dir) / np.linalg.norm(dir)
  arrow_dir = dir_norm * length_scl
  ax.scatter(
      pos[0],
      pos[1],
      pos[2],
      color="blue",
      s=100,
      label="Camera Position",
  )
  ax.quiver(
      pos[0],
      pos[1],
      pos[2],
      arrow_dir[0],
      arrow_dir[1],
      arrow_dir[2],
      color="red",
      linewidth=2,
      arrow_length_ratio=head_size,
  )

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(projection="3d")

pos, _ = cv2.Rodrigues(T[:3, :3])

dir = T[:3, 3]

drawCamera(ax, pos, dir, 2.0, 0.3)

# Cấu hình hiển thị thêm cho đồ thị sinh động
ax.set_xlabel("Trục X")
ax.set_ylabel("Trục Y")
ax.set_zlabel("Trục Z")
ax.set_title("Mô phỏng vị trí và hướng nhìn của Camera trong không gian 3D")
ax.legend()

# Thiết lập giới hạn khung nhìn đồ thị
ax.set_xlim([0, 5])
ax.set_ylim([0, 5])
ax.set_zlim([0, 5])


plt.show()