import cv2
import numpy as np


K = np.loadtxt("./K.txt")
img2 = cv2.imread("./img_0001_dist.jpg")
# Giả sử bạn đã có ảnh img2, ma trận K, k1, k2 và lưới tọa độ u, v của ảnh đích
h, w = img2.shape[:2]
k1 = -1.6774e-06
k2 = 2.5847e-12

# Tạo ảnh đầu ra trống (ảnh màu có 3 kênh)
img_new = np.zeros_like(img2)

# Lưới tọa độ đích u, v
u, v = np.meshgrid(np.arange(w), np.arange(h))

# Tính tọa độ méo (hoặc tính ngược lại tùy thuộc vào chiều bạn muốn làm)
r_exp2 = (u - K[0, 2]) ** 2 + (v - K[1, 2]) ** 2
distortion_factor = 1 + k1 * r_exp2 + k2 * (r_exp2 ** 2)
u_d = (u - K[0, 2]) * distortion_factor + K[0, 2]
v_d = (v - K[1, 2]) * distortion_factor + K[1, 2]

# Duyệt qua từng pixel (Đối với ảnh lớn, cách này chạy bằng vòng lặp Python sẽ hơi chậm)
# Để tối ưu vector hóa NumPy hoàn toàn mà không dùng vòng lặp cho phần nội suy:

x1 = np.floor(u_d).astype(int)
x2 = x1 + 1
y1 = np.floor(v_d).astype(int)
y2 = y1 + 1

# Lọc các điểm nằm trong giới hạn ảnh để tránh lỗi tràn mảng (IndexError)
h_img, w_img = img2.shape[:2]
valid = (x1 >= 0) & (x2 < w_img) & (y1 >= 0) & (y2 < h_img)

# Lấy các hệ số khoảng cách theo công thức ma trận trong tài liệu:
# Công thức: 1 / ((x2 - x1)*(y2 - y1)) * [x2 - x] * [Q] * [y2 - y]
# Vì x2 - x1 = 1 và y2 - y1 = 1 nên mẫu số bằng 1.

dx = x2 - u_d
dx_inv = u_d - x1
dy = y2 - v_d
dy_inv = v_d - y1

# Lấy giá trị pixel tại 4 điểm Q tương ứng:
# Q11: (y1, x1), Q12: (y2, x1), Q21: (y1, x2), Q22: (y2, x2)
# Lưu ý: img2 có dạng [y, x, channel]
for c in range(3):  # Duyệt qua 3 kênh màu BGR
  Q11 = img2[y1[valid], x1[valid], c].astype(float)
  Q12 = img2[y2[valid], x1[valid], c].astype(float)
  Q21 = img2[y1[valid], x2[valid], c].astype(float)
  Q22 = img2[y2[valid], x2[valid], c].astype(float)

  # Tính theo nhân ma trận từ công thức trong ảnh[cite: 2]:
  # [x2 - x, x - x1] * [[Q11, Q12], [Q21, Q22]] * [[y2 - y], [y - y1]]
  v_x = dx[valid]
  v_x_inv = dx_inv[valid]
  v_y = dy[valid]
  v_y_inv = dy_inv[valid]

  # Nhân từng phần theo đúng bản chất ma trận
  term1 = v_x * (Q11 * v_y + Q12 * v_y_inv)
  term2 = v_x_inv * (Q21 * v_y + Q22 * v_y_inv)

  # Gán giá trị sau khi nội suy vào ảnh mới
  img_new[v[valid], u[valid], c] = np.clip(
      np.round(term1 + term2), 0, 255
  ).astype(np.uint8)

# Hiển thị ảnh kết quả tự code thủ công
cv2.imshow("Bilinear Manual Interpolation", img_new)
cv2.waitKey(0)
cv2.destroyAllWindows()