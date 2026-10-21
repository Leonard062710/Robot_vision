import numpy as np
import matplotlib.pyplot as plt


def drawCamera(ax, pos, R, length_scl, head_size):
    pos = np.asarray(pos, dtype=float)
    R = np.asarray(R, dtype=float)

    colors = ("red", "green", "blue")
    offset = length_scl * 0.045
    for i in range(3):
        direction = R[:, i]
        direction = direction / np.linalg.norm(direction)
        arrow = direction * length_scl
        start = pos + offset * direction
        ax.quiver(
            *start, *arrow,
            color=colors[i],
            linewidth = 1.75,
            arrow_length_ratio = head_size,
        )

if __name__ == "__main__":
    from detect_images import T, pW_corners

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection = "3d" )

    pos = T[:3, 3] * 100
    R = T[:3, :3]

    drawCamera(ax, pos, R, 10.0, 0.135)

    world_corners = np.asarray(pW_corners, dtype=float) * 100
    planes = (
        ([0, 1, 3, 2, 0], "tab:red"),
        ([4, 5, 7, 6, 4], "tab:purple"),
        ([8, 10, 11, 9, 8], "tab:green"),
    )
    for corner_indices, color in planes:
        plane_corners = world_corners[corner_indices]
        ax.plot(
            plane_corners[:, 0],
            plane_corners[:, 1],
            plane_corners[:, 2],
            color=color,
            marker="o",
            markerfacecolor="red",
            markeredgecolor="steelblue",
            linewidth=2,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_xlim([20, -25])
    ax.set_ylim([0, 30])
    ax.set_zlim([0, -46])

    ax.set_xticks(np.arange(20, -21, -10))
    ax.set_yticks(np.arange(0, 26, 5))
    ax.set_zticks(np.arange(0, -41, -10))

    ax.set_box_aspect((45, 30, 46))
    plt.show()
