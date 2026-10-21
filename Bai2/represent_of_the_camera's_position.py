from detect_images import M_calib, img1, pW_corners, j
import numpy as np
import cv2
import matplotlib.pyplot as plt
def drawCamera(ax,pos,dir,length_scl,head_size) :
    T = np.zeros((4,4))
    T[:3,:3] = M_calib[:3,:3]
    T[:3,3] = M_calib[:3,3]
    T[3,3] = 1
    print(T)
