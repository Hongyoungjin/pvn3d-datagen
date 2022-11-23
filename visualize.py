import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import cv2
dir = os.path.join(os.getcwd(),"src/points/data")

for file_idx in range(0,100):
    pose_name = ("pose_%05d.npy")%(file_idx)
    mask_name = ("mask_%05d.npy")%(file_idx)
    color_name = ("color_image_%05d.npy")%(file_idx)
    depth_name = ("depth_image_%05d.npy")%(file_idx)

    pose = np.load(dir + "/" + pose_name, allow_pickle=True)
    mask = np.load(dir + "/" + mask_name, allow_pickle=True)
    depth = np.load(dir + "/" + depth_name, allow_pickle=True)
    color = np.load(dir + "/" + color_name, allow_pickle=True)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax1.imshow(depth)
    ax2.imshow(color)
    ax3.imshow(mask)
    
    
    ax1.set_title("depth")
    ax2.set_title("color")
    ax3.set_title("mask")
    plt.show()
    