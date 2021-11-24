import os

import cv2
import numpy as np

import env
import img_processing as ip


IN_DIR, OUT_DIR = env.io_dir(user=os.environ.get("USER"))


def sub_bg(u, q, d, i, freq):
    # load bg img
    bg = np.loadtxt(OUT_DIR + f"bg/u_{u}_q_{q}_d_{d}.csv", delimiter=',')

    # subtract bg img
    for j in range(int(i*env.N/env.T), int((i+1)*env.N/env.T)):
        FILE = env.file_name(u=u, q=q, d=d, i=j)
        img = cv2.imread(IN_DIR + f"{u}/{q}/{d}/" + FILE, 0)
        sub_img, freq = ip.sub_bg(bg=bg, img=img, freq=freq, L=env.L)

        # save sub img
        cv2.imwrite(OUT_DIR + f"bg_sub/{u}/{q}/{d}/{j:08}.bmp", sub_img)
        np.savetxt(OUT_DIR + f"bg_sub/{u}/{q}/{d}/{j:08}.csv", sub_img, delimiter=',')
