import os

import cv2
import numpy as np

import env
import img_processing as ip


if __name__ == '__main__':
    IN_DIR, OUT_DIR = env.io_dir(user=os.environ.get("USER"))
    U, Q, D = env.uqd(user=os.environ.get("USER"))

    for u in U:
        for q in Q:
            for d in D:
                # -------- background subtraction --------

                flag = True  # for skip making bg img

                # make bg img
                if flag:
                    bg = np.full((1024, 1280), 255)
                    for i in range(0, 1200):
                        FILE = env.file_name(u=u, q=q, d=d, i=i)
                        img = cv2.imread(IN_DIR + f"{u}/{q}/{d}/" + FILE, 0)
                        bg = ip.make_bg(bg=bg, img=img)

                    # save bg img
                    cv2.imwrite(OUT_DIR + f"bg/u_{u}_q_{q}_d_{d}.bmp", bg)
                    np.savetxt(OUT_DIR + f"bg/u_{u}_q_{q}_d_{d}.csv", bg, delimiter=',')

                # load bg img
                bg = np.loadtxt(OUT_DIR + f"bg/u_{u}_q_{q}_d_{d}.csv", delimiter=',')

                # subtract bg img
                for i in range(0, 1200):
                    FILE = env.file_name(u=u, q=q, d=d, i=i)
                    img = cv2.imread(IN_DIR + f"{u}/{q}/{d}/" + FILE, 0)
                    sub_img = ip.sub_bg(bg=bg, img=img)

                    # save sub img
                    cv2.imwrite(OUT_DIR + f"bg_sub/{u}/{q}/{d}/{i:08}.bmp", bg)
                    np.savetxt(OUT_DIR + f"bg_sub/{u}/{q}/{d}/{i:08}.csv", bg, delimiter=',')

                # -------- otsu --------


