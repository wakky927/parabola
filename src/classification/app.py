import os
import threading

import cv2
import numpy as np

import env
import img_processing as ip
import module


if __name__ == '__main__':
    IN_DIR, OUT_DIR = env.io_dir(user=os.environ.get("USER"))
    U, Q, D = env.uqd(user=os.environ.get("USER"))

    MAKE_BG_FLAG = False  # for skip making background image process
    SUB_BG_FLAG = False  # for skip background subtraction process
    GET_TH_FLAG = False  # for skip get threshold process
    N_ARY_FLAG = False  # for skip n-ary encoding process
    N_ARY_FLAG = False  # for skip n-ary encoding process
    FREQ_CNT_FLAG = True  # for particle frequency count process

    threads = []

    for u in U:
        for q in Q:
            for d in D:
                # for get threshold process
                freq = np.zeros(env.L+1)
                freq_t = np.zeros((env.T, env.L+1))

                if MAKE_BG_FLAG:
                    print(f"[{u}, {q}, {d}] making background image process")

                    bg = np.full((env.HEIGHT, env.WIDTH), env.L)
                    for i in range(env.N):
                        FILE = env.file_name(u=u, q=q, d=d, i=i)
                        img = cv2.imread(IN_DIR + f"{u}/{q}/{d}/" + FILE, 0)
                        bg = ip.make_bg(bg=bg, img=img)

                    # save bg img
                    cv2.imwrite(OUT_DIR + f"bg/u_{u}_q_{q}_d_{d}.bmp", bg)
                    np.savetxt(OUT_DIR + f"bg/u_{u}_q_{q}_d_{d}.csv", bg, delimiter=',')

                if SUB_BG_FLAG:
                    print(f"[{u}, {q}, {d}] background subtraction process")

                    for i in range(env.T):
                        t = threading.Thread(target=module.sub_bg, args=(u, q, d, i, freq_t[i]))
                        t.start()
                        threads.append(t)

                    for thread in threads:
                        thread.join()

                    for i in range(env.T):
                        freq += freq_t[i]

                    # save freq
                    np.savetxt(OUT_DIR + f"freq/freq_u_{u}_q_{q}_d_{d}_ppm_0.csv", freq, delimiter=',')

                if GET_TH_FLAG:
                    print(f"[{u}, {q}, {d}] get threshold process")

                    # load freq
                    freq = np.loadtxt(OUT_DIR + f"freq/freq_u_{u}_q_{q}_d_{d}_ppm_0.csv", delimiter=',')

                    # get threshold
                    _, _, threshold = ip.get_threshold(freq, N=env.N*env.HEIGHT*env.WIDTH, classes=env.CLASSES, L=env.L)

                    # save threshold
                    np.savetxt(OUT_DIR + f"threshold/th_u_{u}_q_{q}_d_{d}_ppm_0.csv", threshold, delimiter=',')

                if N_ARY_FLAG:
                    print(f"[{u}, {q}, {d}] n-ary encoding process")

                    # load threshold
                    threshold = np.loadtxt(OUT_DIR + f"threshold/th_u_{u}_q_{q}_d_{d}_ppm_0.csv", delimiter=',')

                    # n-ary encoding
                    for i in range(env.N):
                        sub_img = cv2.imread(OUT_DIR + f"bg_sub/{u}/{q}/{d}/{i:08}.bmp", 0)
                        n_ary_img = ip.n_ary_encoding(img=sub_img, th=threshold)

                        # save n-ary img
                        cv2.imwrite(OUT_DIR + f"n_ary/{u}/{q}/{d}/{i:08}.bmp", n_ary_img)
                        np.savetxt(OUT_DIR + f"n_ary/{u}/{q}/{d}/{i:08}.csv", n_ary_img, delimiter=',')

                if FREQ_CNT_FLAG:
                    print(f"[{u}, {q}, {d}] particle frequency count process")

                    freq_data = np.zeros((env.HEIGHT, env.WIDTH))

                    for i in range(env.N):
                        n_ary_img = np.loadtxt(OUT_DIR + f"n_ary/{u}/{q}/{d}/{i:08}.csv", delimiter=',')
                        freq_data += np.where(n_ary_img == env.L/4, 1, 0)

                    freq_data /= env.N

                    # save freq data
                    np.savetxt(OUT_DIR + f"freq_cnt/{u}/{q}/{d}/{i:08}.csv", freq_data, delimiter=',')
