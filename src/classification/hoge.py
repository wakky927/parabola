import os


SUPER_DIR = "/media/lfc/ボリューム/B4/original/2020_12_03/piv/"
DIR = "u_250_q_305_d_18_ppm_0_CR600x2 1836-ST-C-086_1/"

if __name__ == '__main__':
    for i in range(1294):
        os.rename(SUPER_DIR + DIR + f"u_250_q_305_d_18_ppm_0_CR600x2 1836-ST-C-086_{i:08}.bmp",
                  SUPER_DIR + DIR + f"u_250_q_304_d_18_ppm_0_CR600x2 1836-ST-C-086_{i:08}.bmp")
