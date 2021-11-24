def io_dir(user):
    if user == "lfc":
        IN_DIR = "/media/lfc/ボリューム/B4/original/2020_12_03/piv/"
        OUT_DIR = "/media/lfc/ボリューム/M1/result/towing_single_bp/"

    elif user == "phi":
        IN_DIR = ""
        OUT_DIR = ""

    else:
        IN_DIR = ""
        OUT_DIR = ""

    return IN_DIR, OUT_DIR


def uqd(user):
    if user == "lfc":
        U = [175, 200, 225, 250]
        Q = [215, 429]
        D = [24, 30]

    elif user == "phi":
        U = []
        Q = []
        D = []

    else:
        U = []
        Q = []
        D = []

    return U, Q, D


def file_name(u, q, d, i):
    return f"u_{u}_q_{q}_d_{d}_ppm_0_CR600x2 1836-ST-C-086_{i:08}.bmp"


T = 8

N = 1000
HEIGHT, WIDTH = 1024, 1280
CLASSES = 4

L = 255
