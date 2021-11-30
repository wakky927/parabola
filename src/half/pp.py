import matplotlib.pyplot as plt
import numpy as np


SHIFT_DIR = "../../data/shift/"
RESULT_DIR = '../../data/result/'
PP_DIR = '../../data/pp/'


def load_pp(u, q, d, mode):
    pp = None

    if mode == "dg":
        pp = np.loadtxt(RESULT_DIR + f"freq_cnt_u_{u}_q_{q}_d_{d}_ppm_0.csv", delimiter=',')

    elif mode == "lg":
        pp = np.loadtxt(RESULT_DIR + f"2_u_{u}_q_{q}_d_{d}_ppm_0.csv", delimiter=',')

    return pp


def curve(u, m, x0, y0, px2mm):
    theta_original = np.linspace(0, 2 * np.pi, 1001)
    theta = np.append(theta_original[1:500], theta_original[501:-1])
    r = m * (np.pi - theta) / u / np.sin(theta)
    b = r * np.sin(theta) / px2mm - x0
    a = 1024 - ((r * np.cos(theta) + m / u) / px2mm - y0)

    return a, b


def draw_pp(data, x, y, px2mm, mode):
    X, Y = np.mgrid[0:1024, 0:1280] * px2mm

    fig, ax = plt.subplots(figsize=(12, 10))
    plt.xlabel('$\it{x}$ [mm]', fontsize=28)
    plt.ylabel('$\it{y}$ [mm]', fontsize=28)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlim(0, 1024*px2mm)
    plt.ylim(0, 1280*px2mm)

    plt.pcolormesh(X, Y, data, cmap='jet', alpha=1)
    pp = plt.colorbar()
    for t in pp.ax.get_yticklabels():
        t.set_fontsize(24)
    pp.set_label('\nparticle probability', size=28)
    plt.clim(0, 1.0)

    plt.plot(x*px2mm, y*px2mm, color='r', linewidth=6)

    if mode == "save":
        return fig

    else:
        plt.show()
        return


def main():
    # set params
    u_list = [175, 200, 225, 250]
    q_list = [215, 304, 429]
    d_list = [18, 24, 30]

    # calibration
    length = 10
    x1 = 505
    y1 = 447
    x2 = 506
    y2 = 530

    px2mm = length / np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    m = np.loadtxt(RESULT_DIR + "m_fitting_result.csv", delimiter=',')

    i = 0
    for u in u_list:
        for q in q_list:
            for d in d_list:
                data_dg = load_pp(u, q, d, mode="dg")
                data_lg = load_pp(u, q, d, mode="lg")

                x0_y0 = np.loadtxt(SHIFT_DIR + "x0_y0.csv", delimiter=',')
                x0, y0 = x0_y0[i, 0], x0_y0[i, 1]

                a, b = curve(u, m[i], x0, y0, px2mm)

                fig = draw_pp(data_dg, a, b, px2mm, mode="save")
                fig.savefig(PP_DIR + f"dg/u_{u}_q_{q}_d_{d}.png", dpi=300, bbox_inches='tight')

                fig = draw_pp(data_lg, a, b, px2mm, mode="save")
                fig.savefig(PP_DIR + f"lg/u_{u}_q_{q}_d_{d}.png", dpi=300, bbox_inches='tight')

                i += 1

    return 0


if __name__ == '__main__':
    main()
