import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, leastsq
from tqdm import tqdm


SHIFT_DIR = "../../data/shift/"
RESULT_DIR = '../../data/result/'
M_FIT_DIR = '../../data/m_fit/'


def parabola_fit(a, b):
    def _parabola(_x, *params):
        _y = np.zeros_like(_x)

        for _i, _param in enumerate(params):
            _y += _param * _x ** (_i+2)

        return _y

    po, pc = curve_fit(_parabola, b, a, p0=[1])

    return po, pc


def m_fit2(v, a, b):
    def _model_func_m_xy(_p, _x, _y):
        return _y - _x / np.tan(np.pi - v * _x / _p[0])

    params = [8000]
    res = leastsq(_model_func_m_xy, params, args=(b, a), ftol=1e-16, gtol=1e-16, xtol=1e-16)

    return res


def m_fit(v, a, b):
    def _model_func_m_xy(_x, _m):
        return _x / np.tan(np.pi - v * _x / _m)

    return curve_fit(_model_func_m_xy, b, a, p0=[3000])


def check_parabola_fitting(u, q, d, popt, a, b):
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.xlabel('$\it{x}$ [px]', fontsize=28)
    plt.ylabel('$\it{y}$ [px]', fontsize=28)
    plt.title(f"[u, q, d] = [{u}, {q}, {d}]")
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlim(-900, 380)
    plt.ylim(0, 1024)
    plt.grid()

    x = np.linspace(-900, 380 - 1, 1280)
    y = popt[0] * x ** 2

    plt.plot(x, y, color='r', linewidth=3)
    plt.scatter(b, a, c='k')

    plt.show()


def check_fitting(u, q, d, m, a, b, px2mm, mode):
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.xlabel('$\it{x}$ [mm]', fontsize=28)
    plt.ylabel('$\it{y}$ [mm]', fontsize=28)
    plt.title(f"[u, q, d] = [{u}, {q}, {d}]")
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlim(-900*px2mm, 380*px2mm)
    plt.ylim(0, 1024*px2mm)
    plt.grid()

    theta_original = np.linspace(0, 2 * np.pi, 1001)
    theta = np.append(theta_original[1:500], theta_original[501:-1])
    r = m * (np.pi - theta) / u / np.sin(theta)

    plt.plot(r * np.sin(theta), r * np.cos(theta) + m / u, color='b', linewidth=3)

    plt.scatter(b, a, c='k')

    if mode == "save":
        fig.savefig(M_FIT_DIR + f"u_{u}_q_{q}_d_{d}.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()


def fit():
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

    # set result array
    m_list = np.zeros(0)
    e_list = np.zeros(0)

    for u in tqdm(u_list):
        for q in q_list:
            for d in d_list:
                # load shift a, b
                sab = np.loadtxt(SHIFT_DIR + f'u_{u}_q_{q}_d_{d}.csv', delimiter=',')
                sa, sb = sab[0, 1:], sab[1, 1:]

                # parabola fitting
                popt_p, _ = parabola_fit(sa, sb)

                # check parabola fitting
                # check_parabola_fitting(u, q, d, popt_p, sa, sb)

                sa *= px2mm
                sb *= px2mm

                # m fitting
                popt_m = m_fit2(u, sa[sa < 60], sb[sa < 60])

                # check fitting
                check_fitting(u, q, d, popt_m[0], sa[sa < 60], sb[sa < 60], px2mm, mode="save")

                # save tmp result
                m_list = np.append(m_list, popt_m[0])

    # save result
    np.savetxt(RESULT_DIR + 'm_fitting_result.csv', m_list, delimiter=',')


def load_result():
    return np.loadtxt(RESULT_DIR + 'm_fitting_result.csv', delimiter=',')


def graph_d_180(m_list, mode):
    fig, ax = plt.subplots(figsize=(18, 10))
    plt.xlabel('$\it{U}$ [mm/s]', fontsize=32)
    plt.ylabel('$\it{m}$ [mm$^{2}$/s]', fontsize=32)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.xlim(170, 255)
    plt.ylim(3000, 8000)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(25))
    plt.grid(zorder=9)

    # q = 215
    plt.scatter(175, m_list[0], s=400, marker=',', color='b', zorder=10, label='$\it{Q}$ = 2.15 L/min')
    plt.scatter(200, m_list[9], s=400, marker=',', color='b', zorder=10)
    plt.scatter(225, m_list[18], s=400, marker=',', color='b', zorder=10)
    plt.scatter(250, m_list[27], s=400, marker=',', color='b', zorder=10)

    # q = 304
    plt.scatter(175, m_list[3], s=400, marker='o', color='g', zorder=10, label='       3.04')
    plt.scatter(200, m_list[12], s=400, marker='o', color='g', zorder=10)
    plt.scatter(225, m_list[21], s=400, marker='o', color='g', zorder=10)
    plt.scatter(250, m_list[30], s=400, marker='o', color='g', zorder=10)

    # q = 429
    plt.scatter(175, m_list[6], s=400, marker='^', color='r', zorder=10, label='       4.29')
    plt.scatter(200, m_list[15], s=400, marker='^', color='r', zorder=10)
    plt.scatter(225, m_list[24], s=400, marker='^', color='r', zorder=10)
    plt.scatter(250, m_list[33], s=400, marker='^', color='r', zorder=10)

    plt.legend(loc='upper left', fontsize=32, facecolor='#E0E0E0')

    if mode == "save":
        fig.savefig(M_FIT_DIR + f"graph_d_{18}.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()


def graph_d_240(m_list, mode):
    fig, ax = plt.subplots(figsize=(18, 10))
    plt.xlabel('$\it{U}$ [mm/s]', fontsize=32)
    plt.ylabel('$\it{m}$ [mm$^{2}$/s]', fontsize=32)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.xlim(170, 255)
    plt.ylim(3000, 8000)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(25))
    plt.grid(zorder=9)

    # q = 215
    plt.scatter(175, m_list[1], s=400, marker=',', color='b', zorder=10, label='$\it{Q}$ = 2.15 L/min')
    plt.scatter(200, m_list[10], s=400, marker=',', color='b', zorder=10)
    plt.scatter(225, m_list[19], s=400, marker=',', color='b', zorder=10)
    plt.scatter(250, m_list[28], s=400, marker=',', color='b', zorder=10)

    # q = 304
    plt.scatter(175, m_list[4], s=400, marker='o', color='g', zorder=10, label='       3.04')
    plt.scatter(200, m_list[13], s=400, marker='o', color='g', zorder=10)
    plt.scatter(225, m_list[22], s=400, marker='o', color='g', zorder=10)
    plt.scatter(250, m_list[31], s=400, marker='o', color='g', zorder=10)

    # q = 429
    plt.scatter(175, m_list[7], s=400, marker='^', color='r', zorder=10, label='       4.29')
    plt.scatter(200, m_list[16], s=400, marker='^', color='r', zorder=10)
    plt.scatter(225, m_list[25], s=400, marker='^', color='r', zorder=10)
    plt.scatter(250, m_list[34], s=400, marker='^', color='r', zorder=10)

    plt.legend(loc='upper left', fontsize=32, facecolor='#E0E0E0')

    if mode == "save":
        fig.savefig(M_FIT_DIR + f"graph_d_{24}.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()


def graph_d_300(m_list, mode):
    fig, ax = plt.subplots(figsize=(18, 10))
    plt.xlabel('$\it{U}$ [mm/s]', fontsize=32)
    plt.ylabel('$\it{m}$ [mm$^{2}$/s]', fontsize=32)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.xlim(170, 255)
    plt.ylim(3000, 8000)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(25))
    plt.grid(zorder=9)

    # q = 215
    plt.scatter(175, m_list[2], s=400, marker=',', color='b', zorder=10, label='$\it{Q}$ = 2.15 L/min')
    plt.scatter(200, m_list[11], s=400, marker=',', color='b', zorder=10)
    plt.scatter(225, m_list[20], s=400, marker=',', color='b', zorder=10)
    plt.scatter(250, m_list[29], s=400, marker=',', color='b', zorder=10)

    # q = 304
    plt.scatter(175, m_list[5], s=400, marker='o', color='g', zorder=10, label='       3.04')
    plt.scatter(200, m_list[14], s=400, marker='o', color='g', zorder=10)
    plt.scatter(225, m_list[23], s=400, marker='o', color='g', zorder=10)
    plt.scatter(250, m_list[32], s=400, marker='o', color='g', zorder=10)

    # q = 429
    plt.scatter(175, m_list[8], s=400, marker='^', color='r', zorder=10, label='       4.29')
    plt.scatter(200, m_list[17], s=400, marker='^', color='r', zorder=10)
    plt.scatter(225, m_list[26], s=400, marker='^', color='r', zorder=10)
    plt.scatter(250, m_list[35], s=400, marker='^', color='r', zorder=10)

    plt.legend(loc='upper left', fontsize=32, facecolor='#E0E0E0')

    if mode == "save":
        fig.savefig(M_FIT_DIR + f"graph_d_{30}.png", dpi=300, bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    # fit()
    results = load_result()
    graph_d_180(results, mode="save")
    graph_d_240(results, mode="save")
    graph_d_300(results, mode="save")

    print(0)
