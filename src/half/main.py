import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import torch
from tqdm import tqdm

from optimize import calc as c


BINARY_DIR = '../../data/binary/'
BOUNDARY_DIR = '../../data/boundary/'
RESULT_DIR = '../../data/result/'


def detect_boundary(img, hei, wid, t=5):
    data = np.zeros((hei, 5))  # [index, start, end, flag, flag]

    # detection boundary
    for j in range(hei):
        data[j][0] = j  # index

        # detection from left side
        for i in range(wid - t):
            if np.all(img[j][i:i + t] == 255):
                data[j][1] = i
                data[j][3] = 1

                break

        # detection from right side
        if j > 700:
            for i in reversed(range(t, wid)):
                if np.all(img[j][i - t:i] == 255):
                    data[j][2] = i

                    if i > 1200:
                        data[j][4] = 0
                    else:
                        data[j][4] = 1

                    break

    return data


def parabola_fit(data):
    def _parabola(_x, *params):
        _y = np.zeros_like(_x)

        for _i, _param in enumerate(params):
            _y += _param * _x ** _i

        return _y

    a = np.zeros(1)
    b = np.zeros(1)

    for i in range(data.shape[0]):
        # left boundary
        if data[i, 3] == 1:
            a = np.append(a, data[i, 0])
            b = np.append(b, data[i, 1])

        # right boundary
        if data[i, 4] == 1:
            a = np.append(a, data[i, 0])
            b = np.append(b, data[i, 2])

    po, pc = curve_fit(_parabola, b[1:], a[1:], p0=[1, 1, 1])

    return po, pc, a, b


def parabola_fit_adam(data):
    a = np.zeros(1)
    b = np.zeros(1)

    for i in range(data.shape[0]):
        # left boundary
        if data[i, 3] == 1:
            a = np.append(a, data[i, 0])
            b = np.append(b, data[i, 1])

        # right boundary
        if data[i, 4] == 1:
            a = np.append(a, data[i, 0])
            b = np.append(b, data[i, 2])

    p = c.my_adam(b[1:], a[1:])

    return p, a, b


def m_fit(v, a, b):
    def _model_func_m_xy(_x, _m):
        return _x / np.tan(np.pi - v * _x / _m)

    return curve_fit(_model_func_m_xy, b, a, p0=[100])


def check_parabola_fitting(u, q, d, popt, a, b):
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.xlabel('$\it{x}$ [px]', fontsize=28)
    plt.ylabel('$\it{y}$ [px]', fontsize=28)
    plt.title(f"[u, q, d] = [{u}, {q}, {d}]")
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlim(0, 1280)
    plt.ylim(1024, 0)
    plt.grid()

    x = np.linspace(0, 1280 - 1, 1280)
    y = popt[0] + popt[1] * x + popt[2] * x ** 2

    plt.plot(x, y, color='r', linewidth=3)
    plt.scatter(b[1:], a[1:], c='k')

    plt.show()


def check_parabola_fitting2(popt, a, b):
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.xlabel('$\it{x}$ [px]', fontsize=28)
    plt.ylabel('$\it{y}$ [px]', fontsize=28)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlim(0 + popt[1] / popt[2] / 2, 1280 + popt[1] / popt[2] / 2)
    plt.ylim(-(1024 + (popt[1]**2 - 4 * popt[2] * popt[0]) / popt[2] / 4), -(0 + (popt[1]**2 - 4 * popt[2] * popt[0]) / popt[2] / 4))
    plt.grid()

    x = np.linspace(0, 1280 - 1, 1280)
    y = popt[0] + popt[1] * x + popt[2] * x ** 2

    x += popt[1] / popt[2] / 2
    y += (popt[1]**2 - 4 * popt[2] * popt[0]) / popt[2] / 4
    a += popt[1] / popt[2] / 2
    b += (popt[1]**2 - 4 * popt[2] * popt[0]) / popt[2] / 4

    plt.plot(x, -y, color='r', linewidth=3)
    plt.scatter(b[1:], -a[1:], c='k')

    plt.show()


def check_fitting(u, q, d, m, a, b):
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.xlabel('$\it{x}$ [mm]', fontsize=28)
    plt.ylabel('$\it{y}$ [mm]', fontsize=28)
    plt.title(f"[u, q, d] = [{u}, {q}, {d}]")
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xlim(0, 1280)
    plt.ylim(1024, 0)
    # plt.xlim(-100, 100)
    # plt.ylim(-1, 120)
    plt.grid()

    theta_original = np.linspace(0, 2 * np.pi, 1001)
    theta = np.append(theta_original[1:500], theta_original[501:-1])
    r = m * (np.pi - theta) / u / np.sin(theta)

    plt.plot(r * np.sin(theta), r * np.cos(theta) + m / u, color='r', linewidth=3)

    plt.scatter(b[1:], a[1:], c='k')
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
    m_sd_list = np.zeros(0)

    for u in tqdm(u_list):
        for q in q_list:
            for d in d_list:
                # read mask img
                bin_img = cv2.imread(BINARY_DIR + f'b_u_{u}_q_{q}_d_{d}_ppm_0.bmp', 0)

                # detect boundary
                boundary = detect_boundary(img=bin_img, hei=1024, wid=1280)

                # parabola fitting
                popt_p, _, A, B = parabola_fit(boundary)

                # save boundary
                np.savetxt(BOUNDARY_DIR + f'u_{u}_q_{q}_d_{d}.csv', np.array([A, B]), delimiter=',')

                # check parabola fitting
                # check_parabola_fitting(u, q, d, popt_p, A, B)
                # check_parabola_fitting2(popt_p, A, B)

                # A += popt_p[1] / popt_p[2] / 2
                # B += (popt_p[1] ** 2 - 4 * popt_p[2] * popt_p[0]) / popt_p[2] / 4
                # A *= -px2mm
                # B *= px2mm

                # m fitting
                # popt_m, pcov_m = m_fit(u, A[A > 300], B[A > 300])

                # check fitting
                # check_fitting(u, q, d, popt_m[0], A[A > 300], B[A > 300])
                #
                # # save tmp result
                # m_list = np.append(m_list, popt_m[0])
                # m_sd_list = np.append(m_sd_list, np.sqrt(pcov_m[0]))

    # save result
    # np.savetxt(RESULT_DIR + 'm_fitting_result.csv', np.array([m_list, m_sd_list]), delimiter=',')


def load_result():
    return np.loadtxt(RESULT_DIR + 'm_fitting_result.csv', delimiter=',')


def graph_d_180(m_list, m_sd_list):
    fig, ax = plt.subplots(figsize=(18, 10))
    plt.xlabel('$\it{U}$ [mm/s]', fontsize=32)
    plt.ylabel('$\it{m}$ [mm$^{2}$]', fontsize=32)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.xlim(170, 260)
    plt.ylim(3000, 4000)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(25))
    plt.grid(zorder=9)

    # q = 215
    plt.scatter(175, m_list[0], s=400, marker=',', color='b', zorder=10, label='2.15')
    plt.scatter(200, m_list[9], s=400, marker=',', color='b', zorder=10)
    plt.scatter(225, m_list[18], s=400, marker=',', color='b', zorder=10)
    plt.scatter(250, m_list[27], s=400, marker=',', color='b', zorder=10)

    plt.errorbar(175, m_list[0], yerr=m_sd_list[0], capsize=18, capthick=3, ecolor='black')
    plt.errorbar(200, m_list[9], yerr=m_sd_list[9], capsize=18, capthick=3, ecolor='black')
    plt.errorbar(225, m_list[18], yerr=m_sd_list[18], capsize=18, capthick=3, ecolor='black')
    plt.errorbar(250, m_list[27], yerr=m_sd_list[27], capsize=18, capthick=3, ecolor='black')

    # q = 304
    plt.scatter(175, m_list[3], s=400, marker='o', color='g', zorder=10, label='3.04')
    plt.scatter(200, m_list[12], s=400, marker='o', color='g', zorder=10)
    plt.scatter(225, m_list[21], s=400, marker='o', color='g', zorder=10)
    plt.scatter(250, m_list[30], s=400, marker='o', color='g', zorder=10)

    plt.errorbar(175, m_list[3], yerr=m_sd_list[3], capsize=18, capthick=3, ecolor='black')
    plt.errorbar(200, m_list[12], yerr=m_sd_list[12], capsize=18, capthick=3, ecolor='black')
    plt.errorbar(225, m_list[21], yerr=m_sd_list[21], capsize=18, capthick=3, ecolor='black')
    plt.errorbar(250, m_list[30], yerr=m_sd_list[30], capsize=18, capthick=3, ecolor='black')

    # q = 429
    plt.scatter(175, m_list[6], s=400, marker='^', color='r', zorder=10, label='4.29')
    plt.scatter(200, m_list[15], s=400, marker='^', color='r', zorder=10)
    plt.scatter(225, m_list[24], s=400, marker='^', color='r', zorder=10)
    plt.scatter(250, m_list[33], s=400, marker='^', color='r', zorder=10)

    plt.errorbar(175, m_list[6], yerr=m_sd_list[6], capsize=18, capthick=3, ecolor='black')
    plt.errorbar(200, m_list[15], yerr=m_sd_list[15], capsize=18, capthick=3, ecolor='black')
    plt.errorbar(225, m_list[24], yerr=m_sd_list[24], capsize=18, capthick=3, ecolor='black')
    plt.errorbar(250, m_list[33], yerr=m_sd_list[33], capsize=18, capthick=3, ecolor='black')

    plt.legend(loc='upper left', fontsize=32)

    plt.show()


if __name__ == '__main__':
    fit()
    # results = load_result()
    # graph_d_180(results[0], results[1])

    print(0)
