import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm


BINARY_DIR = '../data/binary/'
RESULT_DIR = '../data/result/'


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


def fitting(data):
    def _parabola_func(_x, *_params):
        _y = np.zeros_like(_x)

        for _i, _param in enumerate(_params):
            _y += np.array(_param * _x ** _i)

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

    popt, pcov = curve_fit(_parabola_func, b[1:], a[1:], p0=[1, 1, 1])
    perr = np.sqrt(np.diag(pcov))

    return popt, perr


def analysis():
    # set params
    U = [175, 200, 225, 250]
    Q = [215, 304, 429]
    D = [18, 24, 30]

    length = 10
    x1, y1 = 505, 447
    x2, y2 = 506, 530
    z = length / np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # get a
    a_list = np.zeros(0)
    e_list = np.zeros(0)

    for u in tqdm(U):
        for q in Q:
            for d in D:
                # read mask img
                bin_img = cv2.imread(BINARY_DIR + f'b_u_{u}_q_{q}_d_{d}_ppm_0.bmp', 0)

                # detect boundary
                boundary = detect_boundary(img=bin_img, hei=1024, wid=1280)

                # parabola fitting
                p, pe = fitting(boundary)

                # save tmp result
                a_list = np.append(a_list, -p[2] / z)
                e_list = np.append(e_list, -pe[2] / z)

    # save result
    np.savetxt(RESULT_DIR + 'parabola_fitting_result.csv', np.array([a_list, e_list]), delimiter=',')


def load_result():
    return np.loadtxt(RESULT_DIR + 'parabola_fitting_result.csv', delimiter=',')


def graph_d_180(a_list, e_list):
    fig, ax = plt.subplots(figsize=(18, 10))
    plt.xlabel('$\it{Q}$ [L/min]', fontsize=32)
    plt.ylabel('$\it{a}$ [mm$^{-1}$]', fontsize=32)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    # plt.xlim(0, 250)
    # plt.ylim(0, 0.025)
    # ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(25))
    plt.grid(zorder=9)

    # u = 175
    plt.scatter(2.15, a_list[0], s=400, marker=',', color='b', zorder=10, label='175')
    plt.scatter(3.04, a_list[3], s=400, marker=',', color='b', zorder=10)
    plt.scatter(4.29, a_list[6], s=400, marker=',', color='b', zorder=10)

    plt.errorbar(2.15, a_list[0], yerr=e_list[0], capsize=18, capthick=3, ecolor='black')
    plt.errorbar(3.04, a_list[3], yerr=e_list[3], capsize=18, capthick=3, ecolor='black')
    plt.errorbar(4.29, a_list[6], yerr=e_list[6], capsize=18, capthick=3, ecolor='black')

    # u = 200
    plt.scatter(2.15, a_list[9], s=400, marker='o', color='g', zorder=10, label='200')
    plt.scatter(3.04, a_list[12], s=400, marker='o', color='g', zorder=10)
    plt.scatter(4.29, a_list[15], s=400, marker='o', color='g', zorder=10)

    plt.errorbar(2.15, a_list[9], yerr=e_list[9], capsize=18, capthick=3, ecolor='black')
    plt.errorbar(3.04, a_list[12], yerr=e_list[12], capsize=18, capthick=3, ecolor='black')
    plt.errorbar(4.29, a_list[15], yerr=e_list[15], capsize=18, capthick=3, ecolor='black')

    # u = 225
    plt.scatter(2.15, a_list[18], s=400, marker='^', color='r', zorder=10, label='225')
    plt.scatter(3.04, a_list[21], s=400, marker='^', color='r', zorder=10)
    plt.scatter(4.29, a_list[24], s=400, marker='^', color='r', zorder=10)

    plt.errorbar(2.15, a_list[18], yerr=e_list[18], capsize=18, capthick=3, ecolor='black')
    plt.errorbar(3.04, a_list[21], yerr=e_list[21], capsize=18, capthick=3, ecolor='black')
    plt.errorbar(4.29, a_list[24], yerr=e_list[24], capsize=18, capthick=3, ecolor='black')

    # u = 250
    plt.scatter(2.15, a_list[27], s=400, marker='D', color='m', zorder=10, label='250')
    plt.scatter(3.04, a_list[30], s=400, marker='D', color='m', zorder=10)
    plt.scatter(4.29, a_list[33], s=400, marker='D', color='m', zorder=10)

    plt.errorbar(2.15, a_list[27], yerr=e_list[27], capsize=18, capthick=3, ecolor='black')
    plt.errorbar(3.04, a_list[30], yerr=e_list[30], capsize=18, capthick=3, ecolor='black')
    plt.errorbar(4.29, a_list[33], yerr=e_list[33], capsize=18, capthick=3, ecolor='black')

    plt.legend(loc='upper right', fontsize=32)

    plt.show()


def graph_d_180_inverse(a_list, e_list):
    fig, ax = plt.subplots(figsize=(18, 10))
    plt.xlabel('$\it{U}$ [mm/s]', fontsize=32)
    plt.ylabel('$\it{a}$ [mm$^{-1}$]', fontsize=32)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.xlim(0, 250)
    plt.ylim(0, 100)
    # ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(25))
    plt.grid(zorder=9)

    # q = 215
    plt.scatter(175, 1/a_list[0], s=400, marker=',', color='b', zorder=10, label='2.15')
    plt.scatter(200, 1/a_list[9], s=400, marker=',', color='b', zorder=10)
    plt.scatter(225, 1/a_list[18], s=400, marker=',', color='b', zorder=10)
    plt.scatter(250, 1/a_list[27], s=400, marker=',', color='b', zorder=10)

    # q = 304
    plt.scatter(175, 1/a_list[3], s=400, marker='o', color='g', zorder=10, label='3.04')
    plt.scatter(200, 1/a_list[12], s=400, marker='o', color='g', zorder=10)
    plt.scatter(225, 1/a_list[21], s=400, marker='o', color='g', zorder=10)
    plt.scatter(250, 1/a_list[30], s=400, marker='o', color='g', zorder=10)

    # q = 429
    plt.scatter(175, 1/a_list[6], s=400, marker='^', color='r', zorder=10, label='4.29')
    plt.scatter(200, 1/a_list[15], s=400, marker='^', color='r', zorder=10)
    plt.scatter(225, 1/a_list[24], s=400, marker='^', color='r', zorder=10)
    plt.scatter(250, 1/a_list[33], s=400, marker='^', color='r', zorder=10)

    plt.legend(loc='upper right', fontsize=32)

    plt.show()


if __name__ == '__main__':
    # analysis()
    results = load_result()
    graph_d_180(results[0], results[1])
    # graph_d_180_inverse(results[0], results[1])

    print(0)
