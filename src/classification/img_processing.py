import numpy as np


def make_bg(bg, img):
    return np.minimum(bg, img)


def sub_bg(bg, img):
    return img - bg


def get_threshold(freq, N, classes):
    """
    The threshold values are chosen to maximize the total sum of pairwise
    variances between the thresholded gray-level classes. See Notes and [1]_
    for more details.
    The input image must be grayscale, and the input classes must be 2, 3, 4,
    or 5.
    :param
        freq: (L + 1) ndarray
            Frequency array.
        classes: int
            Number of classes to be thresholded, i.e. the number of
            resulting regions.
    :return
        sigma_max: float
            max(sigma)
        variances: (classes) ndarray
            Array containing between-class variances for the desired classes.
        th: (classes - 1) ndarray
            Array containing the threshold values for the desired classes.
    References
    ----------
    .. [1] Liao, P-S., Chen, T-S. and Chung, P-C., "A fast algorithm for
           multilevel thresholding", Journal of Information Science and
           Engineering 17 (5): 713-727, 2001. Available at:
           <https://ftp.iis.sinica.edu.tw/JISE/2001/200109_01.pdf>
           :DOI:`10.6688/JISE.2001.17.5.1`
    """

    L = 255  # max gray-level

    fi = freq  # frequency: Sum[i=0, 255] fi[i] = N (= wid * hei)
    pi = np.zeros(L + 1)  # probability: Sum[i=0, 255] pi[i] = 1

    for i in range(L + 1):
        pi[i] = fi[i] / N

    p = np.zeros((L + 1, L + 1))  # u-v interval zeroth-order moment; P(u, v)
    s = np.zeros((L + 1, L + 1))  # u-v interval first-order moment; S(u, v)
    h = np.zeros((L + 1, L + 1))  # modified between-class variance; H(u, v)

    p[0][0] = 0
    s[0][0] = 0

    for v in range(L):
        p[0][v + 1] = p[0][v] + pi[v]
        s[0][v + 1] = s[0][v] + (v + 1) * pi[v]

    for u in range(L):
        for v in range(L):
            if u > v:
                p[u][v] = 0.0
                s[u][v] = 0.0
                h[u][v] = 0.0
            else:
                p[u + 1][v] = p[0][v] - p[0][u]
                s[u + 1][v] = s[0][v] - s[0][u]

                if p[u][v] == 0:
                    h[u][v] = 0.0
                else:
                    h[u][v] = s[u][v] ** 2 / p[u][v]

    sigma_max = 0  # max(sigma)
    variances = np.zeros(classes)  # between-class variances array
    th = np.zeros(classes - 1, dtype=int)  # thresholds array

    # calculation sigma, searching arg max: thresholds, and re-formatting image
    if classes == 2:
        for t0 in range(L - classes):
            sigma = h[0][t0] + h[t0 + 1][L - 1]

            if sigma_max < sigma:
                sigma_max = sigma
                variances[0] = h[0][t0]
                variances[1] = h[t0 + 1][L - 1]
                th[0] = t0

    elif classes == 3:
        for t0 in range(L - classes):
            for t1 in range(t0 + 1, L - classes + 1):
                sigma = h[0][t0] + h[t0 + 1][t1] + h[t1 + 1][L - 1]

                if sigma_max < sigma:
                    sigma_max = sigma
                    variances[0] = h[0][t0]
                    variances[1] = h[t0 + 1][t1]
                    variances[2] = h[t1 + 1][L - 1]
                    th[0] = t0
                    th[1] = t1

    elif classes == 4:
        for t0 in range(L - classes):
            for t1 in range(t0 + 1, L - classes + 1):
                for t2 in range(t1 + 1, L - classes + 2):
                    sigma = h[0][t0] + h[t0 + 1][t1] + h[t1 + 1][t2] + \
                            h[t2 + 1][L - 1]

                    if sigma_max < sigma:
                        sigma_max = sigma
                        variances[0] = h[0][t0]
                        variances[1] = h[t0 + 1][t1]
                        variances[2] = h[t1 + 1][t2]
                        variances[3] = h[t2 + 1][L - 1]
                        th[0] = t0
                        th[1] = t1
                        th[2] = t2

    elif classes == 5:
        for t0 in range(L - classes):
            for t1 in range(t0 + 1, L - classes + 1):
                for t2 in range(t1 + 1, L - classes + 2):
                    for t3 in range(t2 + 1, L - classes + 3):
                        sigma = h[0][t0] + h[t0 + 1][t1] + h[t1 + 1][t2] \
                                + h[t2 + 1][t3] + h[t3 + 1][L - 1]

                        if sigma_max < sigma:
                            sigma_max = sigma
                            variances[0] = h[0][t0]
                            variances[1] = h[t0 + 1][t1]
                            variances[2] = h[t1 + 1][t2]
                            variances[3] = h[t2 + 1][t3]
                            variances[4] = h[t3 + 1][L - 1]
                            th[0] = t0
                            th[1] = t1
                            th[2] = t2
                            th[3] = t3

    else:
        return None

    return sigma_max, variances, th


def n_ary_encoding(im, th, m=0):
    classes = len(th) + 1
    th_im = im.copy()
    L = 255

    l = np.linspace(0, L, classes, dtype=int)

    if classes == 2:
        th_im[im < th[0]] = l[0]
        th_im[th[0] <= im] = l[1]

    elif classes == 3:
        th_im[im < th[0]] = l[0]
        th_im[(th[0] <= im) & (im < th[1])] = l[1]
        th_im[th[1] <= im] = l[2]

    elif classes == 4:
        if m == 1:
            th_im[im < th[0]] = l[0]
            th_im[(th[0] <= im) & (im < th[1])] = l[1]
            th_im[(th[1] <= im) & (im < th[2])] = l[0]
            th_im[th[2] <= im] = l[0]

        else:
            th_im[im < th[0]] = l[0]
            th_im[(th[0] <= im) & (im < th[1])] = l[1]
            th_im[(th[1] <= im) & (im < th[2])] = l[2]
            th_im[th[2] <= im] = l[3]

    elif classes == 5:
        th_im[im < th[0]] = l[0]
        th_im[(th[0] <= im) & (im < th[1])] = l[1]
        th_im[(th[1] <= im) & (im < th[2])] = l[2]
        th_im[(th[2] <= im) & (im < th[3])] = l[3]
        th_im[th[3] <= im] = l[4]

    else:
        return None

    return th_im
