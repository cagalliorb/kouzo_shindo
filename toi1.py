#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sympy import *
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.integrate

"""constant"""
E = 70 * (10 ** 9)  # [Pa]
b = 1  # [m] 自由
h = 0.001  # [m]
l = 0.1  # [m]
rho = 2.7 * (10 ** 3)  # [kg/m^3]
const_for_no_dim = 1 / (l ** 2) * (E * ((2 * h) ** 3 * b / 12) / (rho * b * 2 * h)) ** (1 / 2)

"""user variable
phiとしていくつの関数を用いるかをfunc_numに代入
それぞれのphiの決め方は今のところ境界条件を満たすように手計算
"""
x = Symbol('x')
func_num = 7
phi1 = 1
phi2 = x / l
phi3 = 2.5 * (x / l) ** 4 - 3 * (x / l) ** 5 + (x / l) ** 6
phi4 = 7 * (x / l) ** 4 - 63 / 10 * (x / l) ** 5 + (x / l) ** 7
phi5 = 14 * (x / l) ** 4 - 56 / 5 * (x / l) ** 5 + (x / l) ** 8
phi6 = 48 * (x / l) ** 4 - 18 * (x / l) ** 5 + (x / l) ** 9
phi7 = 75 / 2 * (x / l) ** 4 - 27 * (x / l) ** 5 + (x / l) ** 10
k1 = 4.7300 / l
k2 = 7.8532 / l
k3 = 10.9956 / l
k4 = 14.1371 / l
k5 = 17.2787 / l
# phi1 = sinh(k1 * x) + sin(k1 * x) + \
#       (sin(k1 * l) - sinh(k1 * l)) / (cosh(k1 * l) - cos(k1 * l)) * \
#       (cosh(k1 * x) + cos(k1 * x))
# phi2 = sinh(k2 * x) + sin(k2 * x) + \
#       (sin(k2 * l) - sinh(k2 * l)) / (cosh(k2 * l) - cos(k2 * l)) * \
#       (cosh(k2 * x) + cos(k2 * x))
# phi3 = sinh(k3 * x) + sin(k3 * x) + \
#       (sin(k3 * l) - sinh(k3 * l)) / (cosh(k3 * l) - cos(k3 * l)) * \
#       (cosh(k3 * x) + cos(k3 * x))
# phi4 = sinh(k4 * x) + sin(k4 * x) + \
#       (sin(k4 * l) - sinh(k4 * l)) / (cosh(k4 * l) - cos(k4 * l)) * \
#       (cosh(k4 * x) + cos(k4 * x))
# phi5 = sinh(k5 * x) + sin((k5 * x)) + \
#       (sin(k5 * l) - sinh(k5 * l)) / (cosh(k5 * l) - cos(k5 * l)) * \
#       (cosh(k5 * x) + cos(k5 * x))
dd_phi1 = diff(diff(phi1, x), x)
dd_phi2 = diff(diff(phi2, x), x)
dd_phi3 = diff(diff(phi3, x), x)
dd_phi4 = diff(diff(phi4, x), x)
dd_phi5 = diff(diff(phi5, x), x)
dd_phi6 = diff(diff(phi6, x), x)
dd_phi7 = diff(diff(phi7, x), x)
phi_list = [phi1, phi2, phi3, phi4, phi5, phi6, phi7]
dd_phi_list = [dd_phi1, dd_phi2, dd_phi3, dd_phi4, dd_phi5, dd_phi6, dd_phi7]


def calc_EI(x_f, is_tapered):
    """
    :param x_f: fはfloatの意. sympyのxシンボルと区別.
    :param is_tapered: taperの場合1,一様断面ならそれ以外
    :return:
    """
    if is_tapered:
        return E * b * (h ** 3) / 12 * (1 + 2 * x_f / l) ** 3
    else:
        return E * b * (2 * h) ** 3 / 12  # 2hの一様断面の場合


def calc_mu(x_f, is_tapered):
    """
    :param x_f: fはfloatの意. sympyのxシンボルと区別.
    :param is_tapered: taperの場合1,一様断面ならそれ以外
    :return:
    """
    if is_tapered:
        return rho * b * h * (1 + 2 * x_f / l)
    else:
        return rho * b * (2 * h)  # 2hの一様断面の場合


def calc_integ_f_kij(x_f, is_tapered, i, j):
    dd_phi1_2 = dd_phi_list[i].subs([(x, x_f)]) * dd_phi_list[j].subs([(x, x_f)])
    return dd_phi1_2 * calc_EI(x_f, is_tapered)


def calc_integ_f_mij(x_f, is_tapered, i, j):
    phi1_2 = phi_list[i].subs([(x, x_f)]) * phi_list[j].subs([(x, x_f)])
    return phi1_2 * calc_mu(x_f, is_tapered)


def calc_k_matrix(is_tapered):
    kij_f = np.zeros((func_num, func_num))  # kij行列用のarrayを確保
    for i in range(func_num):
        for j in range(func_num):
            if i <= j:
                kij_f[i, j] = float(integrate(calc_EI(x, is_tapered) * dd_phi_list[i] * dd_phi_list[j], (x, 0, l)))
                # integral = scipy.integrate.quad(calc_integ_f_kij, 0, l, args=(is_tapered, i, j))
                # kij_f[i, j] = integral[0]  # o番目が積分値,1番目はerror
            if i > j:
                kij_f[i, j] = float(integrate(calc_EI(x, is_tapered) * dd_phi_list[j] * dd_phi_list[i], (x, 0, l)))
                # integral = scipy.integrate.quad(calc_integ_f_kij, 0, l, args=(is_tapered, j, i))
                # kij_f[i, j] = integral[0]
    return kij_f


def calc_m_matrix(is_tapered):
    mij_f = np.zeros((func_num, func_num))  # mij行列用のarrayを確保
    for i in range(func_num):
        for j in range(func_num):
            if i <= j:
                mij_f[i, j] = float(integrate(calc_mu(x, is_tapered) * phi_list[i] * phi_list[j], (x, 0, l)))
                # integral = scipy.integrate.quad(calc_integ_f_mij, 0, l, args=(is_tapered, i, j))
                # mij_f[i, j] = integral[0]
            if i > j:
                mij_f[i, j] = float(integrate(calc_mu(x, is_tapered) * phi_list[j] * phi_list[i], (x, 0, l)))
                # integral = scipy.integrate.quad(calc_integ_f_mij, 0, l, args=(is_tapered, j, i))
                # mij_f[i, j] = integral[0]
    return mij_f


kij_taper = calc_k_matrix(1)
mij_taper = calc_m_matrix(1)
kij_uni = calc_k_matrix(0)
mij_uni = calc_m_matrix(0)


def calc_omega_ci(kij, mij):
    """
    kij*x = omega**2*mij*x という一般化固有値問題の解として
    omega,ciなどを求める
    :param kij:
    :param mij:
    :return:
    """
    eig_val, eig_vec = scipy.linalg.eig(kij, mij)
    omega = eig_val ** (1 / 2)
    print(omega)
    for i in range(len(eig_vec)):  # 正規化
        eig_vec[:, i] = eig_vec[:, i] / np.linalg.norm(eig_vec[:, i])
    print(eig_vec)
    return eig_vec


def calc_disp(c_i, x_f):
    """
    位置xにおける変位を計算
    :param c_i:
    :param x_f:
    :return:
    """
    return c_i[0] * 1 \
           + c_i[1] * phi2.subs([(x, x_f)]) \
           + c_i[2] * phi3.subs([(x, x_f)]) \
           + c_i[3] * phi4.subs([(x, x_f)]) \
           + c_i[4] * phi5.subs([(x, x_f)]) \
           + c_i[5] * phi6.subs([(x, x_f)]) \
           + c_i[6] * phi7.subs([(x, x_f)])


def main(c_1i, c_2i, c_3i):
    """
    1,2,3次のモードをまとめてプロット
    :param c_1i:
    :param c_2i:
    :param c_3i:
    :return:
    """
    x_list = np.linspace(0, l, 1000)
    y_list1, y_list2, y_list3 = [], [], []
    for x_f in x_list:
        y_list1.append(calc_disp(c_1i, x_f))
        y_list2.append(calc_disp(c_2i, x_f))
        y_list3.append(calc_disp(c_3i, x_f))
    plt.plot(x_list, y_list1, label="1st mode")
    plt.plot(x_list, y_list2, label="2nd mode")
    plt.plot(x_list, y_list3, label="3rd mode")
    plt.legend()
    plt.xlabel("x[m]")
    plt.ylim(-1.5, 1.5)
    plt.show()


if __name__ == "__main__":
    # taper 解析解から
    ci_vector = calc_omega_ci(kij_taper, mij_taper)
    ci3_taper = ci_vector[:, 4]
    ci2_taper = ci_vector[:, 3]
    ci1_taper = ci_vector[:, 2]
    main(0.42 * ci1_taper, 0.3 * 5 / 4 * 0.97 * ci2_taper, 0.3 * 5 / 4 * 1.01 * ci3_taper)  # 縮尺は適当に決めている
