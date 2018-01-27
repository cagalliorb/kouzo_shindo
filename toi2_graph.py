#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

"""constant"""
rho = 0.01  # [kg/m^3]
E = 10  # [Pa]
a = (E / rho) ** (1 / 2)  # [m/s]
V_0 = 1  # [m/s]
m = 1  # [kg]
A = 1  # [m^2]
T = 5  # [s]


def calc_sigma(x, t):
    if (x <= a * t):
        return -rho * a * V_0 * np.exp(rho * A / m * (x - a * t))
    else:
        return 0


def main():
    div_num = 1000
    x_list = np.linspace(0, 2 * a * T, div_num)
    stress_list = []
    for x in x_list:
        stress_list.append(calc_sigma(x, T))

    plt.plot(x_list, stress_list)
    plt.xlabel("x[m]")
    plt.ylabel("$\sigma[Pa]$")
    plt.xlim(0,2*a*T)
    plt.show()

    y_list = np.linspace(0, 20)  # 20は棒のwidthなので適当
    stress_2d_list = []
    for y in y_list:
        stress_2d_list.append([])
        for x in x_list:
            stress_2d_list[-1].append(calc_sigma(x, T))
    plt.pcolormesh(x_list, y_list, stress_2d_list, cmap='hsv')  # 等高線図の生成。cmapで色付けの規則を指定する.
    pp = plt.colorbar(orientation="vertical")  # カラーバーの表示
    pp.set_label("stress[Pa]")  # カラーバーのラベル

    plt.xlabel('x[m]')
    plt.ylabel('width[m]')
    plt.axis('equal')
    plt.xlim(0, 2 * a * T)
    plt.yticks([0,20])
    plt.axis("image")
    plt.show()


if __name__ == "__main__":
    main()
