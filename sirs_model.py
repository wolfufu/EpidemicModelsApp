import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def sirs_model(y, t, beta, gamma, delta):
    S, I, R = y
    dSdt = -beta * S * I + delta * R
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I - delta * R
    return [dSdt, dIdt, dRdt]

def main():
    # Параметры модели
    beta = 0.3   # скорость передачи
    gamma = 0.1  # скорость выздоровления
    delta = 0.05 # скорость потери иммунитета
    S0 = 0.99    # начальное количество восприимчивых
    I0 = 0.01    # начальное количество инфицированных
    R0 = 0.0     # начальное количество выздоровевших
    y0 = [S0, I0, R0]

    # Время моделирования (дни)
    t = np.linspace(0, 160, 160)

    # Решение системы дифференциальных уравнений
    sol = odeint(sirs_model, y0, t, args=(beta, gamma, delta))

    # Построение графика
    S, I, R = sol.T
    plt.plot(t, S, 'b', label='Восприимчивые')
    plt.plot(t, I, 'r', label='Инфицированные')
    plt.plot(t, R, 'g', label='Выздоровевшие')
    plt.xlabel('Дни')
    plt.ylabel('Доля населения')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
