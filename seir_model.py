import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def seir_model(y, t, beta, sigma, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I
    dEdt = beta * S * I - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

def main():
    # Параметры модели
    beta = 0.3   # скорость передачи
    sigma = 1/5.1 # инкубационный период (обратная величина)
    gamma = 0.1  # скорость выздоровления
    S0 = 0.99    # начальное количество восприимчивых
    E0 = 0.0     # начальное количество латентных
    I0 = 0.01    # начальное количество инфицированных
    R0 = 0.0     # начальное количество выздоровевших
    y0 = [S0, E0, I0, R0]

    # Время моделирования (дни)
    t = np.linspace(0, 160, 160)

    # Решение системы дифференциальных уравнений
    sol = odeint(seir_model, y0, t, args=(beta, sigma, gamma))

    # Построение графика
    S, E, I, R = sol.T
    plt.plot(t, S, 'b', label='Восприимчивые')
    plt.plot(t, E, 'y', label='Латентные')
    plt.plot(t, I, 'r', label='Инфицированные')
    plt.plot(t, R, 'g', label='Выздоровевшие')
    plt.xlabel('Дни')
    plt.ylabel('Доля населения')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
