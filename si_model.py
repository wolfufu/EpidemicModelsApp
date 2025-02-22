import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def si_model(y, t, beta):
    S, I = y
    dSdt = -beta * S * I
    dIdt = beta * S * I
    return [dSdt, dIdt]

def main():
    # Параметры модели
    beta = 0.3   # скорость передачи
    S0 = 0.99    # начальное количество восприимчивых
    I0 = 0.01    # начальное количество инфицированных
    y0 = [S0, I0]

    # Время моделирования (дни)
    t = np.linspace(0, 160, 160)

    # Решение системы дифференциальных уравнений
    sol = odeint(si_model, y0, t, args=(beta,))

    # Построение графика
    S, I = sol.T
    plt.plot(t, S, 'b', label='Восприимчивые')
    plt.plot(t, I, 'r', label='Инфицированные')
    plt.xlabel('Дни')
    plt.ylabel('Доля населения')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
