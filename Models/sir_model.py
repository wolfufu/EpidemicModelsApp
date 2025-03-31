import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def sir_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def main():
    beta = 0.3   # скорость передачи
    gamma = 0.1  # скорость выздоровления
    S0 = 0.99    # начальное количество восприимчивых
    I0 = 0.01    # начальное количество инфицированных
    R0 = 0.0     # начальное количество выздоровевших
    y0 = [S0, I0, R0]

    # Время моделирования (дни)
    t = np.linspace(0, 160, 160)

    # Решение системы дифференциальных уравнений
    sol = odeint(sir_model, y0, t, args=(beta, gamma))

    print(sol)

    # Построение графика
    S, I, R = sol.T
    plt.plot(t, S, 'b', label='Восприимчивые')
    plt.plot(t, I, 'r', label='Инфицированные')
    plt.plot(t, R, 'g', label='Выздоровевшие')
    plt.xlabel('Дни')
    plt.ylabel('Доля населения')
    plt.title('SIR-модель')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
