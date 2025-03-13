import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Система уравнений SIQR-модели
def siqr_model(y, t, beta, gamma, delta, mu):
    S, I, Q, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I - delta * I
    dQdt = delta * I - mu * Q
    dRdt = gamma * I + mu * Q
    return [dSdt, dIdt, dQdt, dRdt]

# Параметры модели
beta = 0.3    # Коэффициент заражения
gamma = 0.1   # Коэффициент выздоровления инфицированных
delta = 0.05  # Коэффициент изоляции инфицированных
mu = 0.05     # Коэффициент выздоровления изолированных

# Начальные условия (начальное количество людей в каждой группе)
S0 = 0.99  # Восприимчивые (99%)
I0 = 0.01  # Инфицированные (1%)
Q0 = 0.0   # Изолированные (карантин)
R0 = 0.0   # Выздоровевшие
N = S0 + I0 + Q0 + R0  # Общая популяция (нормализовано)

# Вектор начальных условий
y0 = [S0, I0, Q0, R0]

# Временной интервал для моделирования (например, 160 дней)
t = np.linspace(0, 160, 160)

# Решение системы уравнений
solution = odeint(siqr_model, y0, t, args=(beta, gamma, delta, mu))
S, I, Q, R = solution.T

# Построение графиков
plt.figure(figsize=(10,6))
plt.plot(t, S, 'b', label='Восприимчивые')
plt.plot(t, I, 'r', label='Инфицированные')
plt.plot(t, Q, 'g', label='Изолированные')
plt.plot(t, R, 'k', label='Выздоровевшие')
plt.xlabel('Дни')
plt.ylabel('Доля населения')
plt.title('SIQR-модель')
plt.legend()
plt.grid(True)
plt.show()
