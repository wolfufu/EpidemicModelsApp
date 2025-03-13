import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Функция для вычисления правых частей уравнений системы MSEIR
def mseir_model(y, t, mu, delta, beta, sigma, gamma):
    M, S, E, I, R = y
    N = M + S + E + I + R  # Общее население

    # Дифференциальные уравнения
    dMdt = mu * N - delta * M - mu * M
    dSdt = delta * M - beta * S * I / N - mu * S
    dEdt = beta * S * I / N - sigma * E - mu * E
    dIdt = sigma * E - gamma * I - mu * I
    dRdt = gamma * I - mu * R

    return [dMdt, dSdt, dEdt, dIdt, dRdt]

# Параметры модели
mu = 0.01      # Естественная смертность (и рождаемость, если популяция стабильна)
delta = 0.1    # Скорость потери материнских антител
beta = 0.5     # Коэффициент передачи инфекции
sigma = 1/5    # Инкубационный период (скорость перехода E -> I)
gamma = 1/7    # Скорость выздоровления (скорость перехода I -> R)

# Начальные условия
M0 = 0.1       # Доля людей с материнскими антителами
S0 = 0.8       # Доля восприимчивых
E0 = 0.01      # Доля инфицированных, но не заразных
I0 = 0.05      # Доля инфицированных
R0 = 0.04      # Доля выздоровевших
y0 = [M0, S0, E0, I0, R0]  # Начальные условия для всех групп

# Время моделирования (в днях)
t = np.linspace(0, 160, 160)  # 160 дней

# Решение системы дифференциальных уравнений
solution = odeint(mseir_model, y0, t, args=(mu, delta, beta, sigma, gamma))
M, S, E, I, R = solution.T  # Транспонирование для разделения переменных

# Построение графиков
plt.figure(figsize=(10, 6))
plt.plot(t, M, label='Материнский иммунитет', color='blue')
plt.plot(t, S, label='Восприимчивые', color='green')
plt.plot(t, E, label='Инкубационные', color='orange')
plt.plot(t, I, label='Инфицированные', color='red')
plt.plot(t, R, label='Выздоровевшие', color='purple')
plt.xlabel('Время (дни)')
plt.ylabel('Доля населения')
plt.title('MSEIR-модель')
plt.legend()
plt.grid(True)
plt.show()
