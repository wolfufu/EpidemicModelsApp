import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def multi_stage_model(t, y, beta, k, gamma):
    S, *I, R = y
    dS_dt = -beta * S * I[0] + gamma * R
    dI_dt = [beta * S * I[0] - k[0] * I[0]]
    for j in range(1, len(I)):
        dI_dt.append(k[j-1] * I[j-1] - k[j] * I[j])
    dR_dt = k[-1] * I[-1] - gamma * R
    return [dS_dt] + dI_dt + [dR_dt]

# Параметры модели
beta = 0.5   # коэффициент заражения
k = [0.3, 0.2, 0.1]  # скорости переходов между стадиями инфекции
gamma = 0.05  # скорость потери иммунитета

# Начальные условия
S0 = 0.9  # 90% восприимчивы
I0 = [0.1, 0.0, 0.0]  # Начинаем с 10% инфицированных на первой стадии
R0 = 0.0  # 0% выздоровевших

y0 = [S0] + I0 + [R0]

# Временной интервал моделирования
t_span = (0, 160)
t_eval = np.linspace(*t_span, 1000)

# Решение системы дифференциальных уравнений
sol = solve_ivp(multi_stage_model, t_span, y0, args=(beta, k, gamma), t_eval=t_eval)

# Визуализация результатов
plt.figure(figsize=(10, 5))
plt.plot(sol.t, sol.y[0], label='Восприимчивые', color='blue')
for j in range(len(I0)):
    plt.plot(sol.t, sol.y[j+1], label=f'I{j+1} (Стадия {j+1})', linestyle='dashed')
plt.plot(sol.t, sol.y[-1], label='Выздоровевшие', color='green')
plt.xlabel('Время')
plt.ylabel('Доля популяции')
plt.title('Multi-stage (M) модель распространения инфекции')
plt.legend()
plt.grid()
plt.show()