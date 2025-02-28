import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def SIQR_model(y, t, beta, gamma, delta):
    S, I, Q, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - delta * I - gamma * I
    dQdt = delta * I - gamma * Q
    dRdt = gamma * (I + Q)
    return [dSdt, dIdt, dQdt, dRdt]

def solve_SIQR(S0, I0, Q0, R0, beta, gamma, delta, days):
    y0 = [S0, I0, Q0, R0]
    t = np.linspace(0, days, days)
    sol = odeint(SIQR_model, y0, t, args=(beta, gamma, delta))
    return t, sol.T

S0, I0, Q0, R0 = 0.99, 0.01, 0, 0
beta, gamma, delta = 0.3, 0.1, 0.05
days = 160

t, (S, I, Q, R) = solve_SIQR(S0, I0, Q0, R0, beta, gamma, delta, days)

plt.plot(t, S, label='Восприимчивые')
plt.plot(t, I, label='Инфицированные')
plt.plot(t, Q, label='Карантин')
plt.plot(t, R, label='Выздоровевшие')
plt.xlabel('Дни')
plt.ylabel('Доля населения')
plt.legend()
plt.show()

