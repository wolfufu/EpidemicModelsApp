import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def M_model(y, t, beta, gamma, nu):
    S, I, R, M = y
    dSdt = -beta * S * I + nu * R
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I - nu * R
    dMdt = -nu * M
    return [dSdt, dIdt, dRdt, dMdt]

def solve_M_model(S0, I0, R0, M0, beta, gamma, nu, days):
    y0 = [S0, I0, R0, M0]
    t = np.linspace(0, days, days)
    sol = odeint(M_model, y0, t, args=(beta, gamma, nu))
    return t, sol.T

S0, I0, R0, M0 = 0.99, 0.01, 0, 0
beta, gamma, nu = 0.3, 0.1, 0.01
days = 160

t, (S, I, R, M) = solve_M_model(S0, I0, R0, M0, beta, gamma, nu, days)

plt.plot(t, S, label='Восприимчивые')
plt.plot(t, I, label='Инфицированные')
plt.plot(t, R, label='Выздоровевшие')
plt.plot(t, M, label='Вакцинированные')
plt.xlabel('Дни')
plt.ylabel('Доля населения')
plt.title('M-модель (Multi-stage)')
plt.legend()
plt.show()
