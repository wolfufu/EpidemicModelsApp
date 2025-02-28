import numpy as np
import matplotlib.pyplot as plt

def stochastic_SIR(S0, I0, R0, beta, gamma, population, days):
    S, I, R = S0, I0, R0
    S_history, I_history, R_history = [S], [I], [R]
    
    for _ in range(days):
        infected_prob = beta * I / population
        recovered_prob = gamma * I
        
        new_infected = np.random.binomial(S, infected_prob)
        new_recovered = np.random.binomial(I, recovered_prob)
        
        S -= new_infected
        I += new_infected - new_recovered
        R += new_recovered
        
        S_history.append(S)
        I_history.append(I)
        R_history.append(R)
    
    return S_history, I_history, R_history

def plot_stochastic(S_history, I_history, R_history):
    days = len(S_history)
    t = np.arange(0, days, 1)
    plt.plot(t, S_history, label='Восприимчивые')
    plt.plot(t, I_history, label='Инфицированные')
    plt.plot(t, R_history, label='Выздоровевшие')
    plt.xlabel('Дни')
    plt.ylabel('Число людей')
    plt.legend()
    plt.show()

S0, I0, R0 = 990, 10, 0
beta = 0.3
gamma = 0.1
population = 1000
days = 160

S_history, I_history, R_history = stochastic_SIR(S0, I0, R0, beta, gamma, population, days)
plot_stochastic(S_history, I_history, R_history)
