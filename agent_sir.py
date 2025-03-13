import random
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, state):
        self.state = state  # S, I, or R

def agent_based_SIR(agents, beta, gamma, steps):
    S_history, I_history, R_history = [], [], []
    
    for _ in range(steps):
        new_infections = []
        new_recoveries = []
        
        for agent in agents:
            if agent.state == 'I':
                if random.random() < gamma:
                    agent.state = 'R'
                else:
                    for other in agents:
                        if other.state == 'S' and random.random() < beta:
                            new_infections.append(other)
        for infected in new_infections:
            infected.state = 'I'
        
        S_history.append(sum(1 for a in agents if a.state == 'S'))
        I_history.append(sum(1 for a in agents if a.state == 'I'))
        R_history.append(sum(1 for a in agents if a.state == 'R'))
    
    return S_history, I_history, R_history

def plot_agent_based(S_history, I_history, R_history):
    steps = len(S_history)
    t = range(steps)
    plt.plot(t, S_history, label='Восприимчивые')
    plt.plot(t, I_history, label='Инфицированные')
    plt.plot(t, R_history, label='Выздоровевшие')
    plt.title('Agent-based модель')
    plt.legend()
    plt.show()

agents = [Agent('S') for _ in range(990)] + [Agent('I') for _ in range(10)]
beta, gamma = 0.3, 0.1
steps = 160

S_history, I_history, R_history = agent_based_SIR(agents, beta, gamma, steps)
plot_agent_based(S_history, I_history, R_history)
