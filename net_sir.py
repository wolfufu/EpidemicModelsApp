import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def network_SIR(G, beta, gamma, initial_infected):
    status = {node: 'S' for node in G.nodes}
    for node in initial_infected:
        status[node] = 'I'
        
    S_history, I_history, R_history = [], [], []
    
    while 'I' in status.values():
        new_status = status.copy()
        for node in G.nodes:
            if status[node] == 'I':
                if np.random.rand() < gamma:
                    new_status[node] = 'R'
                for neighbor in G.neighbors(node):
                    if status[neighbor] == 'S' and np.random.rand() < beta:
                        new_status[neighbor] = 'I'
        status = new_status
        
        S_history.append(list(status.values()).count('S'))
        I_history.append(list(status.values()).count('I'))
        R_history.append(list(status.values()).count('R'))
    
    return S_history, I_history, R_history

def plot_network_SIR(S_history, I_history, R_history):
    days = len(S_history)
    t = np.arange(0, days, 1)
    plt.plot(t, S_history, label='Восприимчивые')
    plt.plot(t, I_history, label='Инфицированные')
    plt.plot(t, R_history, label='Выздоровевшие')
    plt.xlabel('Дни')
    plt.ylabel('Число людей')
    plt.title('Сетевая модель')
    plt.legend()
    plt.show()

G = nx.erdos_renyi_graph(1000, 0.01)
beta = 0.3
gamma = 0.1
initial_infected = [np.random.choice(list(G.nodes))]

S_history, I_history, R_history = network_SIR(G, beta, gamma, initial_infected)
plot_network_SIR(S_history, I_history, R_history)
