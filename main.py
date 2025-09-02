import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkcalendar import DateEntry
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import zipfile
from io import BytesIO

class NumericalMethods:
    def __init__(self):
        self.result = None

    def euler_method(self, model_func, y0, t, args):
        y = np.zeros((len(t), len(y0)))
        y[0] = y0
        
        for i in range(1, len(t)):
            dt = t[i] - t[i-1]
            dy = model_func(y[i-1], t[i-1], *args)
            y[i] = y[i-1] + dy * dt
        self.result = y.T
    
    def runge_kutta_4(self, model_func, y0, t, args):
        y = np.zeros((len(t), len(y0)))
        y[0] = y0
        
        for i in range(1, len(t)):
            dt = t[i] - t[i-1]
            h = dt
            
            k1 = model_func(y[i-1], t[i-1], *args)
            k2 = model_func(y[i-1] + 0.5*h*k1, t[i-1] + 0.5*h, *args)
            k3 = model_func(y[i-1] + 0.5*h*k2, t[i-1] + 0.5*h, *args)
            k4 = model_func(y[i-1] + h*k3, t[i-1] + h, *args)
            
            y[i] = y[i-1] + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        
        self.result = y.T

class EpidemicModels(NumericalMethods):
    def __init__(self):
        self.current_models = []
        self.numeric_methods = []
        self.axes = []
        self.canvases = []
        self.figs = []

    def sir_model(self, y, t, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return np.array([dSdt, dIdt, dRdt])
    
    def si_model(self, y, t, beta):
        S, I = y
        dSdt = -beta * S * I
        dIdt = beta * S * I
        return np.array([dSdt, dIdt])
    
    def sirs_model(self, y, t, beta, gamma, delta):
        S, I, R = y
        dSdt = -beta * S * I + delta * R
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I - delta * R
        return np.array([dSdt, dIdt, dRdt])
    
    def siqr_model(self, y, t, beta, gamma, delta, mu):
        S, I, Q, R = y
        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I - delta * I
        dQdt = delta * I - mu * Q
        dRdt = gamma * I + mu * Q
        return np.array([dSdt, dIdt, dQdt, dRdt])
    
    def seir_model(self, y, t, beta, sigma, gamma):
        S, E, I, R = y
        dSdt = -beta * S * I
        dEdt = beta * S * I - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return np.array([dSdt, dEdt, dIdt, dRdt])
    
    def mseir_model(self, y, t, mu, delta, beta, sigma, gamma):
        M, S, E, I, R = y
        N = M + S + E + I + R  
        dMdt = mu * N - delta * M - mu * M
        dSdt = delta * M - beta * S * I / N - mu * S
        dEdt = beta * S * I / N - sigma * E - mu * E
        dIdt = sigma * E - gamma * I - mu * I
        dRdt = gamma * I - mu * R
        return np.array([dMdt, dSdt, dEdt, dIdt, dRdt])
    
    def multi_stage_model(self, y, t, beta, k1, k2, k3, gamma):
        S, I1, I2, I3, R = y
        dSdt = -beta * S * I1 + gamma * R
        dI1dt = beta * S * I1 - k1 * I1
        dI2dt = k1 * I1 - k2 * I2
        dI3dt = k2 * I2 - k3 * I3
        dRdt = k3 * I3 - gamma * R
        return np.array([dSdt, dI1dt, dI2dt, dI3dt, dRdt])
    
    def run_si_model(self, t, params, initials, plot_index, return_solution=False, method="runge_kutta"):
        y0 = [initials["S0"], initials["I0"]]

        if method == "runge_kutta":
            self.runge_kutta_4(self.si_model, y0, t, (params["beta"],))
        else:
            self.euler_method(self.si_model, y0, t, (params["beta"],))

        S, I = self.result
        
        ax = self.axes[plot_index]
        ax.clear()
        ax.plot(t, S, 'b', label='Восприимчивые')
        ax.plot(t, I, 'r', label='Инфицированные')
        ax.set_ylim(0, 1)
        ax.set_xlabel('Дни')
        ax.set_ylabel('Доля населения')
        ax.grid(True)
        ax.legend()
        ax.set_title('SI-модель')
        self.canvases[plot_index].draw()
        
        if return_solution:
            return {"S": S, "I": I}

    def run_sir_model(self, t, params, initials, plot_index, return_solution=False, method="runge_kutta"):
        y0 = [initials["S0"], initials["I0"], initials["R0"]]

        if method == "runge_kutta":
            self.runge_kutta_4(self.sir_model, y0, t, (params["beta"], params["gamma"]))
        else:
            self.euler_method(self.sir_model, y0, t, (params["beta"], params["gamma"]))

        S, I, R = self.result
        
        ax = self.axes[plot_index]
        ax.clear()
        ax.plot(t, S, 'b', label='Восприимчивые')
        ax.plot(t, I, 'r', label='Инфицированные')
        ax.plot(t, R, 'g', label='Выздоровевшие')
        ax.set_ylim(0, 1)
        ax.set_xlabel('Дни')
        ax.set_ylabel('Доля населения')
        ax.grid(True)
        ax.legend()
        ax.set_title('SIR-модель')
        self.canvases[plot_index].draw()

        if return_solution:
            return {"S": S, "I": I, "R": R}

    def run_sirs_model(self, t, params, initials, plot_index, return_solution=False, method="runge_kutta"):
        y0 = [initials["S0"], initials["I0"], initials["R0"]]

        if method == "runge_kutta":
            self.runge_kutta_4(self.sirs_model, y0, t, (params["beta"], params["gamma"], params["delta"]))
        else:
            self.euler_method(self.sirs_model, y0, t, (params["beta"], params["gamma"], params["delta"]))

        S, I, R = self.result
        
        ax = self.axes[plot_index]
        ax.clear()
        ax.plot(t, S, 'b', label='Восприимчивые')
        ax.plot(t, I, 'r', label='Инфицированные')
        ax.plot(t, R, 'g', label='Выздоровевшие')
        ax.set_ylim(0, 1)
        ax.set_xlabel('Дни')
        ax.set_ylabel('Доля населения')
        ax.grid(True)
        ax.legend()
        ax.set_title('SIRS-модель')
        self.canvases[plot_index].draw()

        if return_solution:
            return {"S": S, "I": I, "R": R}

    def run_seir_model(self, t, params, initials, plot_index, return_solution=False, method="runge_kutta"):
        y0 = [initials["S0"], initials["E0"], initials["I0"], initials["R0"]]

        if method == "runge_kutta":
            self.runge_kutta_4(self.seir_model, y0, t, (params["beta"], params["sigma"], params["gamma"]))
        else:
            self.euler_method(self.seir_model, y0, t, (params["beta"], params["sigma"], params["gamma"]))

        S, E, I, R = self.result
        
        ax = self.axes[plot_index]
        ax.clear()
        ax.plot(t, S, 'b', label='Восприимчивые')
        ax.plot(t, E, 'y', label='Латентные')
        ax.plot(t, I, 'r', label='Инфицированные')
        ax.plot(t, R, 'g', label='Выздоровевшие')
        ax.set_ylim(0, 1)
        ax.set_xlabel('Дни')
        ax.set_ylabel('Доля населения')
        ax.grid(True)
        ax.legend()
        ax.set_title('SEIR-модель')
        self.canvases[plot_index].draw()

        if return_solution:
            return {"S": S, "E": E, "I": I, "R": R}

    def run_mseir_model(self, t, params, initials, plot_index, return_solution=False, method="runge_kutta"):
        y0 = [initials["M0"], initials["S0"], initials["E0"], initials["I0"], initials["R0"]]

        if method == "runge_kutta":
            self.runge_kutta_4(self.mseir_model, y0, t, 
                        (params["mu"], params["delta"], params["beta"], params["sigma"], params["gamma"]))
        else:
            self.euler_method(self.mseir_model, y0, t, 
                        (params["mu"], params["delta"], params["beta"], params["sigma"], params["gamma"]))

        M, S, E, I, R = self.result
        
        ax = self.axes[plot_index]
        ax.clear()
        ax.plot(t, M, 'b', label='Материнский иммунитет')
        ax.plot(t, S, 'g', label='Восприимчивые')
        ax.plot(t, E, 'y', label='Латентные')
        ax.plot(t, I, 'r', label='Инфицированные')
        ax.plot(t, R, 'purple', label='Выздоровевшие')
        ax.set_ylim(0, 1)
        ax.set_xlabel('Дни')
        ax.set_ylabel('Доля населения')
        ax.grid(True)
        ax.legend()
        ax.set_title('MSEIR-модель')
        self.canvases[plot_index].draw()

        if return_solution:
            return {"M": M, "S": S, "E": E, "I": I, "R": R}

    def run_siqr_model(self, t, params, initials, plot_index, return_solution=False, method="runge_kutta"):
        y0 = [initials["S0"], initials["I0"], initials["Q0"], initials["R0"]]

        if method == "runge_kutta":
            self.runge_kutta_4(self.siqr_model, y0, t, 
                        (params["beta"], params["gamma"], params["delta"], params["mu"]))
        else:
            self.euler_method(self.siqr_model, y0, t, 
                        (params["beta"], params["gamma"], params["delta"], params["mu"]))

        S, I, Q, R = self.result
        
        ax = self.axes[plot_index]
        ax.clear()
        ax.plot(t, S, 'b', label='Восприимчивые')
        ax.plot(t, I, 'r', label='Инфицированные')
        ax.plot(t, Q, 'g', label='Изолированные')
        ax.plot(t, R, 'k', label='Выздоровевшие')
        ax.set_ylim(0, 1)
        ax.set_xlabel('Дни')
        ax.set_ylabel('Доля населения')
        ax.grid(True)
        ax.legend()
        ax.set_title('SIQR-модель')
        self.canvases[plot_index].draw()

        if return_solution:
            return {"S": S, "I": I, "Q": Q, "R": R}
        
    def run_m_model(self, t, params, initials, plot_index, return_solution=False, method="runge_kutta"):
        y0 = [initials["S0"], initials["I10"], initials["I20"], initials["I30"], initials["R0"]]

        if method == "runge_kutta":
            self.runge_kutta_4(self.multi_stage_model, y0, t, 
                        (params["beta"], params["k1"], params["k2"], params["k3"], params["gamma"]))
        else:
            self.euler_method(self.multi_stage_model, y0, t, 
                        (params["beta"], params["k1"], params["k2"], params["k3"], params["gamma"]))

        S, I1, I2, I3, R = self.result
        
        ax = self.axes[plot_index]
        ax.clear()
        ax.plot(t, S, 'b', label='Восприимчивые')
        ax.plot(t, I1, 'r--', label='I1 (Стадия 1)')
        ax.plot(t, I2, 'm--', label='I2 (Стадия 2)')
        ax.plot(t, I3, 'c--', label='I3 (Стадия 3)')
        ax.plot(t, R, 'g', label='Выздоровевшие')
        ax.set_ylim(0, 1)
        ax.set_xlabel('Дни')
        ax.set_ylabel('Доля населения')
        ax.grid(True)
        ax.legend()
        ax.set_title('M-модель (3 стадии)')
        self.canvases[plot_index].draw()
        
        if return_solution:
            return {"S": S, "I1": I1, "I2": I2, "I3": I3, "R": R}

class EpidemicApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EpidemicModels")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        self.style.configure('TButton', font=('Arial', 10))
        self.style.configure('TCheckbutton', background='#f0f0f0', font=('Arial', 10))
        self.style.configure('TRadiobutton', background='#f0f0f0', font=('Arial', 10))
        
        self.model = EpidemicModels()
        self.result_data = {}

        self.country_population = {
            "Russia": 146000000,
            "Germany": 83000000,
            "France": 68000000,
            "United Kingdom": 67000000,
            "Italy": 59000000,
            "Spain": 47000000,
            "Poland": 38000000,
            "Netherlands": 17000000,
            "Belgium": 11000000,
            "Sweden": 10000000,
            "Portugal": 10000000,
            "Greece": 10000000,
            "Switzerland": 8000000,
            "Austria": 8000000,
            "Hungary": 9000000,
            "Czech Republic": 10000000,
            "Romania": 19000000,
            "Denmark": 5000000,
            "Finland": 5000000,
            "Norway": 5000000,
            "Ireland": 5000000,
            "China": 1412000000,
            "India": 1408000000,
            "Indonesia": 275000000,
            "Pakistan": 235000000,
            "Bangladesh": 171000000,
            "Japan": 125000000,
            "Philippines": 115000000,
            "Vietnam": 98000000,
            "Turkey": 85000000,
            "Iran": 87000000,
            "Thailand": 70000000,
            "Myanmar": 54000000,
            "South Korea": 51000000,
            "Iraq": 43000000,
            "Afghanistan": 41000000,
            "Saudi Arabia": 36000000,
            "Malaysia": 33000000,
            "Uzbekistan": 35000000,
            "Nepal": 30000000,
            "Sri Lanka": 22000000,
            "Kazakhstan": 19000000,
            "Israel": 9000000,
            "Hong Kong": 7000000,
            "Singapore": 5000000,
            "Nigeria": 218000000,
            "Ethiopia": 123000000,
            "Egypt": 109000000,
            "DR Congo": 97000000,
            "South Africa": 60000000,
            "Tanzania": 63000000,
            "Kenya": 55000000,
            "Algeria": 45000000,
            "Morocco": 37000000,
            "Angola": 35000000,
            "Ghana": 32000000,
            "Mozambique": 32000000,
            "Madagascar": 29000000,
            "Cameroon": 27000000,
            "Ivory Coast": 27000000,
            "Niger": 25000000,
            "Burkina Faso": 22000000,
            "Mali": 21000000,
            "Malawi": 20000000,
            "Zambia": 19000000,
            "Senegal": 17000000,
            "Chad": 17000000,
            "Somalia": 17000000,
            "Zimbabwe": 16000000,
            "US": 335000000,
            "Brazil": 215000000,
            "Mexico": 128000000,
            "Colombia": 52000000,
            "Argentina": 46000000,
            "Canada": 38000000,
            "Peru": 34000000,
            "Venezuela": 28000000,
            "Chile": 19000000,
            "Ecuador": 18000000,
            "Guatemala": 17000000,
            "Cuba": 11000000,
            "Dominican Republic": 11000000,
            "Honduras": 10000000,
            "Bolivia": 12000000,
            "El Salvador": 6000000,
            "Paraguay": 7000000,
            "Nicaragua": 6000000,
            "Costa Rica": 5000000,
            "Panama": 4000000,
            "Uruguay": 3000000,
            "Australia": 26000000,
            "Papua New Guinea": 9000000,
            "New Zealand": 5000000,
            "Fiji": 900000,
            "Yemen": 32000000,
            "United Arab Emirates": 10000000,
            "Jordan": 11000000,
            "Azerbaijan": 10000000,
            "Lebanon": 6000000,
            "Oman": 5000000,
            "Qatar": 3000000,
            "Kuwait": 4000000,
            "Taiwan": 23000000,
            "North Korea": 26000000,
            "South Sudan": 11000000,
            "Eritrea": 6000000,
            "Tajikistan": 10000000,
            "Kyrgyzstan": 7000000,
            "Turkmenistan": 6000000
        }

        self.create_widgets()
        self.set_default_values()

    def validate_sum(self, entries_dict, max_sum=1.0):
        try:
            total = sum(float(entry.get()) for entry in entries_dict.values() if entry.get())
            return True
        except ValueError:
            return False

    def create_widgets(self):
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)

        control_frame = ttk.Frame(main_paned, width=350, relief=tk.RIDGE, padding=10)
        main_paned.add(control_frame, weight=0)

        graph_frame = ttk.Frame(main_paned)
        main_paned.add(graph_frame, weight=1)

        control_notebook = ttk.Notebook(control_frame)
        control_notebook.pack(fill=tk.BOTH, expand=True)

        models_tab = ttk.Frame(control_notebook)
        control_notebook.add(models_tab, text="Модели")
        self.create_models_tab(models_tab)

        params_tab = ttk.Frame(control_notebook)
        control_notebook.add(params_tab, text="Параметры")
        self.create_params_tab(params_tab)

        data_tab = ttk.Frame(control_notebook)
        control_notebook.add(data_tab, text="Данные")
        self.create_data_tab(data_tab)

        self.create_graphs(graph_frame)

    def create_model_params_tab(self, model_code):
        frame = ttk.Frame(self.params_notebook)

        validate_num = self.root.register(self.create_validate_func(0, 1))

        param_group = ttk.LabelFrame(frame, text="Параметры модели", padding=10)
        param_group.pack(fill=tk.X, pady=5)

        init_group = ttk.LabelFrame(frame, text="Начальные значения", padding=10)
        init_group.pack(fill=tk.X, pady=5)

        param_entries = {}
        init_entries = {}

        param_definitions = {
            "SI": [("beta", "β")],
            "SIR": [("beta", "β"), ("gamma", "γ")],
            "SIRS": [("beta", "β"), ("gamma", "γ"), ("delta", "δ")],
            "SEIR": [("beta", "β"), ("sigma", "σ"), ("gamma", "γ")],
            "SIQR": [("beta", "β"), ("gamma", "γ"), ("delta", "δ"), ("mu", "μ")],
            "MSEIR": [("mu", "μ"), ("delta", "δ"), ("beta", "β"), ("sigma", "σ"), ("gamma", "γ")],
            "M": [("beta", "β"), ("k1", "k₁"), ("k2", "k₂"), ("k3", "k₃"), ("gamma", "γ")]
        }

        init_definitions = {
            "SI": [("S0", "S₀"), ("I0", "I₀")],
            "SIR": [("S0", "S₀"), ("I0", "I₀"), ("R0", "R₀")],
            "SIRS": [("S0", "S₀"), ("I0", "I₀"), ("R0", "R₀")],
            "SEIR": [("S0", "S₀"), ("E0", "E₀"), ("I0", "I₀"), ("R0", "R₀")],
            "SIQR": [("S0", "S₀"), ("I0", "I₀"), ("Q0", "Q₀"), ("R0", "R₀")],
            "MSEIR": [("M0", "M₀"), ("S0", "S₀"), ("E0", "E₀"), ("I0", "I₀"), ("R0", "R₀")],
            "M": [("S0", "S₀"), ("I10", "I₁₀"), ("I20", "I₂₀"), ("I30", "I₃₀"), ("R0", "R₀")]
        }

        for code, label in param_definitions.get(model_code, []):
            row = ttk.Frame(param_group)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=25).pack(side=tk.LEFT)
            entry = ttk.Entry(row, width=8, validate="key", 
                            validatecommand=(validate_num, '%P'))
            entry.pack(side=tk.RIGHT)
            param_entries[code] = entry
        
        for code, label in init_definitions.get(model_code, []):
            row = ttk.Frame(init_group)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=25).pack(side=tk.LEFT)
            entry = ttk.Entry(row, width=8, validate="key", 
                            validatecommand=(validate_num, '%P'))
            entry.pack(side=tk.RIGHT)
            init_entries[code] = entry

        def on_entry_change(*args):
            if not self.validate_sum(init_entries):
                messagebox.showwarning("Ошибка", "Сумма начальных значений не должна превышать 1")
        
        for entry in param_entries.values():
            entry.bind("<FocusOut>", on_entry_change)
        
        for entry in init_entries.values():
            entry.bind("<FocusOut>", on_entry_change)

        self.model_param_tabs[model_code] = {
            "frame": frame,
            "param_entries": param_entries,
            "init_entries": init_entries
        }
        self.params_notebook.add(frame, text=model_code)

        default_params = {
            "beta": "0.35",
            "gamma": "0.15",
            "delta": "0.02",
            "sigma": "0.25",
            "mu": "0.03",
            "k1": "0.35",
            "k2": "0.25",
            "k3": "0.15"
        }
        
        for param, value in default_params.items():
            if param in param_entries:
                param_entries[param].insert(0, value)

        model_default_inits = {
            "SI": {
                "S0": "0.95",
                "I0": "0.05"
            },
            "SIR": {
                "S0": "0.95",
                "I0": "0.05",
                "R0": "0.0"
            },
            "SIRS": {
                "S0": "0.93",
                "I0": "0.05",
                "R0": "0.02"
            },
            "SEIR": {
                "S0": "0.94",
                "E0": "0.03",
                "I0": "0.02",
                "R0": "0.01"
            },
            "SIQR": {
                "S0": "0.93",
                "I0": "0.05",
                "Q0": "0.01",
                "R0": "0.01"
            },
            "MSEIR": {
                "M0": "0.05",
                "S0": "0.89",
                "E0": "0.03",
                "I0": "0.02",
                "R0": "0.01"
            },
            "M": {
                "S0": "0.90",
                "I10": "0.06",
                "I20": "0.02",
                "I30": "0.01",
                "R0": "0.01"
            }
        }

        for init, value in model_default_inits.get(model_code, {}).items():
            if init in init_entries:
                init_entries[init].insert(0, value)

    def create_models_tab(self, parent):
        models_group = ttk.LabelFrame(parent, text="Выберите модели (макс. 4)", padding=10)
        models_group.pack(fill=tk.BOTH, pady=5)

        self.model_vars = {}
        models = [
            ('SI', 'Модель SI (восприимчивые-инфицированные)'),
            ('SIR', 'Модель SIR (восприимчивые-инфицированные-выздоровевшие)'),
            ('SIRS', 'Модель SIRS (с временным иммунитетом)'),
            ('SEIR', 'Модель SEIR (с латентным периодом)'),
            ('SIQR', 'Модель SIQR (с изоляцией)'),
            ('MSEIR', 'Модель MSEIR (с материнским иммунитетом)'),
            ('M', 'M-модель (3 стадии инфекции)')  
        ]
        
        for model_code, model_desc in models:
            var = tk.BooleanVar()
            cb = ttk.Checkbutton(models_group, text=model_desc, variable=var, 
                               command=self.update_model_selection)
            cb.pack(anchor='w', padx=5, pady=2)
            self.model_vars[model_code] = var

        method_group = ttk.LabelFrame(parent, text="Метод решения", padding=10)
        method_group.pack(fill=tk.BOTH, pady=5)
        
        self.method_var = tk.StringVar(value="runge_kutta")
        ttk.Radiobutton(method_group, text="Рунге-Кутта 4-го порядка", 
                       variable=self.method_var, value="runge_kutta").pack(anchor='w', padx=5, pady=2)
        ttk.Radiobutton(method_group, text="Метод Эйлера", 
                       variable=self.method_var, value="euler").pack(anchor='w', padx=5, pady=2)

        ttk.Button(parent, text="Запустить моделирование", 
                  command=self.run_simulation).pack(fill=tk.X, pady=10)
        
        ttk.Button(parent, text="Справка по моделям", 
                   command=self.open_model_docs).pack(fill=tk.X, pady=5)
    
    def create_params_tab(self, parent):
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.params_notebook = ttk.Notebook(scrollable_frame)
        self.params_notebook.pack(fill=tk.BOTH, expand=True)
        self.model_param_tabs = {}

        time_group = ttk.LabelFrame(scrollable_frame, text="Временной диапазон", padding=10)
        time_group.pack(fill=tk.X, pady=5)

        ttk.Label(time_group, text="Начальная дата:").pack(anchor='w', pady=(0, 5))
        self.start_entry = DateEntry(time_group, date_pattern='dd.mm.yyyy')
        self.start_entry.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(time_group, text="Конечная дата:").pack(anchor='w', pady=(0, 5))
        self.end_entry = DateEntry(time_group, date_pattern='dd.mm.yyyy')
        self.end_entry.pack(fill=tk.X)

    def create_data_tab(self, parent):
        load_group = ttk.LabelFrame(parent, text="Загрузка данных", padding=10)
        load_group.pack(fill=tk.BOTH, pady=5, expand=True)
            
        ttk.Button(load_group, text="Загрузить данные из CSV", 
                command=self.load_csv_data).pack(fill=tk.X, pady=5)

        export_group = ttk.LabelFrame(parent, text="Экспорт результатов", padding=10)
        export_group.pack(fill=tk.BOTH, pady=5)
            
        ttk.Button(export_group, text="Экспорт в Excel и ZIP", 
                command=self.export_results).pack(fill=tk.X, pady=5)
        
    def create_graphs(self, parent):
        self.model.axes = []
        self.model.canvases = []
        self.model.figs = []
        
        for i in range(4):
            fig = plt.Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)
            
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_facecolor('#f8f8f8')
            
            canvas = FigureCanvasTkAgg(fig, master=parent)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.grid(row=i//2, column=i%2, padx=5, pady=5, sticky='nsew')
            canvas_widget.config(borderwidth=2, relief=tk.GROOVE)
            
            parent.grid_rowconfigure(i//2, weight=1)
            parent.grid_columnconfigure(i%2, weight=1)
            
            self.model.figs.append(fig)
            self.model.axes.append(ax)
            self.model.canvases.append(canvas)

    def create_validate_func(self, min_val, max_val):
        def validate(value):
            if value == "":
                return True
            try:
                num = float(value)
                return min_val <= num <= max_val
            except ValueError:
                return False
        return validate
    
    def set_default_values(self):
        today = datetime.now()
        self.start_entry.set_date(today)
        self.end_entry.set_date(today + timedelta(days=100))
    
    def update_model_selection(self):
        selected_models = [code for code, var in self.model_vars.items() if var.get()]

        if len(selected_models) > 4:
            for code in reversed(self.model_vars):
                if self.model_vars[code].get():
                    self.model_vars[code].set(False)
                    messagebox.showwarning("Предупреждение", "Можно выбрать не более 4 моделей")
                    break
            return

        for code in list(self.model_param_tabs):
            if code not in selected_models:
                tab = self.model_param_tabs[code]["frame"]
                self.params_notebook.forget(tab)
                del self.model_param_tabs[code]

        for code in selected_models:
            if code not in self.model_param_tabs:
                self.create_model_params_tab(code)
    
    def run_simulation(self):
        selected_models = [name for name, var in self.model_vars.items() if var.get()]
        if not selected_models:
            messagebox.showwarning("Ошибка", "Выберите хотя бы одну модель")
            return
        
        for model_code in selected_models:
            model_tab = self.model_param_tabs.get(model_code)
            if not model_tab:
                continue
                
            if not self.validate_sum(model_tab["init_entries"]):
                messagebox.showerror("Ошибка", f"Сумма начальных значений для модели {model_code} превышает 1")
                return

        if self.end_entry.get_date() <= self.start_entry.get_date():
            messagebox.showerror("Ошибка", "Конечная дата должна быть позже начальной")
            return

        self.result_data.clear()
        days = (self.end_entry.get_date() - self.start_entry.get_date()).days
        t = np.linspace(0, days, days + 1)
        method = self.method_var.get()

        for ax in self.model.axes:
            ax.clear()

        for i, model_code in enumerate(selected_models):
            if i >= 4:
                break

            model_tab = self.model_param_tabs.get(model_code)
            if not model_tab:
                continue

            try:
                params = {k: float(e.get()) for k, e in model_tab["param_entries"].items()}
                initials = {k: float(e.get()) for k, e in model_tab["init_entries"].items()}
            except ValueError:
                messagebox.showerror("Ошибка", f"Некорректные значения для модели {model_code}")
                continue

            if model_code == "SI":
                sol = self.model.run_si_model(t, params, initials, i, return_solution=True, method=method)
            elif model_code == "SIR":
                sol = self.model.run_sir_model(t, params, initials, i, return_solution=True, method=method)
            elif model_code == "SIRS":
                sol = self.model.run_sirs_model(t, params, initials, i, return_solution=True, method=method)
            elif model_code == "SEIR":
                sol = self.model.run_seir_model(t, params, initials, i, return_solution=True, method=method)
            elif model_code == "SIQR":
                sol = self.model.run_siqr_model(t, params, initials, i, return_solution=True, method=method)
            elif model_code == "MSEIR":
                sol = self.model.run_mseir_model(t, params, initials, i, return_solution=True, method=method)
            elif model_code == "M":
                sol = self.model.run_m_model(t, params, initials, i, return_solution=True, method=method)
            else:
                sol = None

            if sol:
                self.result_data[model_code] = pd.DataFrame(sol, index=t)

        for canvas in self.model.canvases:
            canvas.draw()

    
    def load_csv_data(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        
        try:
            df = pd.read_csv(path)
            df["Date"] = pd.to_datetime(df["Date"])
            df.sort_values("Date", inplace=True)
            self.csv_data = df

            countries = sorted(df["Country/Region"].dropna().unique())
            
            self.country_selection = tk.Toplevel(self.root)
            self.country_selection.title("Выбор страны и дат")
            self.country_selection.geometry("400x300")

            ttk.Label(self.country_selection, text="Выберите страну:").pack(pady=(10, 5))
            self.country_cb = ttk.Combobox(self.country_selection, values=countries, state="readonly")
            self.country_cb.pack(pady=5)

            ttk.Label(self.country_selection, text="Выберите диапазон дат для начальных данных:").pack(pady=(10, 5))
            
            date_frame = ttk.Frame(self.country_selection)
            date_frame.pack(pady=5)
            
            ttk.Label(date_frame, text="С:").pack(side=tk.LEFT)
            self.data_start_entry = DateEntry(date_frame, date_pattern='dd.mm.yyyy')
            self.data_start_entry.pack(side=tk.LEFT, padx=5)
            
            ttk.Label(date_frame, text="По:").pack(side=tk.LEFT)
            self.data_end_entry = DateEntry(date_frame, date_pattern='dd.mm.yyyy')
            self.data_end_entry.pack(side=tk.LEFT, padx=5)

            min_date = df["Date"].min().to_pydatetime()
            max_date = df["Date"].max().to_pydatetime()
            self.data_start_entry.set_date(min_date)
            self.data_end_entry.set_date(max_date)

            ttk.Button(self.country_selection, text="Загрузить данные", 
                      command=self.process_csv_data).pack(pady=10)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить файл: {str(e)}")
    
    def process_csv_data(self):
        country = self.country_cb.get()
        if not country:
            messagebox.showwarning("Ошибка", "Выберите страну")
            return

        try:
            start_date = self.data_start_entry.get_date()
            end_date = self.data_end_entry.get_date()
            
            if start_date > end_date:
                messagebox.showerror("Ошибка", "Начальная дата не может быть позже конечной")
                return

            df_country = self.csv_data[
                (self.csv_data["Country/Region"] == country) &
                (self.csv_data["Date"] >= pd.to_datetime(start_date)) &
                (self.csv_data["Date"] <= pd.to_datetime(end_date))
            ].copy()
            
            if df_country.empty:
                messagebox.showerror("Ошибка", "Нет данных для выбранного диапазона дат")
                return

            latest = df_country.iloc[-1]
            country = self.country_cb.get()
            total_population = self.country_population.get(country, 1)

            if total_population <= 0:
                total_population = 1

            S0 = (total_population - latest["Confirmed"] - latest["Recovered"] - latest["Deaths"]) / total_population
            I0 = latest["Confirmed"] / total_population
            R0 = (latest["Recovered"] + latest["Deaths"]) / total_population

            for tab in self.model_param_tabs.values():
                init_entries = tab["init_entries"]
                for key in ("S0", "I0", "R0"):
                    if key in init_entries:
                        init_entries[key].delete(0, tk.END)

                if "S0" in init_entries:
                    init_entries["S0"].insert(0, f"{S0:.4f}")
                if "I0" in init_entries:
                    init_entries["I0"].insert(0, f"{I0:.4f}")
                if "R0" in init_entries:
                    init_entries["R0"].insert(0, f"{R0:.4f}")

            self.start_entry.set_date(end_date)
            self.end_entry.set_date(end_date + timedelta(days=100))
            
            self.country_selection.destroy()
            messagebox.showinfo("Успех", f"Данные для {country} за период {start_date.strftime('%d.%m.%Y')}-{end_date.strftime('%d.%m.%Y')} успешно загружены")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось обработать данные: {str(e)}")

        try:
            df_country["Infected"] = df_country["Confirmed"] - df_country["Recovered"] - df_country["Deaths"]
            df_country["Removed"] = df_country["Recovered"] + df_country["Deaths"]
            df_country["Susceptible"] = (total_population - df_country["Infected"] - df_country["Removed"]) / total_population

            I = df_country["Infected"].values / total_population
            R = df_country["Removed"].values / total_population
            S = df_country["Susceptible"].values

            beta_list = []
            gamma_list = []

            for i in range(1, len(df_country)):
                I_prev = I[i - 1]
                I_curr = I[i]
                R_prev = R[i - 1]
                R_curr = R[i]
                S_prev = S[i - 1]

                dI = I_curr - I_prev
                dR = R_curr - R_prev

                gamma_i = None
                if I_prev > 1e-6:
                    gamma_i = dR / I_prev
                    if np.isfinite(gamma_i):
                        gamma_list.append(gamma_i)

                if I_prev > 1e-6 and S_prev * I_prev > 1e-6:
                    gamma_eff = gamma_i if gamma_i is not None else 0.1
                    beta_i = (dI + gamma_eff * I_prev) / (S_prev * I_prev)
                    if np.isfinite(beta_i):
                        beta_list.append(beta_i)

            gamma = np.clip(np.mean(gamma_list), 0.01, 1.0) if gamma_list else 0.1
            beta = np.clip(np.mean(beta_list), 0.01, 1.0) if beta_list else 0.3

            for tab in self.model_param_tabs.values():
                param_entries = tab["param_entries"]
                if "beta" in param_entries:
                    param_entries["beta"].delete(0, tk.END)
                    param_entries["beta"].insert(0, f"{beta:.4f}")
                if "gamma" in param_entries:
                    param_entries["gamma"].delete(0, tk.END)
                    param_entries["gamma"].insert(0, f"{gamma:.4f}")
        except Exception as e:
            print("Не удалось вычислить параметры:", e)
    
    def export_results(self):
        if not self.result_data:
            messagebox.showwarning("Нет данных", "Сначала выполните моделирование")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".zip",
            filetypes=[("ZIP files", "*.zip")],
            initialfile="epidemic_results.zip"
        )
        
        if not save_path:
            return

        try:
            mem_zip = BytesIO()
            with zipfile.ZipFile(mem_zip, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
                for model_name, df in self.result_data.items():
                    excel_buffer = BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                        model_tab = self.model_param_tabs.get(model_name)
                        if not model_tab:
                            continue
                        
                        initials_data = {}
                        for key, entry in model_tab["init_entries"].items():
                            initials_data[key] = entry.get()
                        
                        initials_df = pd.DataFrame({
                            'Параметр': ['S₀ (восприимчивые)', 'I₀ (инфицированные)', 'R₀ (выздоровевшие)',
                                        'E₀ (латентные)', 'Q₀ (изолированные)', 'M₀ (материнский иммунитет)'],
                            'Значение': [
                                initials_data.get("S0", ""),
                                initials_data.get("I0", ""),
                                initials_data.get("R0", ""),
                                initials_data.get("E0", ""),
                                initials_data.get("Q0", ""),
                                initials_data.get("M0", "")
                            ]
                        })
                        initials_df.to_excel(writer, sheet_name='Начальные данные', index=False)
                        
                        params_data = {}
                        for key, entry in model_tab["param_entries"].items():
                            params_data[key] = entry.get()
                        
                        params_df = pd.DataFrame({
                            'Параметр': ['β (скорость заражения)', 'γ (скорость выздоровления)',
                                        'δ (потеря иммунитета)', 'σ (переход в инфекционные)',
                                        'μ (выход из изоляции)'],
                            'Значение': [
                                params_data.get("beta", ""),
                                params_data.get("gamma", ""),
                                params_data.get("delta", ""),
                                params_data.get("sigma", ""),
                                params_data.get("mu", "")
                            ]
                        })
                        params_df.to_excel(writer, sheet_name='Параметры', index=False)
                        
                        method_info = pd.DataFrame({
                            'Информация': ['Метод решения:', 'Выбранная модель:'],
                            'Значение': [
                                self.method_var.get().replace("runge_kutta", "Рунге-Кутта 4-го порядка").replace("euler", "Метод Эйлера"), 
                                model_name
                            ]
                        })
                        
                        method_info.to_excel(writer, sheet_name='Решение', startrow=0, index=False, header=False)
                        df.to_excel(writer, sheet_name='Решение', startrow=3, index=True)
                        
                        worksheet = writer.sheets['Решение']
                        worksheet.write(3, 0, 'Дни')
                        
                        graph_sheet = writer.book.add_worksheet('График')
                        chart = writer.book.add_chart({'type': 'line'})
                        
                        max_row = len(df) + 4
                        categories = f"='Решение'!$A$5:$A${max_row}"
                        
                        for i, col in enumerate(df.columns, 1):
                            col_letter = chr(65 + i)
                            chart.add_series({
                                'name': f"='Решение'!${col_letter}$4",
                                'categories': categories,
                                'values': f"='Решение'!${col_letter}$5:${col_letter}${max_row}",
                            })
                        
                        chart.set_x_axis({'name': 'Дни'})
                        chart.set_y_axis({'name': 'Доля населения'})
                        chart.set_title({'name': f'Модель {model_name} ({method_info.iloc[0,1]})'})
                        graph_sheet.insert_chart('B2', chart, {'x_scale': 2, 'y_scale': 2})
                    
                    zf.writestr(f"{model_name}.xlsx", excel_buffer.getvalue())

            with open(save_path, "wb") as f:
                f.write(mem_zip.getvalue())
            
            messagebox.showinfo("Экспорт завершён", f"Файлы успешно сохранены в архив:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось экспортировать данные: {str(e)}")

    def open_model_docs(self):
        doc_window = tk.Toplevel(self.root)
        doc_window.title("Справка по моделям")
        doc_window.geometry("600x500")

        notebook = ttk.Notebook(doc_window)
        notebook.pack(fill=tk.BOTH, expand=True)

        docs = {
            "SI": {
                "desc": "Модель SI описывает распространение инфекции, у которых нет стадии выздоровления.\n\nСистема дифференциальных уравнений:\n  dS/dt = -βSI\n  dI/dt = βSI",
                "params": "β — скорость передачи инфекции.\n\nНачальные: S₀, I₀",
                "recommended": "β ∈ [0.2, 0.6]\n\nS₀ ≈ 0.99\nI₀ ≈ 0.01",
                "usage": "Подходит для: компьютерных вирусов, хронических инфекций без выздоровления"
            },
            "SIR": {
                "desc": "SIR — базовая эпидемиологическая модель.\n\nСистема дифференциальных уравнений:\n  dS/dt = -βSI\n  dI/dt = βSI - γI\n  dR/dt = γI",
                "params": "β — скорость передачи инфекции\nγ — скорость выздоровления",
                "recommended": "β ≈ 0.3\nγ ≈ 0.1\n\nS₀ ≈ 0.99\nI₀ ≈ 0.01\nR₀ ≈ 0.0",
                "usage": "Подходит для: гриппа, COVID-19, кори и т.п."
            },
            "SIRS": {
                "desc": "SIRS — модель, учитывающая потерю иммунитета.\n\nСистема дифференциальных уравнений:\n  dS/dt = -βSI + δR\n  dI/dt = βSI - γI\n  dR/dt = γI - δR",
                "params": "β — скорость передачи инфекции\nγ — скорость выздоровления\nδ — скорость потери иммунитета",
                "recommended": "β ≈ 0.3\nγ ≈ 0.1\nδ ≈ 0.01\n\nS₀ ≈ 0.99\nI₀ ≈ 0.02\nR₀ ≈ 0.00",
                "usage": "Применяется для заболеваний с временным иммунитетом (грипп, риновирус)"
            },
            "SEIR": {
                "desc": "SEIR — учитывает инкубационный период.\n\nСистема дифференциальных уравнений:\n  dS/dt = -βSI\n  dE/dt = βSI - σE\n  dI/dt = σE - γI\n  dR/dt = γI",
                "params": "β — скорость передачи инфекции\nγ — скорость выздоровления\nσ — скорость перехода в инфекционную фазу",
                "recommended": "β ≈ 0.3\nγ ≈ 0.1\nσ ≈ 0.2\n\nS₀ ≈ 0.99\nE₀ ≈ 0.01\nI₀ ≈ 0.02\nR₀ ≈ 0.00",
                "usage": "Подходит для: COVID-19, Эболы, кори"
            },
            "SIQR": {
                "desc": "SIQR — модель с изоляцией инфицированных.\n\nСистема дифференциальных уравнений:\n  dS/dt = -βSI\n  dI/dt = βSI - γI - δI\n  dQ/dt = δI - μQ\n  dR/dt = γI + μQ",
                "params": "β — скорость передачи инфекции\nγ — скорость выздоровления\nδ — скорость изоляции\nμ — скорость выхода из карантина",
                "recommended": "β ≈ 0.3\nγ ≈ 0.1\nδ ≈ 0.05\nμ ≈ 0.05\n\nS₀ ≈ 0.99\nI₀ ≈ 0.02\nQ₀ ≈ 0.00\nR₀ ≈ 0.00",
                "usage": "Применяется в условиях карантина, например, COVID-19"
            },
            "MSEIR": {
                "desc": "MSEIR — добавляет материнский иммунитет. Модель учитывает естественную рождаемость и рождаемость.\n\nСистема дифференциальных уравнений:\n  dM/dt = μN - δM - μM\n  dS/dt = δM - βSI/N - μS\n  dE/dt = βSI/N - σE - μE\n  dI/dt = σE - γI - μI\n  dR/dt = γI - μR",
                "params": "β — скорость передачи инфекции\nγ — скорость выздоровления\nμ — естественная смертность/рождаемость\nδ — скорость потери материнского иммунитета\nσ — скорость перехода в инфекционную фазу",
                "recommended": "β ≈ 0.3\nγ ≈ 0.1\nμ ≈ 0.05\nδ ≈ 0.02\nσ ≈ 0.2\n\nM₀ ≈ 0.1\nS₀ ≈ 0.99\nE₀ ≈ 0.01\nI₀ ≈ 0.02\nR₀ ≈ 0.00",
                "usage": "Применяется для долгосрочного анализа в популяциях с рождениями (например, детские инфекции)"
            },
            "M": {
                "desc": "M-модель описывает инфекцию с 3 стадиями развития.\n\nСистема дифференциальных уравнений:\n  dS/dt = -βSI₁ + γR\n  dI₁/dt = βSI₁ - k₁I₁\n  dI₂/dt = k₁I₁ - k₂I₂\n  dI₃/dt = k₂I₂ - k₃I₃\n  dR/dt = k₃I₃ - γR",
                "params": "β — скорость передачи инфекции\nk₁, k₂, k₃ — скорости переходов между стадиями\nγ — скорость потери иммунитета",
                "recommended": "β ≈ 0.5\nk₁ ≈ 0.3\nk₂ ≈ 0.2\nk₃ ≈ 0.1\nγ ≈ 0.05\n\nS₀ ≈ 0.9\nI₁₀ ≈ 0.1\nI₂₀ ≈ 0.0\nI₃₀ ≈ 0.0\nR₀ ≈ 0.0",
                "usage": "Подходит для инфекций с несколькими стадиями развития (например, ВИЧ, туберкулез)"
            }
        }

        for code, info in docs.items():
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=code)

            text = tk.Text(frame, wrap=tk.WORD, font=("Arial", 10))
            text.insert(tk.END, f"Описание модели:\n{info['desc']}\n\n"
                                f"Параметры:\n{info['params']}\n\n"
                                f"Рекомендуемые значения:\n{info['recommended']}\n\n"
                                f"Применение:\n{info['usage']}")
            text.configure(state='disabled')
            text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = EpidemicApp(root)
    root.mainloop()