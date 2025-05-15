import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkcalendar import DateEntry
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import os
import zipfile
from io import BytesIO
import re

class NumericalMethods:
    def __init__(self):
        self.result = None

    def euler_method(self, model_func, y0, t, args):
        """Реализация метода Эйлера для решения СДУ"""
        y = np.zeros((len(t), len(y0)))
        y[0] = y0
        
        for i in range(1, len(t)):
            dt = t[i] - t[i-1]
            dy = model_func(y[i-1], t[i-1], *args)
            y[i] = y[i-1] + dy * dt
        self.result = y.T
    
    def runge_kutta_4(self, model_func, y0, t, args):
        """Реализация метода Рунге-Кутты 4-го порядка для решения СДУ"""
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
    
    def run_si_model(self, t, params, initials, plot_index, return_solution=False, method="runge_kutta"):
        """Запускает SI модель на указанном графике"""
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
        self.canvases[plot_index].draw()
        
        if return_solution:
            return {"S": S, "I": I}

    def run_sir_model(self, t, params, initials, plot_index, return_solution=False, method="runge_kutta"):
        """Запускает SIR модель на указанном графике"""
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
        self.canvases[plot_index].draw()

        if return_solution:
            return {"S": S, "I": I, "R": R}

    def run_sirs_model(self, t, params, initials, plot_index, return_solution=False, method="runge_kutta"):
        """Запускает SIRS модель на указанном графике"""
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
        self.canvases[plot_index].draw()

        if return_solution:
            return {"S": S, "I": I, "R": R}

    def run_seir_model(self, t, params, initials, plot_index, return_solution=False, method="runge_kutta"):
        """Запускает SEIR модель на указанном графике"""
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
        self.canvases[plot_index].draw()

        if return_solution:
            return {"S": S, "E": E, "I": I, "R": R}

    def run_mseir_model(self, t, params, initials, plot_index, return_solution=False, method="runge_kutta"):
        """Запускает MSEIR модель на указанном графике"""
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
        self.canvases[plot_index].draw()

        if return_solution:
            return {"M": M, "S": S, "E": E, "I": I, "R": R}

    def run_siqr_model(self, t, params, initials, plot_index, return_solution=False, method="runge_kutta"):
        """Запускает SIQR модель на указанном графике"""
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
        self.canvases[plot_index].draw()

        if return_solution:
            return {"S": S, "I": I, "Q": Q, "R": R}

class EpidemicApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Эпидемиологическое моделирование")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Настройка стиля
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        self.style.configure('TButton', font=('Arial', 10))
        self.style.configure('TCheckbutton', background='#f0f0f0', font=('Arial', 10))
        self.style.configure('TRadiobutton', background='#f0f0f0', font=('Arial', 10))
        
        self.model = EpidemicModels()
        self.result_data = {}

        self.create_widgets()
        self.set_default_values()


    def create_widgets(self):
        """создание основного интерфейса"""
        # Главный контейнер с разделением на левую и правую части
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)

        # Левая панель (управление)
        control_frame = ttk.Frame(main_paned, width=350, relief=tk.RIDGE, padding=10)
        main_paned.add(control_frame, weight=0)

        # Правая панель (графики)
        graph_frame = ttk.Frame(main_paned)
        main_paned.add(graph_frame, weight=1)

        # Создаем Notebook для вкладок в левой панели
        control_notebook = ttk.Notebook(control_frame)
        control_notebook.pack(fill=tk.BOTH, expand=True)

        # Вкладка 1: Выбор моделей
        models_tab = ttk.Frame(control_notebook)
        control_notebook.add(models_tab, text="Модели")
        self.create_models_tab(models_tab)

        # Вкладка 2: Параметры
        params_tab = ttk.Frame(control_notebook)
        control_notebook.add(params_tab, text="Параметры")
        self.create_params_tab(params_tab)

        # Вкладка 3: Данные
        data_tab = ttk.Frame(control_notebook)
        control_notebook.add(data_tab, text="Данные")
        self.create_data_tab(data_tab)

        # Создаем графики в правой панели
        self.create_graphs(graph_frame)
    
    def create_models_tab(self, parent):
        """вкладка выбора моделей"""
        # Группа выбора моделей
        models_group = ttk.LabelFrame(parent, text="Выберите модели (макс. 4)", padding=10)
        models_group.pack(fill=tk.BOTH, pady=5)

        self.model_vars = {}
        models = [
            ('SI', 'Модель SI (восприимчивые-инфицированные)'),
            ('SIR', 'Модель SIR (восприимчивые-инфицированные-выздоровевшие)'),
            ('SIRS', 'Модель SIRS (с временным иммунитетом)'),
            ('SEIR', 'Модель SEIR (с латентным периодом)'),
            ('SIQR', 'Модель SIQR (с изоляцией)'),
            ('MSEIR', 'Модель MSEIR (с материнским иммунитетом)')
        ]
        
        for model_code, model_desc in models:
            var = tk.BooleanVar()
            cb = ttk.Checkbutton(models_group, text=model_desc, variable=var, 
                               command=self.update_model_selection)
            cb.pack(anchor='w', padx=5, pady=2)
            self.model_vars[model_code] = var

        # Группа метода решения
        method_group = ttk.LabelFrame(parent, text="Метод решения", padding=10)
        method_group.pack(fill=tk.BOTH, pady=5)
        
        self.method_var = tk.StringVar(value="runge_kutta")
        ttk.Radiobutton(method_group, text="Рунге-Кутта 4-го порядка", 
                       variable=self.method_var, value="runge_kutta").pack(anchor='w', padx=5, pady=2)
        ttk.Radiobutton(method_group, text="Метод Эйлера", 
                       variable=self.method_var, value="euler").pack(anchor='w', padx=5, pady=2)

        # Кнопка запуска
        ttk.Button(parent, text="Запустить моделирование", 
                  command=self.run_simulation).pack(fill=tk.X, pady=10)
        
    def create_params_tab(self, parent):
        """вкладка параметров и начальных условий"""
        # Фрейм с прокруткой
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

        # Параметры модели
        params_group = ttk.LabelFrame(scrollable_frame, text="Параметры модели", padding=10)
        params_group.pack(fill=tk.X, pady=5)
        
        self.param_entries = {}
        params = [
            ("beta", "β (скорость заражения)", 0.0, 1.0),
            ("gamma", "γ (скорость выздоровления)", 0.0, 1.0),
            ("delta", "δ (потеря иммунитета)", 0.0, 1.0),
            ("sigma", "σ (переход в инфекционные)", 0.0, 1.0),
            ("mu", "μ (выход из изоляции)", 0.0, 1.0)
        ]
        
        for param_code, param_desc, min_val, max_val in params:
            row = ttk.Frame(params_group)
            row.pack(fill=tk.X, pady=2)
            
            ttk.Label(row, text=param_desc, width=25).pack(side=tk.LEFT)
            entry = ttk.Entry(row, width=8)
            entry.pack(side=tk.RIGHT)
            self.param_entries[param_code] = entry

        # Начальные значения
        initials_group = ttk.LabelFrame(scrollable_frame, text="Начальные значения (доля)", padding=10)
        initials_group.pack(fill=tk.X, pady=5)
        
        self.init_entries = {}
        initials = [
            ("S0", "S₀ (восприимчивые)", 0.0, 1.0),
            ("I0", "I₀ (инфицированные)", 0.0, 1.0),
            ("R0", "R₀ (выздоровевшие)", 0.0, 1.0),
            ("E0", "E₀ (латентные)", 0.0, 1.0),
            ("Q0", "Q₀ (изолированные)", 0.0, 1.0),
            ("M0", "M₀ (материнский иммунитет)", 0.0, 1.0)
        ]
        
        for init_code, init_desc, min_val, max_val in initials:
            row = ttk.Frame(initials_group)
            row.pack(fill=tk.X, pady=2)
            
            ttk.Label(row, text=init_desc, width=25).pack(side=tk.LEFT)
            entry = ttk.Entry(row, width=8)
            entry.pack(side=tk.RIGHT)
            self.init_entries[init_code] = entry

        # Временные границы
        time_group = ttk.LabelFrame(scrollable_frame, text="Временной диапазон", padding=10)
        time_group.pack(fill=tk.X, pady=5)
        
        ttk.Label(time_group, text="Начальная дата:").pack(anchor='w', pady=(0, 5))
        self.start_entry = DateEntry(time_group, date_pattern='dd.mm.yyyy')
        self.start_entry.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(time_group, text="Конечная дата:").pack(anchor='w', pady=(0, 5))
        self.end_entry = DateEntry(time_group, date_pattern='dd.mm.yyyy')
        self.end_entry.pack(fill=tk.X)

    def create_data_tab(self, parent):
        """вкладка для загрузки и экспорта данных"""
        # Группа загрузки данных
        load_group = ttk.LabelFrame(parent, text="Загрузка данных", padding=10)
        load_group.pack(fill=tk.BOTH, pady=5, expand=True)
        
        ttk.Button(load_group, text="Загрузить данные из CSV", 
                  command=self.load_csv_data).pack(fill=tk.X, pady=5)

        # Группа экспорта результатов
        export_group = ttk.LabelFrame(parent, text="Экспорт результатов", padding=10)
        export_group.pack(fill=tk.BOTH, pady=5)
        
        ttk.Button(export_group, text="Экспорт в Excel и ZIP", 
                  command=self.export_results).pack(fill=tk.X, pady=5)
        
    def create_graphs(self, parent):
        """настройка графиков для отображения результатов"""
        # Создаем 4 графика в сетке 2x2
        self.model.axes = []
        self.model.canvases = []
        self.model.figs = []
        
        for i in range(4):
            fig = plt.Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)
            
            # Настройка внешнего вида графика
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_facecolor('#f8f8f8')
            
            # Создаем холст для графика
            canvas = FigureCanvasTkAgg(fig, master=parent)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.grid(row=i//2, column=i%2, padx=5, pady=5, sticky='nsew')
            canvas_widget.config(borderwidth=2, relief=tk.GROOVE)
            
            # Настройка растягивания
            parent.grid_rowconfigure(i//2, weight=1)
            parent.grid_columnconfigure(i%2, weight=1)
            
            self.model.figs.append(fig)
            self.model.axes.append(ax)
            self.model.canvases.append(canvas)

    def create_validate_func(self, min_val, max_val):
        """создание функции валидации ввода"""
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
        """установка значений по умолчанию"""
        # Установка значений по умолчанию для параметров
        self.param_entries["beta"].insert(0, "0.3")
        self.param_entries["gamma"].insert(0, "0.1")
        self.param_entries["delta"].insert(0, "0.01")
        self.param_entries["sigma"].insert(0, "0.2")
        self.param_entries["mu"].insert(0, "0.05")
        
        # Установка значений по умолчанию для начальных условий
        self.init_entries["S0"].insert(0, "0.99")
        self.init_entries["I0"].insert(0, "0.01")
        self.init_entries["R0"].insert(0, "0.0")
        self.init_entries["E0"].insert(0, "0.0")
        self.init_entries["Q0"].insert(0, "0.0")
        self.init_entries["M0"].insert(0, "0.0")
        
        # Установка дат по умолчанию
        today = datetime.now()
        self.start_entry.set_date(today)
        self.end_entry.set_date(today + timedelta(days=100))
    
    def update_model_selection(self):
        """ограничение выбора моделей до 4"""
        # Подсчет выбранных моделей
        selected = sum(var.get() for var in self.model_vars.values())
        
        # Если выбрано больше 4 моделей, снимаем последний выбор
        if selected > 4:
            for model_code, var in reversed(self.model_vars.items()):
                if var.get():
                    var.set(False)
                    messagebox.showwarning("Предупреждение", "Можно выбрать не более 4 моделей одновременно")
                    break
    
    def run_simulation(self):
        """запуск моделирования"""
        # Проверка, что выбрана хотя бы одна модель
        selected_models = [name for name, var in self.model_vars.items() if var.get()]
        if not selected_models:
            messagebox.showwarning("Ошибка", "Выберите хотя бы одну модель")
            return
        
        # Проверка корректности ввода параметров
        try:
            params = {k: float(e.get()) if e.get() else 0.0 for k, e in self.param_entries.items()}
            initials = {k: float(e.get()) if e.get() else 0.0 for k, e in self.init_entries.items()}
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректные значения параметров")
            return
        
        # Проверка дат
        if self.end_entry.get_date() <= self.start_entry.get_date():
            messagebox.showerror("Ошибка", "Конечная дата должна быть позже начальной")
            return
        
        self.result_data.clear()
        days = (self.end_entry.get_date() - self.start_entry.get_date()).days
        t = np.linspace(0, days, days + 1)
        method = self.method_var.get()
        
        # Очистка графиков перед новым моделированием
        for ax in self.model.axes:
            ax.clear()
        
        # Запуск выбранных моделей
        for i, model_code in enumerate(selected_models):
            if i >= 4:  # Не более 4 моделей
                break
                
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
            
            if sol:
                self.result_data[model_code] = pd.DataFrame(sol, index=t)
        
        # Обновление всех графиков
        for canvas in self.model.canvases:
            canvas.draw()
    
    def load_csv_data(self):
        """загрузка данных из CSV"""
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        
        try:
            df = pd.read_csv(path)
            df["Date"] = pd.to_datetime(df["Date"])
            df.sort_values("Date", inplace=True)
            self.csv_data = df

            # Получаем уникальные страны
            countries = sorted(df["Country/Region"].dropna().unique())
            
            # Создаем окно выбора страны
            self.country_selection = tk.Toplevel(self.root)
            self.country_selection.title("Выбор страны и дат")
            self.country_selection.geometry("400x300")

            # Выбор страны
            ttk.Label(self.country_selection, text="Выберите страну:").pack(pady=(10, 5))
            self.country_cb = ttk.Combobox(self.country_selection, values=countries, state="readonly")
            self.country_cb.pack(pady=5)

            # Выбор диапазона дат для начальных данных
            ttk.Label(self.country_selection, text="Выберите диапазон дат для начальных данных:").pack(pady=(10, 5))
            
            date_frame = ttk.Frame(self.country_selection)
            date_frame.pack(pady=5)
            
            ttk.Label(date_frame, text="С:").pack(side=tk.LEFT)
            self.data_start_entry = DateEntry(date_frame, date_pattern='dd.mm.yyyy')
            self.data_start_entry.pack(side=tk.LEFT, padx=5)
            
            ttk.Label(date_frame, text="По:").pack(side=tk.LEFT)
            self.data_end_entry = DateEntry(date_frame, date_pattern='dd.mm.yyyy')
            self.data_end_entry.pack(side=tk.LEFT, padx=5)

            # Установка минимальной и максимальной даты из данных
            min_date = df["Date"].min().to_pydatetime()
            max_date = df["Date"].max().to_pydatetime()
            self.data_start_entry.set_date(min_date)
            self.data_end_entry.set_date(max_date)

            # Кнопка загрузки
            ttk.Button(self.country_selection, text="Загрузить данные", 
                      command=self.process_csv_data).pack(pady=10)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить файл: {str(e)}")
    
    def process_csv_data(self):
        """обработка загруженных данных"""
        country = self.country_cb.get()
        if not country:
            messagebox.showwarning("Ошибка", "Выберите страну")
            return
        
        try:
            # Получаем выбранные даты
            start_date = self.data_start_entry.get_date()
            end_date = self.data_end_entry.get_date()
            
            if start_date > end_date:
                messagebox.showerror("Ошибка", "Начальная дата не может быть позже конечной")
                return

            # Фильтруем данные по стране и дате
            df_country = self.csv_data[
                (self.csv_data["Country/Region"] == country) &
                (self.csv_data["Date"] >= pd.to_datetime(start_date)) &
                (self.csv_data["Date"] <= pd.to_datetime(end_date))
            ].copy()
            
            if df_country.empty:
                messagebox.showerror("Ошибка", "Нет данных для выбранного диапазона дат")
                return

            # Берем последние значения в выбранном диапазоне как начальные
            latest = df_country.iloc[-1]
            total = latest["Confirmed"] + latest["Recovered"] + latest["Deaths"]
            if total == 0:
                total = 1  # во избежание деления на 0
            
            S0 = 1 - (latest["Confirmed"] + latest["Recovered"] + latest["Deaths"]) / total
            I0 = latest["Confirmed"] / total
            R0 = latest["Recovered"] / total

            # Обновление начальных значений в основном интерфейсе
            for entry in self.init_entries.values():
                entry.delete(0, tk.END)
                
            self.init_entries["S0"].insert(0, f"{S0:.4f}")
            self.init_entries["I0"].insert(0, f"{I0:.4f}")
            self.init_entries["R0"].insert(0, f"{R0:.4f}")
            
            # Устанавливаем даты моделирования (по умолчанию продолжаем после выбранного диапазона)
            self.start_entry.set_date(end_date)
            self.end_entry.set_date(end_date + timedelta(days=100))
            
            self.country_selection.destroy()
            messagebox.showinfo("Успех", f"Данные для {country} за период {start_date.strftime('%d.%m.%Y')}-{end_date.strftime('%d.%m.%Y')} успешно загружены")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось обработать данные: {str(e)}")
    
    def export_results(self):
        """экспорт результатов в ZIP-архив"""
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
                        # Лист 1: Начальные данные
                        initials_df = pd.DataFrame({
                            'Параметр': ['S₀ (восприимчивые)', 'I₀ (инфицированные)', 'R₀ (выздоровевшие)',
                                         'E₀ (латентные)', 'Q₀ (изолированные)', 'M₀ (материнский иммунитет)'],
                            'Значение': [self.init_entries["S0"].get(), self.init_entries["I0"].get(),
                                        self.init_entries["R0"].get(), self.init_entries["E0"].get(),
                                        self.init_entries["Q0"].get(), self.init_entries["M0"].get()]
                        })
                        initials_df.to_excel(writer, sheet_name='Начальные данные', index=False)
                        
                        # Лист 2: Параметры модели
                        params_df = pd.DataFrame({
                            'Параметр': ['β (скорость заражения)', 'γ (скорость выздоровления)',
                                        'δ (потеря иммунитета)', 'σ (переход в инфекционные)',
                                        'μ (выход из изоляции)'],
                            'Значение': [self.param_entries["beta"].get(), self.param_entries["gamma"].get(),
                                        self.param_entries["delta"].get(), self.param_entries["sigma"].get(),
                                        self.param_entries["mu"].get()]
                        })
                        params_df.to_excel(writer, sheet_name='Параметры', index=False)
                        
                        # Лист 3: Решение
                        # Создаем DataFrame с информацией о методе
                        method_info = pd.DataFrame({
                            'Информация': ['Метод решения:', 'Выбранная модель:'],
                            'Значение': [self.method_var.get().replace("runge_kutta", "Рунге-Кутта 4-го порядка").replace("euler", "Метод Эйлера"), 
                                        model_name]
                        })
                        
                        # Записываем информацию о методе
                        method_info.to_excel(writer, sheet_name='Решение', startrow=0, index=False, header=False)
                        
                        # Записываем основные данные с отступом в 2 строки
                        df.to_excel(writer, sheet_name='Решение', startrow=3, index=True)
                        
                        # Получаем объект worksheet для дополнительного форматирования
                        worksheet = writer.sheets['Решение']
                        
                        # Добавляем подпись к индексу
                        worksheet.write(3, 0, 'Дни')
                        
                        # Лист 4: График
                        graph_sheet = writer.book.add_worksheet('График')
                        
                        # Создаем график
                        chart = writer.book.add_chart({'type': 'line'})
                        
                        max_row = len(df) + 4  # Учитываем смещение из-за заголовков
                        categories = f"='Решение'!$A$5:$A${max_row}"
                        
                        for i, col in enumerate(df.columns, 1):
                            chart.add_series({
                                'name': f"='Решение'!${chr(66+i)}$4",
                                'categories': categories,
                                'values': f"='Решение'!${chr(66+i)}$5:${chr(66+i)}${max_row}",
                            })
                        
                        chart.set_x_axis({'name': 'Дни'})
                        chart.set_y_axis({'name': 'Доля населения'})
                        chart.set_title({'name': f'Модель {model_name} ({method_info.iloc[0,1]})'})
                        
                        # Вставляем график на лист График
                        graph_sheet.insert_chart('B2', chart, {'x_scale': 2, 'y_scale': 2})
                    
                    zf.writestr(f"{model_name}.xlsx", excel_buffer.getvalue())

            with open(save_path, "wb") as f:
                f.write(mem_zip.getvalue())
            
            messagebox.showinfo("Экспорт завершён", f"Файлы успешно сохранены в архив:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось экспортировать данные: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = EpidemicApp(root)
    root.mainloop()