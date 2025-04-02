import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import DateEntry
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class EpidemicModels:
    def __init__(self):
        self.current_models = []

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
    
    def multi_stage_model(self, t, y, beta, k, gamma):
        S, *I, R = y
        dS_dt = -beta * S * I[0] + gamma * R
        dI_dt = [beta * S * I[0] - k[0] * I[0]]
        for j in range(1, len(I)):
            dI_dt.append(k[j-1] * I[j-1] - k[j] * I[j])
        dR_dt = k[-1] * I[-1] - gamma * R
        return np.array([dS_dt] + dI_dt + [dR_dt])

class EpidemicModelsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Моделирование эпидемий")
        self.root.geometry("1400x800")
        
        self.models_obj = EpidemicModels()
        self.available_models = {
            "SIR": {"name": "SIR-модель", "func": self.models_obj.sir_model, "params": self.get_model_parameters("sir_model")},
            "SI": {"name": "SI-модель", "func": self.models_obj.si_model, "params": self.get_model_parameters("si_model")},
            "SIRS": {"name": "SIRS-модель", "func": self.models_obj.sirs_model, "params": self.get_model_parameters("sirs_model")},
            "SEIR": {"name": "SEIR-модель", "func": self.models_obj.seir_model, "params": self.get_model_parameters("seir_model")},
            "MSEIR": {"name": "MSEIR-модель", "func": self.models_obj.mseir_model, "params": self.get_model_parameters("mseir_model")},
            "SIQR": {"name": "SIQR-модель", "func": self.models_obj.siqr_model, "params": self.get_model_parameters("siqr")},
            "M-модель": {"name": "M-модель", "func": self.models_obj.multi_stage_model, "params": self.get_model_parameters("m_model")}
        }
        
        self.selected_models = []
        self.model_widgets = []
        self.model_vars = []
        
        self.create_left_panel()
        self.create_right_panel()
        
        # Добавляем первую модель по умолчанию
        self.add_model_field()
        
    def get_model_parameters(self, model_name):
        """Возвращает параметры для конкретной модели"""
        if model_name == "si_model":
            return {
                "beta": {"name": "Коэффициент заражения (beta)", "default": 0.3, "min": 0, "max": 1, "step": 0.01},
                "initial": {
                    "S0": {"name": "Восприимчивые (S0)", "default": 0.99, "min": 0, "max": 1, "step": 0.01},
                    "I0": {"name": "Инфицированные (I0)", "default": 0.01, "min": 0, "max": 1, "step": 0.01}
                }
            }
        elif model_name == "sir_model":
            return {
                "beta": {"name": "Коэффициент заражения (beta)", "default": 0.3, "min": 0, "max": 1, "step": 0.01},
                "gamma": {"name": "Коэффициент выздоровления (gamma)", "default": 0.1, "min": 0, "max": 1, "step": 0.01},
                "initial": {
                    "S0": {"name": "Восприимчивые (S0)", "default": 0.99, "min": 0, "max": 1, "step": 0.01},
                    "I0": {"name": "Инфицированные (I0)", "default": 0.01, "min": 0, "max": 1, "step": 0.01},
                    "R0": {"name": "Выздоровевшие (R0)", "default": 0.0, "min": 0, "max": 1, "step": 0.01}
                }
            }
        elif model_name == "sirs_model":
            return {
                "beta": {"name": "Коэффициент заражения (beta)", "default": 0.3, "min": 0, "max": 1, "step": 0.01},
                "gamma": {"name": "Коэффициент выздоровления (gamma)", "default": 0.1, "min": 0, "max": 1, "step": 0.01},
                "delta": {"name": "Потеря иммунитета (delta)", "default": 0.05, "min": 0, "max": 1, "step": 0.01},
                "initial": {
                    "S0": {"name": "Восприимчивые (S0)", "default": 0.99, "min": 0, "max": 1, "step": 0.01},
                    "I0": {"name": "Инфицированные (I0)", "default": 0.01, "min": 0, "max": 1, "step": 0.01},
                    "R0": {"name": "Выздоровевшие (R0)", "default": 0.0, "min": 0, "max": 1, "step": 0.01}
                }
            }
        elif model_name == "seir_model":
            return {
                "beta": {"name": "Коэффициент заражения (beta)", "default": 0.3, "min": 0, "max": 1, "step": 0.01},
                "sigma": {"name": "Инкубационный период (sigma)", "default": 1/5.1, "min": 0, "max": 1, "step": 0.01},
                "gamma": {"name": "Коэффициент выздоровления (gamma)", "default": 0.1, "min": 0, "max": 1, "step": 0.01},
                "initial": {
                    "S0": {"name": "Восприимчивые (S0)", "default": 0.99, "min": 0, "max": 1, "step": 0.01},
                    "E0": {"name": "Латентные (E0)", "default": 0.0, "min": 0, "max": 1, "step": 0.01},
                    "I0": {"name": "Инфицированные (I0)", "default": 0.01, "min": 0, "max": 1, "step": 0.01},
                    "R0": {"name": "Выздоровевшие (R0)", "default": 0.0, "min": 0, "max": 1, "step": 0.01}
                }
            }
        elif model_name == "mseir_model":
            return {
                "mu": {"name": "Смертность/рождаемость (mu)", "default": 0.01, "min": 0, "max": 0.1, "step": 0.001},
                "delta": {"name": "Потеря антител (delta)", "default": 0.1, "min": 0, "max": 1, "step": 0.01},
                "beta": {"name": "Коэффициент заражения (beta)", "default": 0.5, "min": 0, "max": 1, "step": 0.01},
                "sigma": {"name": "Инкубационный период (sigma)", "default": 1/5, "min": 0, "max": 1, "step": 0.01},
                "gamma": {"name": "Коэффициент выздоровления (gamma)", "default": 1/7, "min": 0, "max": 1, "step": 0.01},
                "initial": {
                    "M0": {"name": "Материнский иммунитет (M0)", "default": 0.3, "min": 0, "max": 1, "step": 0.01},
                    "S0": {"name": "Восприимчивые (S0)", "default": 0.8, "min": 0, "max": 1, "step": 0.01},
                    "E0": {"name": "Латентные (E0)", "default": 0.01, "min": 0, "max": 1, "step": 0.01},
                    "I0": {"name": "Инфицированные (I0)", "default": 0.05, "min": 0, "max": 1, "step": 0.01},
                    "R0": {"name": "Выздоровевшие (R0)", "default": 0.04, "min": 0, "max": 1, "step": 0.01}
                }
            }
        elif model_name == "siqr":
            return {
                "beta": {"name": "Коэффициент заражения (beta)", "default": 0.3, "min": 0, "max": 1, "step": 0.01},
                "gamma": {"name": "Выздоровление инфицированных (gamma)", "default": 0.1, "min": 0, "max": 1, "step": 0.01},
                "delta": {"name": "Изоляция инфицированных (delta)", "default": 0.05, "min": 0, "max": 1, "step": 0.01},
                "mu": {"name": "Выздоровление изолированных (mu)", "default": 0.05, "min": 0, "max": 1, "step": 0.01},
                "initial": {
                    "S0": {"name": "Восприимчивые (S0)", "default": 0.99, "min": 0, "max": 1, "step": 0.01},
                    "I0": {"name": "Инфицированные (I0)", "default": 0.01, "min": 0, "max": 1, "step": 0.01},
                    "Q0": {"name": "Изолированные (Q0)", "default": 0.0, "min": 0, "max": 1, "step": 0.01},
                    "R0": {"name": "Выздоровевшие (R0)", "default": 0.0, "min": 0, "max": 1, "step": 0.01}
                }
            }
        elif model_name == "m_model":
            return {
                "beta": {"name": "Коэффициент заражения (beta)", "default": 0.5, "min": 0, "max": 1, "step": 0.01},
                "k": {"name": "Скорости переходов (k)", "default": [0.3, 0.2, 0.1], "min": 0, "max": 1, "step": 0.01},
                "gamma": {"name": "Потеря иммунитета (gamma)", "default": 0.05, "min": 0, "max": 1, "step": 0.01},
                "initial": {
                    "S0": {"name": "Восприимчивые (S0)", "default": 0.9, "min": 0, "max": 1, "step": 0.01},
                    "I0": {"name": "Инфицированные (I0)", "default": [0.1, 0.0, 0.0], "min": 0, "max": 1, "step": 0.01},
                    "R0": {"name": "Выздоровевшие (R0)", "default": 0.0, "min": 0, "max": 1, "step": 0.01}
                }
            }
        return {}

    def create_left_panel(self):
        """Создает левую панель с выбором моделей и параметрами"""
        left_frame = ttk.Frame(self.root, padding=10, width=400)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5)
        left_frame.pack_propagate(False)
            
        # Выбор моделей
        model_frame = ttk.LabelFrame(left_frame, text="Выбор моделей (до 4)", padding=10)
        model_frame.pack(fill=tk.X, pady=5)
        
        # Контейнер для полей выбора моделей
        self.models_container = ttk.Frame(model_frame)
        self.models_container.pack(fill=tk.X)
        
        # Кнопка добавления новой модели
        self.add_button = ttk.Button(model_frame, text="+ Добавить модель", 
                                   command=self.add_model_field, state=tk.NORMAL)
        self.add_button.pack(pady=5)
        
        # Диапазон дат
        date_frame = ttk.LabelFrame(left_frame, text="Диапазон дат", padding=10)
        date_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(date_frame, text="Начальная дата:").grid(row=0, column=0, sticky="w")
        self.start_date_entry = DateEntry(
            date_frame,
            width=12,
            background='darkblue',
            foreground='white',
            borderwidth=2,
            date_pattern='dd.mm.yyyy'
        )
        self.start_date_entry.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        self.start_date_entry.set_date(datetime.now() - timedelta(days=30))
        
        ttk.Label(date_frame, text="Конечная дата:").grid(row=1, column=0, sticky="w")
        self.end_date_entry = DateEntry(
            date_frame,
            width=12,
            background='darkblue',
            foreground='white',
            borderwidth=2,
            date_pattern='dd.mm.yyyy'
        )
        self.end_date_entry.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        self.end_date_entry.set_date(datetime.now())
        
        # Кнопка запуска
        ttk.Button(left_frame, text="Запустить моделирование", 
                  command=self.run_models).pack(pady=10)
        
        # Панель параметров
        self.params_notebook = ttk.Notebook(left_frame)
        self.params_notebook.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def create_right_panel(self):
        """Создает правую панель с графиками"""
        right_frame = ttk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Создаем 4 графических фрейма, но будем показывать только нужные
        self.plot_frames = []
        self.figs = []
        self.axes = []
        self.canvases = []
        
        for i in range(4):
            frame = ttk.LabelFrame(right_frame, text=f"График {i+1}", padding=5)
            frame.grid(row=i//2, column=i%2, sticky="nsew", padx=5, pady=5)
            right_frame.rowconfigure(i//2, weight=1)
            right_frame.columnconfigure(i%2, weight=1)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.set_xlabel('Дни')
            ax.set_ylabel('Доля населения')
            ax.grid(True)
            
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self.plot_frames.append(frame)
            self.figs.append(fig)
            self.axes.append(ax)
            self.canvases.append(canvas)
            
            # Сначала скрываем все графики
            frame.grid_remove()
    
    def add_model_field(self):
        """Добавляет новое поле выбора модели"""
        if len(self.model_widgets) >= 4:
            messagebox.showwarning("Предупреждение", "Можно добавить не более 4 моделей")
            self.add_button.config(state=tk.DISABLED)
            return
        
        frame = ttk.Frame(self.models_container)
        frame.pack(fill=tk.X, pady=2)
        
        var = tk.StringVar()
        cb = ttk.Combobox(frame, textvariable=var, 
                        values=[""] + list(self.available_models.keys()),
                        state="readonly")
        cb.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        btn = ttk.Button(frame, text="×", width=2, 
                       command=lambda: self.remove_model_field(frame, var))
        btn.pack(side=tk.RIGHT)
        
        # Привязываем обработчик выбора модели
        cb.bind("<<ComboboxSelected>>", lambda e, v=var: self.model_selected(v))
        
        self.model_vars.append(var)
        self.model_widgets.append({"frame": frame, "combobox": cb, "button": btn})
        
        # Обновляем состояние кнопки добавления
        if len(self.model_widgets) >= 4:
            self.add_button.config(state=tk.DISABLED)
        else:
            self.add_button.config(state=tk.NORMAL)
    
    def remove_model_field(self, frame, var):
        """Удаляет поле выбора модели"""
        model_name = var.get()
        if model_name in self.selected_models:
            self.selected_models.remove(model_name)
            self.update_params_notebook()
        
        # Удаляем из списков
        for i, widget in enumerate(self.model_widgets):
            if widget["frame"] == frame:
                self.model_widgets.pop(i)
                self.model_vars.pop(i)
                break
        
        # Удаляем фрейм
        frame.destroy()
        
        # Обновляем состояние кнопки добавления
        if len(self.model_widgets) < 4:
            self.add_button.config(state=tk.NORMAL)
        
        # Обновляем графики
        self.update_plots_visibility()
    
    def model_selected(self, var):
        """Обработчик выбора модели"""
        model_name = var.get()
        if not model_name:
            return
        
        # Проверяем, не выбрана ли эта модель уже в другом поле
        for v in self.model_vars:
            if v != var and v.get() == model_name:
                messagebox.showwarning("Предупреждение", "Эта модель уже выбрана")
                var.set("")
                return
        
        if model_name not in self.selected_models:
            self.selected_models.append(model_name)
            self.update_params_notebook()
        
        # Обновляем видимость графиков
        self.update_plots_visibility()
    
    def update_plots_visibility(self):
        """Обновляет видимость графиков в зависимости от количества выбранных моделей"""
        num_models = len(self.selected_models)
        
        for i in range(4):
            if i < num_models:
                self.plot_frames[i].grid()
            else:
                self.plot_frames[i].grid_remove()
                self.axes[i].clear()
                self.canvases[i].draw()
    
    def update_params_notebook(self):
        """Обновляет блокнот с параметрами для выбранных моделей"""
        # Удаляем все существующие вкладки
        for tab in self.params_notebook.tabs():
            self.params_notebook.forget(tab)
        
        # Создаем новые вкладки для выбранных моделей
        for i, model_name in enumerate(self.selected_models):
            tab = ttk.Frame(self.params_notebook)
            self.params_notebook.add(tab, text=model_name)
            
            model_info = self.available_models[model_name]
            params = model_info["params"]
            
            # Создаем вкладки для параметров и начальных условий
            notebook = ttk.Notebook(tab)
            notebook.pack(fill=tk.BOTH, expand=True)
            
            # Вкладка с параметрами модели
            params_tab = ttk.Frame(notebook)
            notebook.add(params_tab, text="Параметры")
            
            # Вкладка с начальными условиями
            initial_tab = ttk.Frame(notebook)
            notebook.add(initial_tab, text="Начальные условия")
            
            # Сохраняем виджеты параметров
            if not hasattr(self, 'model_params'):
                self.model_params = {}
            self.model_params[model_name] = {
                "param_widgets": {},
                "initial_widgets": {}
            }
            
            # Заполняем вкладку параметров
            row = 0
            for param, info in params.items():
                if param == "initial":
                    continue
                    
                ttk.Label(params_tab, text=info["name"]).grid(row=row, column=0, sticky="w", padx=5, pady=2)
                
                if isinstance(info["default"], list):
                    # Для параметров, которые являются списками
                    frame = ttk.Frame(params_tab)
                    frame.grid(row=row, column=1, sticky="w")
                    
                    entries = []
                    for j, val in enumerate(info["default"]):
                        ttk.Label(frame, text=f"k{j+1}:").grid(row=0, column=j*2, padx=2)
                        entry = ttk.Entry(frame, width=6)
                        entry.insert(0, str(val))
                        entry.grid(row=0, column=j*2+1, padx=2)
                        entries.append(entry)
                    
                    self.model_params[model_name]["param_widgets"][param] = entries
                else:
                    # Для обычных параметров
                    spin = ttk.Spinbox(
                        params_tab,
                        from_=info["min"],
                        to=info["max"],
                        increment=info["step"],
                        width=8
                    )
                    spin.set(info["default"])
                    spin.grid(row=row, column=1, sticky="w", padx=5, pady=2)
                    self.model_params[model_name]["param_widgets"][param] = spin
                
                row += 1
            
            # Заполняем вкладку начальных условий
            row = 0
            for param, info in params["initial"].items():
                ttk.Label(initial_tab, text=info["name"]).grid(row=row, column=0, sticky="w", padx=5, pady=2)
                
                if isinstance(info["default"], list):
                    # Для начальных условий, которые являются списками
                    frame = ttk.Frame(initial_tab)
                    frame.grid(row=row, column=1, sticky="w")
                    
                    entries = []
                    for j, val in enumerate(info["default"]):
                        ttk.Label(frame, text=f"I{j+1}:").grid(row=0, column=j*2, padx=2)
                        entry = ttk.Entry(frame, width=6)
                        entry.insert(0, str(val))
                        entry.grid(row=0, column=j*2+1, padx=2)
                        entries.append(entry)
                    
                    self.model_params[model_name]["initial_widgets"][param] = entries
                else:
                    # Для обычных начальных условий
                    spin = ttk.Spinbox(
                        initial_tab,
                        from_=info["min"],
                        to=info["max"],
                        increment=info["step"],
                        width=8
                    )
                    spin.set(info["default"])
                    spin.grid(row=row, column=1, sticky="w", padx=5, pady=2)
                    self.model_params[model_name]["initial_widgets"][param] = spin
                
                row += 1
    
    def get_parameter_values(self, model_name):
        """Возвращает значения параметров для указанной модели"""
        params = {}
        if model_name in self.model_params:
            for param, widget in self.model_params[model_name]["param_widgets"].items():
                if isinstance(widget, list):
                    params[param] = [float(entry.get()) for entry in widget]
                else:
                    try:
                        params[param] = float(widget.get())
                    except ValueError:
                        params[param] = 0.0
        return params
    
    def get_initial_values(self, model_name):
        """Возвращает начальные условия для указанной модели"""
        initials = {}
        if model_name in self.model_params:
            for param, widget in self.model_params[model_name]["initial_widgets"].items():
                if isinstance(widget, list):
                    initials[param] = [float(entry.get()) for entry in widget]
                else:
                    try:
                        initials[param] = float(widget.get())
                    except ValueError:
                        initials[param] = 0.0
        return initials
    
    def euler_method(self, model_func, y0, t, args):
        """Реализация метода Эйлера для решения СДУ"""
        y = np.zeros((len(t), len(y0)))
        y[0] = y0
        
        for i in range(1, len(t)):
            dt = t[i] - t[i-1]
            dy = model_func(y[i-1], t[i-1], *args)
            y[i] = y[i-1] + dy * dt
        print(y.T)
        return y.T
    
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
        
        print(y)
        return y.T
    
    def run_models(self):
        """Запускает все выбранные модели"""
        try:
            # Получаем даты
            start_date = self.start_date_entry.get_date()
            end_date = self.end_date_entry.get_date()
            
            if end_date <= start_date:
                messagebox.showerror("Ошибка", "Конечная дата должна быть позже начальной")
                return
            
            delta = end_date - start_date
            t = np.linspace(0, delta.days, delta.days + 1)
            
            # Очищаем все графики
            for ax in self.axes:
                ax.clear()
            
            # Запускаем каждую выбранную модель
            for i, model_name in enumerate(self.selected_models):
                if i >= 4:  # Не больше 4 моделей
                    break
                
                # Получаем параметры для модели
                params = self.get_parameter_values(model_name)
                initials = self.get_initial_values(model_name)
                
                # Запускаем соответствующую модель
                if model_name == "SI":
                    self.run_si_model(t, params, initials, i)
                elif model_name == "SIR":
                    self.run_sir_model(t, params, initials, i)
                elif model_name == "SIRS":
                    self.run_sirs_model(t, params, initials, i)
                elif model_name == "SEIR":
                    self.run_seir_model(t, params, initials, i)
                elif model_name == "MSEIR":
                    self.run_mseir_model(t, params, initials, i)
                elif model_name == "SIQR":
                    self.run_siqr_model(t, params, initials, i)
                elif model_name == "M-модель":
                    self.run_m_model(t, params, initials, i)
                
                # Настраиваем график
                self.axes[i].set_title(self.available_models[model_name]["name"])
                self.axes[i].legend()
                self.canvases[i].draw()
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при выполнении моделирования: {str(e)}")
    
    # Модифицированные методы для моделей (используем метод Эйлера)
    def run_si_model(self, t, params, initials, plot_index):
        """Запускает SI модель на указанном графике"""
        y0 = [initials["S0"], initials["I0"]]
        solution = self.runge_kutta_4(self.models_obj.si_model, y0, t, (params["beta"],))
        S, I = solution
        
        ax = self.axes[plot_index]
        ax.plot(t, S, 'b', label='Восприимчивые')
        ax.plot(t, I, 'r', label='Инфицированные')
        ax.grid()
        ax.set_ylim(0, 1)
    
    def run_sir_model(self, t, params, initials, plot_index):
        """Запускает SIR модель на указанном графике"""
        y0 = [initials["S0"], initials["I0"], initials["R0"]]
        solution = self.runge_kutta_4(self.models_obj.sir_model, y0, t, (params["beta"], params["gamma"]))
        S, I, R = solution
        
        ax = self.axes[plot_index]
        ax.plot(t, S, 'b', label='Восприимчивые')
        ax.plot(t, I, 'r', label='Инфицированные')
        ax.plot(t, R, 'g', label='Выздоровевшие')
        ax.grid()
        ax.set_ylim(0, 1)
    
    def run_sirs_model(self, t, params, initials, plot_index):
        """Запускает SIRS модель на указанном графике"""
        y0 = [initials["S0"], initials["I0"], initials["R0"]]
        solution = self.runge_kutta_4(self.models_obj.sirs_model, y0, t, (params["beta"], params["gamma"], params["delta"]))
        S, I, R = solution
        
        ax = self.axes[plot_index]
        ax.plot(t, S, 'b', label='Восприимчивые')
        ax.plot(t, I, 'r', label='Инфицированные')
        ax.plot(t, R, 'g', label='Выздоровевшие')
        ax.grid()
        ax.set_ylim(0, 1)
    
    def run_seir_model(self, t, params, initials, plot_index):
        """Запускает SEIR модель на указанном графике"""
        y0 = [initials["S0"], initials["E0"], initials["I0"], initials["R0"]]
        solution = self.runge_kutta_4(self.models_obj.seir_model, y0, t, (params["beta"], params["sigma"], params["gamma"]))
        S, E, I, R = solution
        
        ax = self.axes[plot_index]
        ax.plot(t, S, 'b', label='Восприимчивые')
        ax.plot(t, E, 'y', label='Латентные')
        ax.plot(t, I, 'r', label='Инфицированные')
        ax.plot(t, R, 'g', label='Выздоровевшие')
        ax.grid()
        ax.set_ylim(0, 1)
    
    def run_mseir_model(self, t, params, initials, plot_index):
        """Запускает MSEIR модель на указанном графике"""
        y0 = [initials["M0"], initials["S0"], initials["E0"], initials["I0"], initials["R0"]]
        solution = self.runge_kutta_4(self.models_obj.mseir_model, y0, t, 
                                   (params["mu"], params["delta"], params["beta"], params["sigma"], params["gamma"]))
        M, S, E, I, R = solution
        
        ax = self.axes[plot_index]
        ax.plot(t, M, 'b', label='Материнский иммунитет')
        ax.plot(t, S, 'g', label='Восприимчивые')
        ax.plot(t, E, 'y', label='Латентные')
        ax.plot(t, I, 'r', label='Инфицированные')
        ax.plot(t, R, 'purple', label='Выздоровевшие')
        ax.grid()
        ax.set_ylim(0, 1)
    
    def run_siqr_model(self, t, params, initials, plot_index):
        """Запускает SIQR модель на указанном графике"""
        y0 = [initials["S0"], initials["I0"], initials["Q0"], initials["R0"]]
        solution = self.runge_kutta_4(self.models_obj.siqr_model, y0, t, 
                                   (params["beta"], params["gamma"], params["delta"], params["mu"]))
        S, I, Q, R = solution
        
        ax = self.axes[plot_index]
        ax.plot(t, S, 'b', label='Восприимчивые')
        ax.plot(t, I, 'r', label='Инфицированные')
        ax.plot(t, Q, 'g', label='Изолированные')
        ax.plot(t, R, 'k', label='Выздоровевшие')
        ax.grid()
        ax.set_ylim(0, 1)
    
    def run_m_model(self, t, params, initials, plot_index):
        """Запускает M-модель на указанном графике"""
        y0 = [initials["S0"]] + initials["I0"] + [initials["R0"]]
        solution = self.runge_kutta_4(self.models_obj.multi_stage_model, y0, t, 
                                   (params["beta"], params["k"], params["gamma"]))
        
        ax = self.axes[plot_index]
        ax.plot(t, solution[0], 'b', label='Восприимчивые')
        for j in range(len(initials["I0"])):
            ax.plot(t, solution[j+1], linestyle='dashed', label=f'I{j+1} (Стадия {j+1})')
        ax.plot(t, solution[-1], 'g', label='Выздоровевшие')
        ax.grid()
        ax.set_ylim(0, 1)

if __name__ == "__main__":
    root = tk.Tk()
    app = EpidemicModelsApp(root)
    def on_closing():
        for fig in app.figs:
            plt.close(fig)
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()