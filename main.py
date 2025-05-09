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

class NumericalMethods:
    def __init__(self):
        result = None

    def euler_method(self, model_func, y0, t, args):
        """Реализация метода Эйлера для решения СДУ"""
        y = np.zeros((len(t), len(y0)))
        y[0] = y0
        
        for i in range(1, len(t)):
            dt = t[i] - t[i-1]
            dy = model_func(y[i-1], t[i-1], *args)
            y[i] = y[i-1] + dy * dt
        print(y.T)
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
        
        print(y)
        self.result = y.T

class EpidemicModels(NumericalMethods):
    def __init__(self):
        self.current_models = []
        self.numeric_methods = []

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
    
    def run_si_model(self, t, params, initials, plot_index, return_solution=False, method="runge_kutta"):
        """Запускает SI модель на указанном графике"""
        y0 = [initials["S0"], initials["I0"]]

        if method == "runge_kutta":
            solution = self.runge_kutta_4(self.models_obj.si_model, y0, t, (params["beta"],))
        else:
            solution = self.euler_method(self.models_obj.si_model, y0, t, (params["beta"],))

        S, I = solution
        
        ax = self.axes[plot_index]
        ax.plot(t, S, 'b', label='Восприимчивые')
        ax.plot(t, I, 'r', label='Инфицированные')
        ax.grid()
        ax.set_ylim(0, 1)
        ax.set_xlabel('Дни')
        ax.set_ylabel('Доля населения')
        ax.grid(True)
        
        if return_solution:
            return {"S": S, "I": I}
    
    def run_sir_model(self, t, params, initials, plot_index, return_solution=False, method="runge_kutta"):
        """Запускает SIR модель на указанном графике"""
        y0 = [initials["S0"], initials["I0"], initials["R0"]]

        if method == "runge_kutta":
            solution = self.runge_kutta_4(self.models_obj.sir_model, y0, t, (params["beta"], params["gamma"]))
        else:
            solution = self.euler_method(self.models_obj.sir_model, y0, t, (params["beta"], params["gamma"]))

        S, I, R = solution
        
        ax = self.axes[plot_index]
        ax.plot(t, S, 'b', label='Восприимчивые')
        ax.plot(t, I, 'r', label='Инфицированные')
        ax.plot(t, R, 'g', label='Выздоровевшие')
        ax.grid()
        ax.set_ylim(0, 1)
        ax.set_xlabel('Дни')
        ax.set_ylabel('Доля населения')
        ax.grid(True)

        if return_solution:
            return {"S": S, "I": I, "R": R}
    
    def run_sirs_model(self, t, params, initials, plot_index, return_solution=False, method="runge_kutta"):
        """Запускает SIRS модель на указанном графике"""
        y0 = [initials["S0"], initials["I0"], initials["R0"]]

        if method == "runge_kutta":
            solution = self.runge_kutta_4(self.models_obj.sirs_model, y0, t, (params["beta"], params["gamma"], params["delta"]))
        else:
            solution = self.euler_method(self.models_obj.sirs_model, y0, t, (params["beta"], params["gamma"], params["delta"]))

        S, I, R = solution
        
        ax = self.axes[plot_index]
        ax.plot(t, S, 'b', label='Восприимчивые')
        ax.plot(t, I, 'r', label='Инфицированные')
        ax.plot(t, R, 'g', label='Выздоровевшие')
        ax.grid()
        ax.set_ylim(0, 1)
        ax.set_xlabel('Дни')
        ax.set_ylabel('Доля населения')
        ax.grid(True)

        if return_solution:
            return {"S": S, "I": I, "R": R, "S": S}
    
    def run_seir_model(self, t, params, initials, plot_index, return_solution=False, method="runge_kutta"):
        """Запускает SEIR модель на указанном графике"""
        y0 = [initials["S0"], initials["E0"], initials["I0"], initials["R0"]]

        if method == "runge_kutta":
            solution = self.runge_kutta_4(self.models_obj.seir_model, y0, t, (params["beta"], params["sigma"], params["gamma"]))
        else:
            solution = self.euler_method(self.models_obj.seir_model, y0, t, (params["beta"], params["sigma"], params["gamma"]))

        S, E, I, R = solution
        
        ax = self.axes[plot_index]
        ax.plot(t, S, 'b', label='Восприимчивые')
        ax.plot(t, E, 'y', label='Латентные')
        ax.plot(t, I, 'r', label='Инфицированные')
        ax.plot(t, R, 'g', label='Выздоровевшие')
        ax.grid()
        ax.set_ylim(0, 1)
        ax.set_xlabel('Дни')
        ax.set_ylabel('Доля населения')
        ax.grid(True)

        if return_solution:
            return {"S": S, "E": E, "I": I, "R": R}
    
    def run_mseir_model(self, t, params, initials, plot_index, return_solution=False, method="runge_kutta"):
        """Запускает MSEIR модель на указанном графике"""
        y0 = [initials["M0"], initials["S0"], initials["E0"], initials["I0"], initials["R0"]]

        if method == "runge_kutta":
            solution = self.runge_kutta_4(self.models_obj.mseir_model, y0, t, 
                                   (params["mu"], params["delta"], params["beta"], params["sigma"], params["gamma"]))
        else:
            solution = self.euler_method(self.models_obj.mseir_model, y0, t, 
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
        ax.set_xlabel('Дни')
        ax.set_ylabel('Доля населения')
        ax.grid(True)

        if return_solution:
            return {"M": M, "S": S, "E": E, "I": I, "R": R}
    
    def run_siqr_model(self, t, params, initials, plot_index, return_solution=False, method="runge_kutta"):
        """Запускает SIQR модель на указанном графике"""
        y0 = [initials["S0"], initials["I0"], initials["Q0"], initials["R0"]]

        if method == "runge_kutta":
            solution = self.runge_kutta_4(self.models_obj.siqr_model, y0, t, 
                                   (params["beta"], params["gamma"], params["delta"], params["mu"]))
        else:
            solution = self.euler_method(self.models_obj.siqr_model, y0, t, 
                                   (params["beta"], params["gamma"], params["delta"], params["mu"]))

        S, I, Q, R = solution
        
        ax = self.axes[plot_index]
        ax.plot(t, S, 'b', label='Восприимчивые')
        ax.plot(t, I, 'r', label='Инфицированные')
        ax.plot(t, Q, 'g', label='Изолированные')
        ax.plot(t, R, 'k', label='Выздоровевшие')
        ax.grid()
        ax.set_ylim(0, 1)
        ax.set_xlabel('Дни')
        ax.set_ylabel('Доля населения')
        ax.grid(True)

        if return_solution:
            return {"S": S, "I": I, "Q": Q, "R": R}
    
    def run_m_model(self, t, params, initials, plot_index, return_solution=False, method="runge_kutta"):
        """Запускает M-модель на указанном графике"""
        y0 = [initials["S0"]] + initials["I0"] + [initials["R0"]]

        if method == "runge_kutta":
            solution = self.runge_kutta_4(self.models_obj.multi_stage_model, y0, t, 
                                (params["beta"], params["k"], params["gamma"]))
        else:
            solution = self.euler_method(self.models_obj.multi_stage_model, y0, t, 
                                (params["beta"], params["k"], params["gamma"]))
        
        ax = self.axes[plot_index]
        ax.plot(t, solution[0], 'b', label='Восприимчивые')
        for j in range(len(initials["I0"])):
            ax.plot(t, solution[j+1], linestyle='dashed', label=f'I{j+1} (Стадия {j+1})')
        ax.plot(t, solution[-1], 'g', label='Выздоровевшие')
        ax.grid()
        ax.set_ylim(0, 1)
        ax.set_xlabel('Дни')
        ax.set_ylabel('Доля населения')
        ax.grid(True)
        
        if return_solution:
            result = {
                "S": solution[0],
                "R": solution[-1]
            }
            # Добавляем все стадии инфицированных
            for j in range(len(initials["I0"])):
                result[f"I{j+1}"] = solution[j+1]
            
            return result
    
class EpidemicModelsTechLog:
    def __init__(self):
        pass

    def load_excel_data(self):
        """Загружает данные из Excel файла"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # Читаем Excel файл
            self.excel_data = pd.read_excel(file_path)
            self.excel_data_columns = list(self.excel_data.columns)
            
            # Создаем окно для выбора столбцов
            self.create_data_mapping_window()
            
            messagebox.showinfo("Успех", "Данные успешно загружены")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить данные: {str(e)}")

    def create_data_mapping_window(self):
        """Создает окно для сопоставления столбцов из Excel с параметрами модели"""
        if self.excel_data is None:
            return
        
        mapping_window = tk.Toplevel(self.root)
        mapping_window.title("Сопоставление данных")
        mapping_window.geometry("500x400")
        
        # Фрейм для выбора модели
        model_frame = ttk.LabelFrame(mapping_window, text="Выберите модель для сопоставления")
        model_frame.pack(fill=tk.X, padx=5, pady=5)
        
        model_var = tk.StringVar()
        model_combobox = ttk.Combobox(model_frame, textvariable=model_var, 
                                    values=list(self.available_models.keys()))
        model_combobox.pack(fill=tk.X, padx=5, pady=5)
        
        # Фрейм для сопоставления столбцов
        mapping_frame = ttk.LabelFrame(mapping_window, text="Сопоставьте столбцы")
        mapping_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Получаем параметры модели при выборе
        def update_mappings(event):
            # Очищаем предыдущие сопоставления
            for widget in mapping_frame.winfo_children():
                widget.destroy()
            
            model_name = model_var.get()
            if not model_name:
                return
            
            model_params = self.available_models[model_name]["params"]
            initial_params = model_params["initial"]
            
            self.mapping_widgets = {}
            
            row = 0
            for param, info in initial_params.items():
                ttk.Label(mapping_frame, text=info["name"]).grid(row=row, column=0, sticky="w", padx=5, pady=2)
                
                var = tk.StringVar()
                cb = ttk.Combobox(mapping_frame, textvariable=var, 
                                  values=[""] + self.excel_data_columns)
                cb.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
                
                self.mapping_widgets[param] = var
                row += 1
            
            # Кнопка применения
            ttk.Button(mapping_frame, text="Применить", 
                      command=lambda: self.apply_data_mapping(model_name)).grid(
                      row=row, column=0, columnspan=2, pady=10)
        
        model_combobox.bind("<<ComboboxSelected>>", update_mappings)
        
    def apply_data_mapping(self, model_name):
        """Применяет сопоставление данных к выбранной модели"""
        try:
            mappings = {}
            for param, var in self.mapping_widgets.items():
                col_name = var.get()
                if col_name:
                    mappings[param] = col_name
            
            if not mappings:
                messagebox.showwarning("Предупреждение", "Не выбрано ни одного сопоставления")
                return
            
            # Обновляем начальные условия в интерфейсе
            model_info = self.available_models[model_name]
            initial_params = model_info["params"]["initial"]
            
            # Находим вкладку с параметрами модели
            for tab_id in self.params_notebook.tabs():
                if self.params_notebook.tab(tab_id, "text") == model_name:
                    # Находим вкладку с начальными условиями
                    notebook = self.params_notebook.nametowidget(tab_id).winfo_children()[0]
                    for inner_tab_id in notebook.tabs():
                        if notebook.tab(inner_tab_id, "text") == "Начальные условия":
                            initial_tab = notebook.nametowidget(inner_tab_id)
                            
                            # Обновляем значения
                            for param, widget in self.model_params[model_name]["initial_widgets"].items():
                                if param in mappings:
                                    col_name = mappings[param]
                                    value = self.excel_data[col_name].iloc[0]
                                    
                                    if isinstance(widget, list):
                                        # Для списков значений (например, нескольких стадий)
                                        for i, entry in enumerate(widget):
                                            if i < len(value):
                                                entry.delete(0, tk.END)
                                                entry.insert(0, str(value[i]))
                                    else:
                                        # Для одиночных значений
                                        widget.delete(0, tk.END)
                                        widget.insert(0, str(value))
            
            messagebox.showinfo("Успех", "Данные успешно применены")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось применить данные: {str(e)}")

    def export_results(self):
        """Экспортирует результаты моделирования в Excel-файлы и упаковывает в ZIP-архив"""
        if not hasattr(self, 'model_results') or not self.model_results:
            messagebox.showwarning("Предупреждение", "Сначала выполните моделирование")
            return
        
        # Запрашиваем место сохранения
        file_path = filedialog.asksaveasfilename(
            defaultextension=".zip",
            filetypes=[("ZIP архив", "*.zip")],
            title="Сохранить результаты моделирования"
        )
        
        if not file_path:
            return  # Пользователь отменил сохранение
        
        try:
            temp_dir = "temp_epidemic_models"
            os.makedirs(temp_dir, exist_ok=True)
            
            for model_name, data in self.model_results.items():
                # Создаем Excel-файл
                excel_file = os.path.join(temp_dir, f"{model_name}.xlsx")
                
                with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
                    # Лист с параметрами
                    params_df = pd.DataFrame.from_dict(data["parameters"], orient='index', columns=['Значение'])
                    params_df.index.name = 'Параметр'
                    params_df.to_excel(writer, sheet_name='Параметры')
                    
                    # Лист с начальными условиями
                    initials_df = pd.DataFrame.from_dict(data["initial_conditions"], orient='index', columns=['Значение'])
                    initials_df.index.name = 'Переменная'
                    initials_df.to_excel(writer, sheet_name='Начальные условия')
                    
                    # Лист с решением
                    solution_df = pd.DataFrame(data["solution"])
                    solution_df.index = data["time_points"]
                    solution_df.index.name = 'День'
                    solution_df.to_excel(writer, sheet_name='Решение')
                    
                    workbook = writer.book
                    worksheet = workbook.add_worksheet('График')
                    
                    # Вставляем изображение
                    data["graph"].seek(0)
                    worksheet.insert_image('B2', f"{model_name}_graph.png", {'image_data': data["graph"]})
            
            # Создаем ZIP-архив
            with zipfile.ZipFile(file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        zipf.write(os.path.join(root, file), file)
            
            # Удаляем временную папку
            for root, dirs, files in os.walk(temp_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(temp_dir)
            
            messagebox.showinfo("Успех", f"Результаты успешно экспортированы в {file_path}")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при экспорте результатов: {str(e)}")
            # Удаляем временную папку в случае ошибки
            if os.path.exists(temp_dir):
                for root, dirs, files in os.walk(temp_dir, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
                os.rmdir(temp_dir)

class EpidemicModelsApp(EpidemicModelsTechLog, EpidemicModels):
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
            
        model_frame = ttk.LabelFrame(left_frame, text="Выбор моделей (до 4)", padding=10)
        model_frame.pack(fill=tk.X, pady=5)
        
        self.models_container = ttk.Frame(model_frame)
        self.models_container.pack(fill=tk.X)
        
        self.add_button = ttk.Button(model_frame, text="+ Добавить модель", 
                                   command=lambda: self.add_model_field(show_remove_button=True), state=tk.NORMAL)
        self.add_button.pack(pady=5)
        
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

        # Выбор численного метода
        method_frame = ttk.LabelFrame(left_frame, text="Численный метод", padding=10)
        method_frame.pack(fill=tk.X, pady=5)

        self.method_var = tk.StringVar(value="runge_kutta")
        method_combobox = ttk.Combobox(method_frame, textvariable=self.method_var,
                                        values=["runge_kutta", "euler"], state="readonly")
        method_combobox.pack(fill=tk.X, padx=5)

        
        ttk.Button(left_frame, text="Запустить моделирование", 
                  command=self.run_models).pack(pady=10)
        
        ttk.Button(left_frame, text="Загрузить данные из Excel", 
                 command=self.load_excel_data).pack(pady=5)
        
        self.excel_data = None
        self.excel_data_columns = []
        
        ttk.Button(left_frame, text="Экспорт в Excel", 
                 command=self.export_results).pack(pady=5)
        
        self.params_notebook = ttk.Notebook(left_frame)
        self.params_notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
    
    def create_right_panel(self):
        """Создает правую панель с графиками"""
        right_frame = ttk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.plot_frames = []
        self.figs = []
        self.axes = []
        self.canvases = []
        
        for i in range(4):
            frame = ttk.LabelFrame(right_frame, text=f'График {i+1}', padding=5)
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
            
            frame.grid_remove()
    
    def add_model_field(self, initial_model=None, show_remove_button=True):
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
        
        if initial_model:
            var.set(initial_model)
            self.model_selected(var)
        
        if show_remove_button:
            btn = ttk.Button(frame, text="×", width=2, 
                           command=lambda: self.remove_model_field(frame, var))
            btn.pack(side=tk.RIGHT)
        else:
            ttk.Label(frame, width=2).pack(side=tk.RIGHT)
        
        cb.bind("<<ComboboxSelected>>", lambda e, v=var: self.model_selected(v))
        
        self.model_vars.append(var)
        self.model_widgets.append({"frame": frame, "combobox": cb, "button": btn if show_remove_button else None})
        
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
        
        for i, widget in enumerate(self.model_widgets):
            if widget["frame"] == frame:
                self.model_widgets.pop(i)
                self.model_vars.pop(i)
                break
        
        frame.destroy()
        
        if len(self.model_widgets) < 4:
            self.add_button.config(state=tk.NORMAL)
        
        self.update_plots_visibility()
    
    def model_selected(self, var):
        """Обработчик выбора модели"""
        model_name = var.get()
        if not model_name:
            return
        
        for v in self.model_vars:
            if v != var and v.get() == model_name:
                messagebox.showwarning("Предупреждение", "Эта модель уже выбрана")
                var.set("")
                return
        
        if model_name not in self.selected_models:
            self.selected_models.append(model_name)
            self.update_params_notebook()
        
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
        for tab in self.params_notebook.tabs():
            self.params_notebook.forget(tab)
        
        for i, model_name in enumerate(self.selected_models):
            tab = ttk.Frame(self.params_notebook)
            self.params_notebook.add(tab, text=model_name)
            
            model_info = self.available_models[model_name]
            params = model_info["params"]
            
            notebook = ttk.Notebook(tab)
            notebook.pack(fill=tk.BOTH, expand=True)
            
            params_tab = ttk.Frame(notebook)
            notebook.add(params_tab, text="Параметры")
            
            initial_tab = ttk.Frame(notebook)
            notebook.add(initial_tab, text="Начальные условия")
            
            if not hasattr(self, 'model_params'):
                self.model_params = {}
            self.model_params[model_name] = {
                "param_widgets": {},
                "initial_widgets": {}
            }
            
            row = 0
            for param, info in params.items():
                if param == "initial":
                    continue
                    
                ttk.Label(params_tab, text=info["name"]).grid(row=row, column=0, sticky="w", padx=5, pady=2)
                
                if isinstance(info["default"], list):
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
        """Запускает все выбранные модели и сохраняет результаты для экспорта"""
        try:
            start_date = self.start_date_entry.get_date()
            end_date = self.end_date_entry.get_date()
            method = self.method_var.get()
            
            if end_date <= start_date:
                messagebox.showerror("Ошибка", "Конечная дата должна быть позже начальной")
                return
            
            delta = end_date - start_date
            t = np.linspace(0, delta.days, delta.days + 1)
            
            # Очищаем предыдущие результаты
            self.model_results = {}
            
            for ax in self.axes:
                ax.clear()
            
            for i, model_name in enumerate(self.selected_models):
                if i >= 4:  # Не больше 4 моделей
                    break
                
                params = self.get_parameter_values(model_name)
                initials = self.get_initial_values(model_name)
                
                # Сохраняем параметры и начальные условия
                model_data = {
                    "parameters": params,
                    "initial_conditions": initials,
                    "time_points": t,
                    "solution": None,
                    "graph": None
                }
                
                if model_name == "SI":
                    solution = self.run_si_model(t, params, initials, i, return_solution=True, method=method)
                elif model_name == "SIR":
                    solution = self.run_sir_model(t, params, initials, i, return_solution=True, method=method)
                elif model_name == "SIRS":
                    solution = self.run_sirs_model(t, params, initials, i, return_solution=True, method=method)
                elif model_name == "SEIR":
                    solution = self.run_seir_model(t, params, initials, i, return_solution=True, method=method)
                elif model_name == "MSEIR":
                    solution = self.run_mseir_model(t, params, initials, i, return_solution=True, method=method)
                elif model_name == "SIQR":
                    solution = self.run_siqr_model(t, params, initials, i, return_solution=True, method=method)
                elif model_name == "M-модель":
                    solution = self.run_m_model(t, params, initials, i, return_solution=True, method=method h)
                
                # Сохраняем решение
                model_data["solution"] = solution
                
                # Сохраняем график
                buf = BytesIO()
                self.figs[i].savefig(buf, format='png')
                buf.seek(0)
                model_data["graph"] = buf
                
                # Добавляем в результаты
                self.model_results[model_name] = model_data
                
                self.axes[i].set_title(self.available_models[model_name]["name"])
                self.axes[i].legend()
                self.canvases[i].draw()
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при выполнении моделирования: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = EpidemicModelsApp(root)
    def on_closing():
        for fig in app.figs:
            plt.close(fig)
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()