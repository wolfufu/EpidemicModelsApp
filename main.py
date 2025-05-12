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
        
        print(y)
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

    def run_m_model(self, t, params, initials, plot_index, return_solution=False, method="runge_kutta"):
        """Запускает M-модель на указанном графике"""
        y0 = [initials["S0"]] + initials["I0"] + [initials["R0"]]

        if method == "runge_kutta":
            self.runge_kutta_4(self.multi_stage_model, y0, t, 
                        (params["beta"], params["k"], params["gamma"]))
        else:
            self.euler_method(self.multi_stage_model, y0, t, 
                        (params["beta"], params["k"], params["gamma"]))

        solution = self.result
        
        ax = self.axes[plot_index]
        ax.clear()
        ax.plot(t, solution[0], 'b', label='Восприимчивые')
        for j in range(len(initials["I0"])):
            ax.plot(t, solution[j+1], linestyle='dashed', label=f'I{j+1} (Стадия {j+1})')
        ax.plot(t, solution[-1], 'g', label='Выздоровевшие')
        ax.set_ylim(0, 1)
        ax.set_xlabel('Дни')
        ax.set_ylabel('Доля населения')
        ax.grid(True)
        ax.legend()
        self.canvases[plot_index].draw()
        
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
        self.file_data = None
        self.date_column = None
        self.data_start_date = None
        self.data_end_date = None

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

    def load_csv_country_data(self):
        """Загружает CSV файл с данными по странам и позволяет выбрать страну и сопоставить столбцы"""
        # Проверяем, есть ли выбранные модели
        if len(self.selected_models) > 0:
            messagebox.showwarning("Предупреждение", 
                                "Пожалуйста, сначала удалите все выбранные модели перед загрузкой данных")
            return

        file_path = filedialog.askopenfilename(
            filetypes=[("CSV файлы", "*.csv"), ("Все файлы", "*.*")]
        )

        if not file_path:
            return

        try:
            df = pd.read_csv(file_path)
            
            # Удаляем строки с NaN в датах
            df = df.dropna(subset=['Date'])  # Замените 'Date' на фактическое название столбца с датами
            
            self.file_data = df
            
            # Автоматически ищем столбец с датами
            date_col = None
            for col in df.columns:
                if 'date' in col.lower():
                    date_col = col
                    break
            
            if not date_col:
                messagebox.showerror("Ошибка", "Не найден столбец с датами в файле")
                return
                
            self.date_column = date_col
            
            # Преобразуем даты, игнорируя ошибки
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
            # Удаляем строки с невалидными датами
            df = df.dropna(subset=[date_col])
            
            # Проверяем, остались ли данные
            if df.empty:
                messagebox.showerror("Ошибка", "Нет валидных данных после обработки")
                return
                
            # Определяем диапазон дат
            self.data_start_date = df[date_col].min()
            self.data_end_date = df[date_col].max()
            
            # Ищем столбец с названиями стран
            country_col = None
            for col in df.columns:
                if 'country' in col.lower() or 'region' in col.lower():
                    country_col = col
                    break
            
            if not country_col:
                messagebox.showerror("Ошибка", "Не найден столбец с названиями стран/регионов")
                return
            
            # Создаем окно для выбора страны
            country_window = tk.Toplevel(self.root)
            country_window.title("Выбор страны/региона")
            country_window.geometry("400x200")
            
            ttk.Label(country_window, text="Выберите страну/регион:").pack(pady=10)
            
            # Получаем уникальные страны/регионы, удаляя NaN
            countries = [c for c in df[country_col].unique() if pd.notna(c)]
            if not countries:
                messagebox.showerror("Ошибка", "Не найдено валидных стран/регионов")
                return
                
            self.country_var = tk.StringVar()
            country_combobox = ttk.Combobox(country_window, textvariable=self.country_var, values=countries)
            country_combobox.pack(pady=5)
            
            ttk.Button(country_window, text="Применить", 
                    command=lambda: self.on_country_selected(country_window, country_col)).pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить CSV: {str(e)}")

    def on_country_selected(self, window, country_col):
        """Обрабатывает выбор страны и переходит к выбору дат"""
        selected_country = self.country_var.get()
        if not selected_country:
            messagebox.showwarning("Предупреждение", "Пожалуйста, выберите страну/регион")
            return
        
        try:
            # Фильтруем данные по выбранной стране
            self.file_data = self.file_data[self.file_data[country_col] == selected_country]
            
            # Убедимся, что данные не пустые
            if self.file_data.empty:
                messagebox.showerror("Ошибка", f"Нет данных для страны: {selected_country}")
                return
                
            # Обновляем диапазон дат после фильтрации
            self.data_start_date = self.file_data[self.date_column].min()
            self.data_end_date = self.file_data[self.date_column].max()
            
            # Проверяем, что даты валидны
            if pd.isna(self.data_start_date) or pd.isna(self.data_end_date):
                messagebox.showerror("Ошибка", "Не удалось определить валидный диапазон дат")
                return
                
            window.destroy()
            
            # Теперь показываем окно выбора дат
            self.update_date_selection_ui()
        
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при обработке данных: {str(e)}")
    
    def update_date_selection_ui(self):
        """Обновляет интерфейс выбора дат на основе загруженных данных"""
        if not self.data_start_date or not self.data_end_date:
            return
            
        # Создаем новое окно для выбора диапазона дат
        date_window = tk.Toplevel(self.root)
        date_window.title("Выбор временного диапазона")
        date_window.geometry("400x300")
        
        ttk.Label(date_window, text=f"Данные доступны с {self.data_start_date.date()} по {self.data_end_date.date()}").pack(pady=10)
        
        # Фрейм для исторических данных
        hist_frame = ttk.LabelFrame(date_window, text="Исторические данные")
        hist_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(hist_frame, text="Начальная дата:").pack()
        self.hist_start_entry = DateEntry(
            hist_frame,
            width=12,
            background='darkblue',
            foreground='white',
            borderwidth=2,
            date_pattern='dd.mm.yyyy'
        )
        self.hist_start_entry.pack()
        self.hist_start_entry.set_date(self.data_start_date)
        
        ttk.Label(hist_frame, text="Конечная дата:").pack()
        self.hist_end_entry = DateEntry(
            hist_frame,
            width=12,
            background='darkblue',
            foreground='white',
            borderwidth=2,
            date_pattern='dd.mm.yyyy'
        )
        self.hist_end_entry.pack()
        self.hist_end_entry.set_date(self.data_end_date)
        
        # Фрейм для прогнозирования
        pred_frame = ttk.LabelFrame(date_window, text="Прогнозирование")
        pred_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(pred_frame, text="Конечная дата прогноза:").pack()
        self.pred_end_entry = DateEntry(
            pred_frame,
            width=12,
            background='darkblue',
            foreground='white',
            borderwidth=2,
            date_pattern='dd.mm.yyyy'
        )
        self.pred_end_entry.pack()
        self.pred_end_entry.set_date(self.data_end_date + timedelta(days=30))
        
        def apply_and_close():
            self.apply_date_selection(date_window)
        
        ttk.Button(date_window, text="Применить", command=apply_and_close).pack(pady=10)
    
    def apply_date_selection(self, window):
        """Применяет выбранные даты и закрывает окно"""
        try:
            self.hist_start_date = self.hist_start_entry.get_date()
            self.hist_end_date = self.hist_end_entry.get_date()
            self.pred_end_date = self.pred_end_entry.get_date()
            
            if self.hist_start_date >= self.hist_end_date:
                messagebox.showerror("Ошибка", "Конечная дата исторических данных должна быть позже начальной")
                return
                
            if self.hist_end_date >= self.pred_end_date:
                messagebox.showerror("Ошибка", "Дата прогноза должна быть позже конечной даты исторических данных")
                return
                
            window.destroy()
            
            # Заполняем начальные условия из данных
            last_row = self.file_data.iloc[-1]
            self.fill_initial_conditions_from_data(last_row)
            
            # Сообщаем пользователю, что можно добавлять модели
            messagebox.showinfo("Информация", "Теперь вы можете добавить модели для анализа данных")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при выборе дат: {str(e)}")

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
        
        # После выбора модели автоматически запускаем моделирование
        if hasattr(self, 'file_data') and self.file_data is not None:
            self.run_models_with_historical_data()

    

    def fill_initial_conditions_from_data(self, last_row):
        """Заполняет начальные условия на основе данных из файла"""
        if not hasattr(self, 'model_params'):
            return
            
        # Попробуем найти столбцы с нужными данными
        confirmed_col = next((col for col in last_row.index if 'confirmed' in col.lower()), None)
        recovered_col = next((col for col in last_row.index if 'recovered' in col.lower()), None)
        deaths_col = next((col for col in last_row.index if 'death' in col.lower()), None)
        population_col = next((col for col in last_row.index if 'population' in col.lower()), None)
        
        # Если нашли население - используем, иначе считаем популяцию = 1
        population = 1.0
        if population_col:
            try:
                population = float(last_row[population_col])
            except:
                pass
                
        # Заполняем начальные условия для всех моделей
        for model_name in self.model_params:
            if model_name == "SIR":
                if confirmed_col and recovered_col:
                    confirmed = float(last_row[confirmed_col])
                    recovered = float(last_row[recovered_col])
                    
                    # Нормализуем до 1.0
                    infected = (confirmed - recovered) / population
                    recovered = recovered / population
                    susceptible = 1 - infected - recovered
                    
                    # Заполняем поля
                    if "S0" in self.model_params[model_name]["initial_widgets"]:
                        self.model_params[model_name]["initial_widgets"]["S0"].delete(0, tk.END)
                        self.model_params[model_name]["initial_widgets"]["S0"].insert(0, str(susceptible))
                    
                    if "I0" in self.model_params[model_name]["initial_widgets"]:
                        self.model_params[model_name]["initial_widgets"]["I0"].delete(0, tk.END)
                        self.model_params[model_name]["initial_widgets"]["I0"].insert(0, str(infected))
                    
                    if "R0" in self.model_params[model_name]["initial_widgets"]:
                        self.model_params[model_name]["initial_widgets"]["R0"].delete(0, tk.END)
                        self.model_params[model_name]["initial_widgets"]["R0"].insert(0, str(recovered))

class EpidemicModelsApp(EpidemicModelsTechLog, EpidemicModels):
    def __init__(self, root):
        EpidemicModelsTechLog.__init__(self)  # Initialize TechLog parent
        EpidemicModels.__init__(self)         # Initialize Models parent
        self.root = root
        self.root.title("Моделирование эпидемий")
        self.root.geometry("1400x800")
        
        self.models_obj = EpidemicModels()
        self.available_models = {
            "SIR": {"name": "SIR-модель", "func": self.sir_model, "params": self.get_model_parameters("sir_model")},
            "SI": {"name": "SI-модель", "func": self.si_model, "params": self.get_model_parameters("si_model")},
            "SIRS": {"name": "SIRS-модель", "func": self.sirs_model, "params": self.get_model_parameters("sirs_model")},
            "SEIR": {"name": "SEIR-модель", "func": self.seir_model, "params": self.get_model_parameters("seir_model")},
            "MSEIR": {"name": "MSEIR-модель", "func": self.mseir_model, "params": self.get_model_parameters("mseir_model")},
            "SIQR": {"name": "SIQR-модель", "func": self.siqr_model, "params": self.get_model_parameters("siqr")},
            "M-модель": {"name": "M-модель", "func": self.multi_stage_model, "params": self.get_model_parameters("m_model")}
        }
        
        self.selected_models = []
        self.model_widgets = []
        self.model_vars = []
        
        self.create_left_panel()
        self.create_right_panel()
        
        # Сразу создаем 4 поля для выбора моделей
        for _ in range(4):
            self.add_model_field(show_remove_button=False)

    def run_models(self):
        """Запускает моделирование с учетом исторических данных и прогноза"""
        # Проверяем, что выбрана хотя бы одна модель
        if len(self.selected_models) == 0:
            messagebox.showwarning("Предупреждение", "Пожалуйста, выберите хотя бы одну модель")
            return
            
        try:
            if hasattr(self, 'file_data') and self.file_data is not None:
                # Режим с историческими данными
                self.run_models_with_historical_data()
            else:
                # Обычный режим моделирования
                self.run_models_standard()
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при выполнении моделирования: {str(e)}")

    def run_models_with_historical_data(self):
        """Запускает моделирование с историческими данными и прогнозом"""
        # Получаем выбранные даты
        hist_start = self.hist_start_entry.get_date()
        hist_end = self.hist_end_entry.get_date()
        pred_end = self.pred_end_entry.get_date()
        
        # Фильтруем данные по выбранному диапазону
        mask = (self.file_data[self.date_column] >= hist_start) & (self.file_data[self.date_column] <= hist_end)
        hist_data = self.file_data.loc[mask]
        
        # Создаем временные точки для исторического периода
        hist_days = (hist_end - hist_start).days + 1
        t_hist = np.linspace(0, hist_days - 1, hist_days)
        
        # Создаем временные точки для прогноза
        pred_days = (pred_end - hist_end).days
        t_pred = np.linspace(hist_days, hist_days + pred_days - 1, pred_days)
        
        # Объединенные временные точки
        t_total = np.concatenate([t_hist, t_pred])
        
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
                "time_points": t_total,
                "solution": None,
                "graph": None
            }
            
            # Запускаем модель для прогноза
            if model_name == "SI":
                solution_pred = self.run_si_model(t_pred, params, initials, i, return_solution=True, method=self.method_var.get())
            elif model_name == "SIR":
                solution_pred = self.run_sir_model(t_pred, params, initials, i, return_solution=True, method=self.method_var.get())
            elif model_name == "SIRS":
                solution_pred = self.run_sir_model(t_pred, params, initials, i, return_solution=True, method=self.method_var.get())
            elif model_name == "SIQR":
                solution_pred = self.run_sir_model(t_pred, params, initials, i, return_solution=True, method=self.method_var.get())   
            elif model_name == "SIER":
                solution_pred = self.run_sir_model(t_pred, params, initials, i, return_solution=True, method=self.method_var.get())        
            elif model_name == "MSEIR":
                solution_pred = self.run_sir_model(t_pred, params, initials, i, return_solution=True, method=self.method_var.get())        
            elif model_name == "M-model":
                solution_pred = self.run_sir_model(t_pred, params, initials, i, return_solution=True, method=self.method_var.get())


            # Объединяем исторические данные и прогноз
            full_solution = {}
            for key in solution_pred.keys():
                # Создаем массив для всех точек
                full_solution[key] = np.zeros(len(t_total))
                
                # Заполняем историческую часть (здесь нужно адаптировать под ваши данные)
                # Например, если у вас есть столбец 'infected' в данных:
                if key == "I" and 'infected' in hist_data.columns:
                    full_solution[key][:len(t_hist)] = hist_data['infected'].values / hist_data['infected'].max()
                
                # Заполняем прогнозную часть
                full_solution[key][len(t_hist):] = solution_pred[key]
            
            # Сохраняем полное решение
            model_data["solution"] = full_solution
            
            # Обновляем график
            ax = self.axes[i]
            ax.clear()
            
            # Рисуем исторические данные
            if 'S' in full_solution:
                ax.plot(t_hist, full_solution['S'][:len(t_hist)], 'b', label='Восприимчивые (данные)')
                ax.plot(t_pred, full_solution['S'][len(t_hist):], 'b--', label='Восприимчивые (прогноз)')
            
            if 'I' in full_solution:
                ax.plot(t_hist, full_solution['I'][:len(t_hist)], 'r', label='Инфицированные (данные)')
                ax.plot(t_pred, full_solution['I'][len(t_hist):], 'r--', label='Инфицированные (прогноз)')

            if 'R' in full_solution:
                ax.plot(t_hist, full_solution['R'][:len(t_hist)], 'r', label='Выздоровевшие (данные)')
                ax.plot(t_pred, full_solution['R'][len(t_hist):], 'r--', label='Выздоровевшие (прогноз)')

            if 'E' in full_solution:
                ax.plot(t_hist, full_solution['E'][:len(t_hist)], 'r', label='Латентные (данные)')
                ax.plot(t_pred, full_solution['E'][len(t_hist):], 'r--', label='Латентные (прогноз)')

            if 'Q' in full_solution:
                ax.plot(t_hist, full_solution['Q'][:len(t_hist)], 'r', label='Изолированные (данные)')
                ax.plot(t_pred, full_solution['Q'][len(t_hist):], 'r--', label='Изолированные (прогноз)')

            if 'M' in full_solution:
                ax.plot(t_hist, full_solution['M'][:len(t_hist)], 'r', label='Иммунные (данные)')
                ax.plot(t_pred, full_solution['M'][len(t_hist):], 'r--', label='Иммунные (прогноз)')
                        
            ax.axvline(x=t_hist[-1], color='gray', linestyle='--', label='Начало прогноза')
            ax.set_ylim(0, 1)
            ax.set_xlabel('Дни')
            ax.set_ylabel('Доля населения')
            ax.grid(True)
            ax.legend()
            self.canvases[i].draw()
            
            # Сохраняем график
            buf = BytesIO()
            self.figs[i].savefig(buf, format='png')
            buf.seek(0)
            model_data["graph"] = buf
            
            # Добавляем в результаты
            self.model_results[model_name] = model_data
        
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
            
        model_frame = ttk.LabelFrame(left_frame, text="Выбор моделей", padding=10)
        model_frame.pack(fill=tk.X, pady=5)
        
        self.models_container = ttk.Frame(model_frame)
        self.models_container.pack(fill=tk.X)
        
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
        
        ttk.Button(left_frame, text="Загрузить данные из файла", 
           command=self.load_csv_country_data).pack(pady=5)

        
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
    
    def run_models_standard(self):
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
                    solution = self.run_m_model(t, params, initials, i, return_solution=True, method=method)
                
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