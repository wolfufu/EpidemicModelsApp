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
        self.country_population = {}
        self.create_widgets()
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
        param_definitions = {"SI": [("beta", "β")],"SIR": [("beta", "β"), ("gamma", "γ")],"SIRS": [("beta", "β"), ("gamma", "γ"), ("delta", "δ")],"SEIR": [("beta", "β"), ("sigma", "σ"), ("gamma", "γ")],"SIQR": [("beta", "β"), ("gamma", "γ"), ("delta", "δ"), ("mu", "μ")],"MSEIR": [("mu", "μ"), ("delta", "δ"), ("beta", "β"), ("sigma", "σ"), ("gamma", "γ")],"M": [("beta", "β"), ("k1", "k₁"), ("k2", "k₂"), ("k3", "k₃"), ("gamma", "γ")]}
        init_definitions = {"SI": [("S0", "S₀"), ("I0", "I₀")],"SIR": [("S0", "S₀"), ("I0", "I₀"), ("R0", "R₀")],"SIRS": [("S0", "S₀"), ("I0", "I₀"), ("R0", "R₀")],"SEIR": [("S0", "S₀"), ("E0", "E₀"), ("I0", "I₀"), ("R0", "R₀")],"SIQR": [("S0", "S₀"), ("I0", "I₀"), ("Q0", "Q₀"), ("R0", "R₀")],"MSEIR": [("M0", "M₀"), ("S0", "S₀"), ("E0", "E₀"), ("I0", "I₀"), ("R0", "R₀")],"M": [("S0", "S₀"), ("I10", "I₁₀"), ("I20", "I₂₀"), ("I30", "I₃₀"), ("R0", "R₀")]}
        for code, label in param_definitions.get(model_code, []):
            row = ttk.Frame(param_group)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=25).pack(side=tk.LEFT)
            entry = ttk.Entry(row, width=8, validate="key", validatecommand=(validate_num, '%P'))
            entry.pack(side=tk.RIGHT)
            param_entries[code] = entry
        for code, label in init_definitions.get(model_code, []):
            row = ttk.Frame(init_group)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=25).pack(side=tk.LEFT)
            entry = ttk.Entry(row, width=8, validate="key", validatecommand=(validate_num, '%P'))
            entry.pack(side=tk.RIGHT)
            init_entries[code] = entry
        def on_entry_change(*args):
            if not self.validate_sum(init_entries):
                messagebox.showwarning("Ошибка", "Сумма начальных значений не должна превышать 1")
        for entry in param_entries.values():
            entry.bind("<FocusOut>", on_entry_change)
        for entry in init_entries.values():
            entry.bind("<FocusOut>", on_entry_change)
        self.model_param_tabs[model_code] = {"frame": frame,"param_entries": param_entries,"init_entries": init_entries}
        self.params_notebook.add(frame, text=model_code)
    def create_models_tab(self, parent):
        models_group = ttk.LabelFrame(parent, text="Выберите модели (макс. 4)", padding=10)
        models_group.pack(fill=tk.BOTH, pady=5)
        self.model_vars = {}
        models = [('SI', 'Модель SI (восприимчивые-инфицированные)'),('SIR', 'Модель SIR (восприимчивые-инфицированные-выздоровевшие)'),('SIRS', 'Модель SIRS (с временным иммунитетом)'),('SEIR', 'Модель SEIR (с латентным периодом)'),('SIQR', 'Модель SIQR (с изоляцией)'),('MSEIR', 'Модель MSEIR (с материнским иммунитетом)'),('M', 'M-модель (3 стадии инфекции)')  ]
        for model_code, model_desc in models:
            var = tk.BooleanVar()
            cb = ttk.Checkbutton(models_group, text=model_desc, variable=var, command=self.update_model_selection)
            cb.pack(anchor='w', padx=5, pady=2)
            self.model_vars[model_code] = var
        method_group = ttk.LabelFrame(parent, text="Метод решения", padding=10)
        method_group.pack(fill=tk.BOTH, pady=5)
        self.method_var = tk.StringVar(value="runge_kutta")
        ttk.Radiobutton(method_group, text="Рунге-Кутта 4-го порядка", variable=self.method_var, value="runge_kutta").pack(anchor='w', padx=5, pady=2)
        ttk.Radiobutton(method_group, text="Метод Эйлера", variable=self.method_var, value="euler").pack(anchor='w', padx=5, pady=2)
        ttk.Button(parent, text="Запустить моделирование", command=self.run_simulation).pack(fill=tk.X, pady=10)
        ttk.Button(parent, text="Справка по моделям", command=self.open_model_docs).pack(fill=tk.X, pady=5)
    def create_params_tab(self, parent):
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
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
        ttk.Button(load_group, text="Загрузить данные из CSV", command=self.load_csv_data).pack(fill=tk.X, pady=5)
        export_group = ttk.LabelFrame(parent, text="Экспорт результатов", padding=10)
        export_group.pack(fill=tk.BOTH, pady=5)
        ttk.Button(export_group, text="Экспорт в Excel и ZIP", command=self.export_results).pack(fill=tk.X, pady=5)
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
            self.model.figs.append(fig)
            self.model.axes.append(ax)
            self.model.canvases.append(canvas)
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
        save_path = filedialog.asksaveasfilename(defaultextension=".zip",filetypes=[("ZIP files", "*.zip")],initialfile="epidemic_results.zip")
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
                        initials_df = pd.DataFrame({'Параметр': ['S₀ (восприимчивые)', 'I₀ (инфицированные)', 'R₀ (выздоровевшие)','E₀ (латентные)', 'Q₀ (изолированные)', 'M₀ (материнский иммунитет)'],'Значение': [initials_data.get("S0", ""),initials_data.get("I0", ""),initials_data.get("R0", ""),initials_data.get("E0", ""),initials_data.get("Q0", ""),initials_data.get("M0", "")]})
                        initials_df.to_excel(writer, sheet_name='Начальные данные', index=False)
                        params_data = {}
                        for key, entry in model_tab["param_entries"].items():
                            params_data[key] = entry.get()
                        params_df = pd.DataFrame({'Параметр': ['β (скорость заражения)', 'γ (скорость выздоровления)','δ (потеря иммунитета)', 'σ (переход в инфекционные)','μ (выход из изоляции)'],'Значение': [params_data.get("beta", ""),params_data.get("gamma", ""),params_data.get("delta", ""),params_data.get("sigma", ""),params_data.get("mu", "")]})
                        params_df.to_excel(writer, sheet_name='Параметры', index=False)
                        method_info = pd.DataFrame({'Информация': ['Метод решения:', 'Выбранная модель:'],'Значение': [self.method_var.get().replace("runge_kutta", "Рунге-Кутта 4-го порядка").replace("euler", "Метод Эйлера"), model_name]})
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
                            chart.add_series({'name': f"='Решение'!${col_letter}$4",'categories': categories,'values': f"='Решение'!${col_letter}$5:${col_letter}${max_row}",})
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
if __name__ == "__main__":
    root = tk.Tk()
    app = EpidemicApp(root)
    root.mainloop()