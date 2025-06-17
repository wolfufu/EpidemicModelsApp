import covasim as cv

# Параметры симуляции
params = {
    'pop_size': 10_000,       # Численность населения
    'pop_infected': 100,      # Начальное число заражённых
    'n_days': 90,             # Длительность симуляции (дни)
    'beta': 0.05,             # Коэффициент заразности
    'contacts': 20            # Среднее число контактов в день
}

# Создание и запуск симуляции
sim = cv.Sim(params)
sim.run()

# Визуализация
sim.plot(style='ggplot', fig_args={'figsize': (12, 8)})