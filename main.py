import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib

# Указываем, что нужно использовать QtAgg для корректного вывода в PyCharm
matplotlib.use('QtAgg')


def generate_samples():
    sample_sizes = [20, 50, 100, 1000]
    normal_samples = {size: np.random.normal(loc=0, scale=1, size=size) for size in sample_sizes}
    uniform_samples = {size: np.random.uniform(low=-2, high=2, size=size) for size in sample_sizes}
    binomial_samples = {size: np.random.binomial(n=10, p=0.5, size=size) for size in sample_sizes}
    return normal_samples, uniform_samples, binomial_samples


def descriptive_statistics(sample):
    """ Возвращает статистические показатели """
    mean = np.mean(sample)
    mode = stats.mode(sample, keepdims=True)[0][0]
    median = np.median(sample)
    range_ = np.ptp(sample)
    variance_biased = np.var(sample)
    variance_unbiased = np.var(sample, ddof=1)
    quartiles = np.percentile(sample, [25, 50, 75])
    interquartile_range = quartiles[2] - quartiles[0]

    stats_dict = {
        "Mean": mean,
        "Mode": mode,
        "Median": median,
        "Range": range_,
        "Variance (biased)": variance_biased,
        "Variance (unbiased)": variance_unbiased,
        "Q1": quartiles[0],
        "Q2 (Median)": quartiles[1],
        "Q3": quartiles[2],
        "IQR": interquartile_range
    }
    return stats_dict


def plot_graphs(sample, sample_size, dist_type):
    """ Строит таблицу и графики """
    stats_values = descriptive_statistics(sample)  # Вычисляем статистику
    stats_text = "\n".join([f"{key}: {value:.3f}" for key, value in stats_values.items()])

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # Два графика на одной строке

    # 1️⃣ Гистограмма
    sns.histplot(sample, bins=30, kde=True, ax=axs[0])
    axs[0].set_title(f"Histogram - {dist_type} (size={sample_size})")
    axs[0].set_xlabel("Values")
    axs[0].set_ylabel("Density")

    # 2️⃣ ECDF-график
    sns.ecdfplot(sample, ax=axs[1])
    axs[1].set_title(f"ECDF - {dist_type} (size={sample_size})")
    axs[1].set_xlabel("Values")
    axs[1].set_ylabel("Cumulative Probability")

    # Вывод таблицы статистики
    plt.figtext(0.92, 0.5, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    plt.show()  # Ожидание закрытия окна перед продолжением


def check_sigma_rule(sample):
    """ Проверяет правило 3-х сигм (только для нормального распределения) """
    mean = np.mean(sample)
    std_dev = np.std(sample)

    within_1_sigma = np.sum((sample >= mean - std_dev) & (sample <= mean + std_dev)) / len(sample)
    within_2_sigma = np.sum((sample >= mean - 2 * std_dev) & (sample <= mean + 2 * std_dev)) / len(sample)
    within_3_sigma = np.sum((sample >= mean - 3 * std_dev) & (sample <= mean + 3 * std_dev)) / len(sample)

    print(f"📊 Rule of 3 Sigma for Normal Distribution:")
    print(f"  🔹 Within 1σ: {within_1_sigma * 100:.2f}%")
    print(f"  🔹 Within 2σ: {within_2_sigma * 100:.2f}%")
    print(f"  🔹 Within 3σ: {within_3_sigma * 100:.2f}%")


# Генерация данных
normal_samples, uniform_samples, binomial_samples = generate_samples()

# Объединяем все типы выборок в один словарь
distributions = {
    "Normal": normal_samples,
    "Uniform": uniform_samples,
    "Binomial": binomial_samples
}

# Цикл по распределениям и размерам выборок
for dist_name, samples in distributions.items():
    for sample_size, sample in samples.items():
        print(f"\n📌 {dist_name} Distribution, Sample Size: {sample_size}")
        plot_graphs(sample, sample_size, dist_name)  # Вывод графиков

        if dist_name == "Normal":
            check_sigma_rule(sample)  # Проверка правила 3 сигм
