import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat
from scipy.optimize import curve_fit
import yfinance as yf
import pandas as pd
import os.path as op
from metalog import metalog

def main():
    stock_name, price_data, time_data = get_user_input(False)
    plot_graph(stock_name, price_data, time_data)


def func(x, a, b):
    return a * np.exp(b * x)


def lin_func(x, a, b):
    return a + b * x


def weighted_var(sample, weights, mean):
    square_sum = np.sum(np.square(sample - mean * weights))
    total_weight = np.sum(weights)
    sample_len = len(sample)
    correction = sample_len / (sample_len - 1.5)
    output = square_sum / total_weight * correction
    return output


def sp500_lognormal():
    sp_data = pd.read_csv("s&p500.csv", index_col=0)
    sp_data = sp_data["Real Price"]
    times = pd.to_datetime(sp_data.index)
    values = sp_data.values
    log_deltas = np.log(values[1:] / values[:-1])
    weights = (times[1:] - times[:-1]).days
    T = (times[-1] - times[0]).days
    log_mean = np.log(values[-1] / values[0]) / T
    log_var = weighted_var(log_deltas, weights, log_mean)
    log_std = np.sqrt(log_var)
    return log_mean, log_var, log_std


def get_user_input(real_input=True):
    if not real_input:
        stock_name = "MSFT"
        adj_stock = stock_name.replace(".", "_")
        data_file = f"{adj_stock}.csv"
        start = "1970-01-01"
        today = str(np.datetime64('today'))
        end = today

        stock_data = yf.download(stock_name, start="1970-01-01", end=today)
        stock_data.to_csv(data_file)
    else:
        stock_name = input("Stock Ticker: ")
        adj_stock = stock_name.replace(".", "_")
        data_file = f"{adj_stock}.csv"
        start = input("Start date (YYYY-MM-DD): ")
        if not start:
            start = "1970-01-01"

        today = str(np.datetime64('today'))
        end = input("End date (YYYY-MM-DD): ")
        if not end:
            end = today
        file_exists = op.isfile(data_file)
        reset = False
        ignore = False
        if file_exists:
            choice = int(input("File Found! Reset? [0]No [1]Yes [2]Ignore file: "))
            if choice == 1:
                reset = True
            elif choice == 2:
                ignore = True

        if not file_exists or reset:
            print(data_file, "created/reset")
            stock_data = yf.download(stock_name, start="1970-01-01", end=today)
            stock_data.to_csv(data_file)
        elif ignore:
            stock_data = yf.download(stock_name, start="1970-01-01", end=today)
        else:
            stock_data = pd.read_csv(data_file, index_col=0)

    stock_data = stock_data.truncate(before=start, after=end)

    output = stock_name, np.mean(stock_data.values[:, 0:4], axis=1), pd.to_datetime(stock_data.index)
    return output


def print_data(price_data, time_data):
    log_mean, log_std, log_deltas, weights, z = graph_aux(price_data, time_data)
    y = 365.2425
    log_mean_0, _, _ = sp500_lognormal()
    p_value = 1 - stat.lognorm.cdf(np.exp(log_mean - log_mean_0), s=log_std)
    confidence_interval = np.expm1(log_mean + np.array([-1, 1]) * z * log_std)
    growth = np.expm1(log_mean * y)
    print(
        f"Data:"
        f"\nlog_mean = {100 * log_mean}%;"
        f"\nlog_std = {100 * log_std}%;"
        f"\nz-value = {z};"
	f"\np-value = {p_value}"
        f"\n\n"
        f"Important Variables:"
        f"\nannual outperfomance = {y * 100 * (log_mean - log_mean_0)}%;"
        f"\nperfomance anomaly = {(1 - 2*p_value)}; "
        f"\nusual daily variation = [{100 * confidence_interval[0]}%;  {100 * confidence_interval[1]}%];"
        f"\nannual growth = {100 * growth}%"
    )


def plot_graph(stock_name, price_data, time_data, fake=False):
    print_data(price_data, time_data)
    data = graph_aux(price_data, time_data)
    start, end = time_data[0].date(), time_data[-1].date()
    if not fake:
        log_mean, log_std, log_deltas, weights, z = data
        title = f"{stock_name} data: {start} to {end}"
    else:
        price_data = fake_price_data(data, price_data, time_data)
        title = f" Fake {stock_name} data: {start} to {end}"
        log_mean, log_std, log_deltas, weights, z = graph_aux(price_data, time_data)

    fig, ax = plt.subplots(2, 2, layout="constrained")
    fig.suptitle(title)
    ran = max(np.min(log_deltas), log_mean-z*log_std), min(np.max(log_deltas), log_mean+z*log_std)
    ran1, ran2 = ran

    ax[0, 0].set_title("Histogram of log-deltas")
    n = 7
    bin_count = int(np.sqrt(len(log_deltas)))
    x = np.linspace(ran1, ran2, bin_count)
    mlog_par = metalog.fit(x=log_deltas, boundedness='u', term_limit=n)
    ml_y = metalog.d(mlog_par, x, n)
    ax[0, 0].hist(log_deltas, bins=bin_count, density=True, range=ran, alpha=0.5)
    ax[0, 0].plot(x, ml_y, "C0")
    y = stat.norm.pdf(x, loc=log_mean, scale=log_std)
    ax[0, 0].plot(x, y)

    ax[1, 0].set_title("Normalized log-deltas")
    sample_size = len(time_data) - 1
    black_centerline = np.full(sample_size, log_mean)
    low_redline, high_redline = black_centerline - z * log_std, black_centerline + z * log_std
    low_blueline, high_blueline = black_centerline - 2 * log_std, black_centerline + 2 * log_std
    t = time_data[1:]
    ax[1, 0].plot(t, log_deltas / weights)
    ax[1, 0].plot(t, low_redline, "r")
    ax[1, 0].plot(t, black_centerline, "k")
    ax[1, 0].plot(t, high_redline, "r")
    ax[1, 0].plot(t, low_blueline, "c")
    ax[1, 0].plot(t, high_blueline, "c")

    time_values = (time_data - time_data[0]).days
    popt, pcov = curve_fit(lin_func, time_values, np.log(price_data))
    pval = np.sqrt(np.diag(pcov))
    a_std, b_std = pval[0], pval[1]

    a, b = popt
    a_low, a_high = a - z * a_std, a + z * a_std
    b_low, b_high = b - z * b_std, b + z * b_std
    ax[1, 1].set_title(f"Log-Price")
    ax[1, 1].plot(time_data, np.log(price_data))
    ax[1, 1].plot(time_data, a + b * time_values, "C1")
    y1 = a_low + b_low * time_values
    y2 = a_high + b_high * time_values
    ax[1, 1].fill_between(time_data, y1, y2, color= "C1", alpha=0.1)
    a_low, a, a_high = np.exp(a_low), np.exp(a), np.exp(a_high)
    # a = np.exp(a)
    ax[0, 1].set_title(f"Price")
    ax[0, 1].plot(time_data, price_data)
    ax[0, 1].plot(time_data, a * np.exp(b * time_values), "C1")
    y1 = np.exp(y1)
    y2 = np.exp(y2)
    ax[0, 1].fill_between(time_data, y1, y2, color= "C1", alpha=0.2)
    plt.show()


def fake_price_data(data, price_data, time_data):
    log_mean, log_std, log_deltas, weights, z = data
    fake_deltas = np.random.default_rng(0).normal(log_mean * weights, log_std * np.sqrt(weights), len(time_data) - 1)
    fake_cumulative_deltas = np.append(0, np.cumsum(fake_deltas))
    output = price_data[0] * np.exp(fake_cumulative_deltas)  # Fake
    return output


def graph_aux(price_data, time_data):
    data_interval_days = (time_data[-1] - time_data[0]).days

    log_mean = np.log(price_data[-1] / price_data[0]) / data_interval_days
    log_deltas = np.log(price_data[1:] / price_data[0:-1])
    weights = (time_data[1:] - time_data[:-1]).days
    log_std = np.sqrt(weighted_var(log_deltas, weights, log_mean))
    z = stat.norm.ppf(1 - 0.05 / len(weights[weights < np.max(weights)]))  # p = 0.1/N

    output = log_mean, log_std, log_deltas, weights, z
    return output


if __name__ == '__main__':
    main()
