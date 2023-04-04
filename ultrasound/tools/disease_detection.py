import sys
sys.path.insert(1, './')

import us.forecasting.algos as fore
import pandas as pd
import matplotlib.pyplot as plt
from rich import print
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import numpy as np
from aquarel import load_theme


def compute_consommation(data, silo_name):
    deriv = savgol_filter(data[silo_name], 5, 2, 1)
    data["deriv"] = deriv
    return data


def remove_fills(data, silo_name):
    no_fill = []
    no_fill.append(data[silo_name].values[0])
    offset = 0
    for i in range(1, len(data)):
        current = data[silo_name].values[i]
        prev = data[silo_name].values[i - 1]
        diff = current - prev
        if diff > 0:
            offset -= diff
        no_fill.append(current + offset)
    data[f"{silo_name}-nofill"] = no_fill
    return data


def func3(x, a, b, c):
    return a * x**b + c


def func(x, a, b, c):
    return a + b * x**2 + c * x**3


def theoretical_consommation(data_path):
    def func2(x, a, b, c):
        return a * np.exp(b * x)

    df = pd.read_csv(data_path)
    popt, pcov = curve_fit(func, df["day"].values, df["sum_consom"].values / max(df["sum_consom"].values),
                           bounds=([0, -np.inf, -np.inf], [0.000000001, np.inf, np.inf]))
    print(popt)
    popt2, pcov2 = curve_fit(func, df["day"].values, df["sum_consom"].values, bounds=([0, -np.inf, 0], [0.000000001, np.inf, 0.00000000001]))
    # plt.plot(df["day"], df["sum_consom"] / max(df["sum_consom"].values), ".")
    # plt.plot(df["day"], func(df["day"], *popt), "r-")
    # plt.show()

    return df


if __name__ == "__main__":
    # data = pd.read_csv("data/disease_detection/1483+1485.csv", converters={"AcquisitionTime": pd.to_datetime})

    dfT = theoretical_consommation("data/disease_detection/chicken_consommation.csv")

    silo_name = "Avinor-1485"
    dfA = fore.read_load_cell_data(f"data/disease_detection/Data/{silo_name.split('-')[1]}.csv", f"{silo_name}A", "v3", "FeedRemaining_g")
    dfB = fore.read_load_cell_data(f"data/disease_detection/Data/{silo_name.split('-')[1]}.csv", f"{silo_name}B", "v3", "FeedRemaining_g")
    dfA[f"{silo_name}A"] /= 1000000
    dfB[f"{silo_name}B"] /= 1000000

    colsA = dfA.columns.tolist()
    colsA = colsA[-1:] + colsA[:-1]

    dfA = dfA[colsA]

    colsB = dfB.columns.tolist()
    colsB = colsB[-1:] + colsB[:-1]
    dfB = dfB[colsB]

    dfA = remove_fills(dfA, f"{silo_name}A")
    dfB = remove_fills(dfB, f"{silo_name}B")

    dfA = compute_consommation(dfA, f"{silo_name}A")
    dfA_2 = dfA.copy().diff().resample("1D", label="right").sum()
    dfB = compute_consommation(dfB, f"{silo_name}B")

    not_both_flat = fore.find_not_both_flat(dfA, dfB, min_len=48, debug=False)
    big_cycles = fore.find_big_cycles(not_both_flat, start_at="end" if "1485" in silo_name else "start", debug=False)
    # 1483
    if silo_name == "Avinor-1483":
        c = big_cycles.pop(2)
        big_cycles[1] = fore.Cycle(start=big_cycles[1].start, end=c.end)
        big_cycles[0] = fore.Cycle(start=big_cycles[0].start - 72, end=big_cycles[0].end)
        # big_cycles[2] = fore.Cycle(start=big_cycles[2].start - 72, end=big_cycles[2].end)
        nb_chickens = [41000, 43000, 43000]

    # 1485
    if silo_name == "Avinor-1485":
        print(big_cycles)
        big_cycles[1] = fore.Cycle(start=big_cycles[1].start - 48, end=big_cycles[1].end)
        big_cycles[2] = fore.Cycle(start=big_cycles[2].start - 24, end=big_cycles[2].end)
        big_cycles[3] = fore.Cycle(start=big_cycles[3].start - 96, end=big_cycles[3].end)
        big_cycles.pop(0)
        nb_chickens = [29900, 26000, 29000]

    dfC = dfA.copy()
    dfC[f"{silo_name}C-nofill"] = dfA[f"{silo_name}A-nofill"] + dfB[f"{silo_name}B-nofill"]

    def plot_cycles(cycles, ax):
        for cycle in cycles:
            ax.axvline(dfA.index[cycle.start], color="green", linestyle="--")
            ax.axvline(dfA.index[cycle.end], color="red", linestyle="--")

    fig, ax = plt.subplots(3 + 1)  # + len(big_cycles))
    fig.suptitle(f"Silo: {silo_name}", fontsize=16)
    ax[0].plot(dfA[f"{silo_name}A"])
    ax[0].plot(dfA[f"{silo_name}A-nofill"])
    plot_cycles(big_cycles, ax[0])
    xlim = ax[0].get_xlim()

    ax[1].plot(dfB[f"{silo_name}B"])
    ax[1].plot(dfB[f"{silo_name}B-nofill"])
    plot_cycles(big_cycles, ax[1])

    # ax[2].plot(dfC.index, np.array(not_both_flat) * 100)
    ax[2].plot(-dfC[f"{silo_name}C-nofill"])
    plot_cycles(big_cycles, ax[2])

    for i, cycle in enumerate(big_cycles):
        i = 0
        df_cycle = dfC.iloc[cycle.start:cycle.end].resample("1D", label="right").mean()
        plot_cycles(big_cycles, ax[3 + i])
        origin = -df_cycle[f"{silo_name}C-nofill"].iloc[0]
        df_cycle["per_chicken"] = (-df_cycle[f"{silo_name}C-nofill"] - origin).diff().fillna(0) * 1000000 / nb_chickens[i]
        ax[3 + i].plot(df_cycle.index, df_cycle["per_chicken"], '.', color="blue")  # (-df_cycle[f"{silo_name}C-nofill"] - origin), '.')
        ax[3 + i].set_xlim(xlim)

        dfT_cycle = dfT.iloc[:len(df_cycle)]
        popt, pcov = curve_fit(func, dfT_cycle["day"].values, dfT_cycle["sum_consom"].values / max(dfT_cycle["sum_consom"].values),
                               bounds=([0, -np.inf, 4.04e-6], [0.000000001, np.inf, 4.05e-6]))

        popt2, pcov2 = curve_fit(
            func,
            np.arange(0, len(df_cycle.index)),
            (-df_cycle[f"{silo_name}C-nofill"].values - origin) / max(-df_cycle[f"{silo_name}C-nofill"].values - origin),
            bounds=([0, -np.inf, 4.04e-6], [0.000000001, np.inf, 4.05e-6])
        )
        # ax[3 + i].plot(df_cycle.index, func(np.arange(0, len(df_cycle.index)), *popt), 'x')

        # ax2 = ax[3 + i].twinx()
        # ax[3 + i].plot(df_cycle.index, dfT_cycle["daily_consom"], "x")  # dfT_cycle["sum_consom"] / 1000 / 1000 * nb_chickens[i])

        ax[3 + i].plot(df_cycle.index, np.divide((dfT_cycle["daily_consom"].values - df_cycle["per_chicken"].values) * 100, dfT_cycle["daily_consom"].values,
                       out=np.zeros_like(df_cycle["per_chicken"].values), where=dfT_cycle["daily_consom"].values != 0), 'x', color="orange")
        ax[3 + i].plot(df_cycle.index, dfT_cycle["daily_consom"], "--", color="green")  # dfT_cycle["sum_consom"] / 1000 / 1000 * nb_chickens[i])
        ax[3 + i].axhline(0, color="gray")

        if i == 0 and "1485" in silo_name:
            ax[3 + i].axvline(np.datetime64('2022-11-27'), color="red")
        if i == 0 and "1483" in silo_name:
            ax[3 + i].axvline(np.datetime64('2022-10-24'), color="red")

    plt.show()

    import matplotlib

    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    matplotlib.rc('font', size=SMALL_SIZE)          # controls default text sizes
    matplotlib.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    matplotlib.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    matplotlib.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    matplotlib.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    matplotlib.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    matplotlib.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.style.use('seaborn-darkgrid')
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    f, (a0, a1, a2) = plt.subplots(3, 1, height_ratios=[1, 2, 2])
    a0.plot(dfA[f"{silo_name}A"], label=f"{silo_name}A")
    a0.plot(dfB[f"{silo_name}B"] - dfB[f"{silo_name}B"].max(), label=f"{silo_name}B")
    a0.set_ylabel("Quantité de grains [T]")
    a0.legend(loc="upper left")
    plot_cycles(big_cycles, a0)

    # plt.plot(-dfC[f"{silo_name}C-nofill"])
    if "1485" in silo_name:
        big_cycles[2] = fore.Cycle(start=big_cycles[2].start, end=big_cycles[2].end - 48)
    plot_cycles(big_cycles, a1)
    big_cycles
    for i, cycle in enumerate(big_cycles, start=1):
        a1.plot(dfC.index[cycle.start:cycle.end], -dfC[f"{silo_name}C-nofill"]
                .iloc[cycle.start:cycle.end] - (-dfC[f"{silo_name}C-nofill"].iloc[cycle.start]), label=f"Cycle #{i}")
    a1.legend(loc="upper left")
    a1.set_ylabel("Quantité de grains\nconsommés par cycle [T]")
    a1.set_xlim(xlim)

    plot_cycles(big_cycles, a2)

    a2_copy = a2.twinx()
    a2_copy.set_ylim([-100, 100])
    a2_copy.grid(False)
    a2_copy.set_ylabel("Différence avec la théorie [%]")
    a2_copy.axhline(0, color="lightgray", linestyle="--")
    for i, cycle in enumerate(big_cycles):
        df_cycle = dfC.iloc[cycle.start:cycle.end].resample("1D", label="right").mean()
        origin = -df_cycle[f"{silo_name}C-nofill"].iloc[0]
        df_cycle["per_chicken"] = (-df_cycle[f"{silo_name}C-nofill"] - origin).diff().fillna(0) * 1000000 / nb_chickens[i]

        a2.plot(df_cycle["per_chicken"], '.', label=f"Cycle #{i+1}")
        a2.set_xlim(xlim)
        dfT_cycle = dfT.iloc[:len(df_cycle)]
        if i == 2:
            label = "Théorique"
        else:
            label = None
        a2.plot(df_cycle.index, dfT_cycle["daily_consom"], color=colors[4], label=label)

        blu = np.divide((dfT_cycle["daily_consom"].values - df_cycle["per_chicken"].values) * 100, dfT_cycle["daily_consom"].values,
                        out=np.zeros_like(df_cycle["per_chicken"].values), where=dfT_cycle["daily_consom"].values != 0)
        a2_copy.plot(df_cycle.index, blu, "x", color="lightgray")

    a2.legend(loc="upper left")
    a2.set_ylabel("Quantité de grains\nconsommés par poulet [g/jour]")
    a2.set_xlabel("Date")
    plt.show()
