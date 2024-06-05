import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# defining list for starting conidtions in the format [h,c,k,T]
p0 = [6.62607 * (10 ** (-34)), 299792458, 1.380649 * (10 ** (-23)), 4997]


def plankEqua(f, h, c, k, T):
    return 2 * h * (f**3) / ((c**2) * (np.exp(h * f / (k * T)) - 1))


def main():
    filename = "./a3-data/dataset3.txt"

    # extracting the data from the specified file
    data = np.loadtxt(filename)
    xarr, yarr = data[:, 0], data[:, 1]

    params, _ = curve_fit(plankEqua, xarr, yarr, p0=p0)
    h, c, k, T = params
    cf_yarr = plankEqua(xarr, h, c, k, T)
    print(
        f"Planks constant: {h} Js\nSpeed of light in vacuum: {c} m/s\nBoltzmann constant: {k} J/k\nTemperature = {T} k"
    )
    plt.plot(xarr, yarr, label="Observed plot with noise")
    plt.plot(xarr, cf_yarr, label="Bestfitted plot")
    plt.legend(loc="best")
    plt.xlabel("Frequency(f)")
    plt.ylabel("B(f,T)")
    plt.savefig("dataset3_2")

main()
