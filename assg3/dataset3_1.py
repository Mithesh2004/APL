import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

h, c, k = [6.62607015 * (10 ** (-34)), 299792458, 1.380649 * (10 ** (-23))]


def plankEqua(f, T):
    return 2 * h * (f**3) / ((c**2) * (np.exp(h * f / (k * T)) - 1))


def main():
    filename = "./a3-data/dataset3.txt"
    # extracting the data from the specified file
    data = np.loadtxt(filename)

    xarr, yarr = data[:, 0], data[:, 1]

    # Here I am giving p0 in such a way there is no overflow warning.I arrived this value by trial and error
    params, covariance = curve_fit(plankEqua, xarr, yarr, p0=[200])
    [T] = params
    cf_yarr = plankEqua(xarr, T)

    print(f"The temperature is {T} Kelvin")

    plt.plot(xarr, yarr, label="Observed plot with noise")
    plt.plot(xarr, cf_yarr, label="Best fitted plot")
    plt.legend(loc="best")
    plt.xlabel("Frequency(f)")
    plt.ylabel("B(f,T)")
    plt.savefig("dataset3_1")


main()
