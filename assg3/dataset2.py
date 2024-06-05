import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

filename = "./a3-data/dataset2.txt"


# function for curve fitting
def sinfunc(x, a, b, c, f):
    return (
        a * np.sin(2 * np.pi * f * x)
        + b * np.sin(2 * np.pi * 3 * f * x)
        + c * np.sin(2 * np.pi * 5 * f * x)
    )


def plotLstSq(xarr, yarr, f):
    M = np.column_stack(
        [
            np.sin(2 * np.pi * f * xarr),
            np.sin(2 * np.pi * 3 * f * xarr),
            np.sin(2 * np.pi * 5 * f * xarr),
        ]
    )
    a, b, c = np.linalg.lstsq(M, yarr, rcond=False)[0]
    sqf_yarr = np.array(
        [
            a * np.sin(2 * np.pi * f * x)
            + b * np.sin(2 * np.pi * 3 * f * x)
            + c * np.sin(2 * np.pi * 5 * f * x)
            for x in xarr
        ]
    )
    plt.plot(
        xarr,
        sqf_yarr,
        color="yellow",
        label=f"Lstsq fitted: y = {a:.3f}sin({2*np.pi*f:.3f}t) + {b:.3f}sin({2*np.pi*3*f:.3f}t) + {c:.3f}sin({2*np.pi*5*f:.3f}t)",
    )
    print(
        f"The equation obtained in case of Lease Square fitting method is:\ny = {a}sin({2*np.pi*f}t) + {b}sin({2*np.pi*3*f}t) + {c}sin({2*np.pi*5*f}t)"
    )


def plotCurveFit(xarr, yarr, freq_est):
    params, covariance = curve_fit(sinfunc, xarr, yarr, p0=[5, 5, 5, freq_est])
    a, b, c, f = params
    cf_yarr = sinfunc(xarr, a, b, c, f)
    plt.plot(
        xarr,
        cf_yarr,
        color="red",
        label=f"Curve fitted: y = {a:.3f}sin({2*np.pi*f:.3f}t) + {b:.3f}sin({2*np.pi*3*f:.3f}t) + {c:.3f}sin({2*np.pi*5*f:.3f}t)",
    )
    print(
        f"The equation obtained in case of Curve fitting method is:\ny = {a}sin({2*np.pi*f}t) + {b}sin({2*np.pi*3*f}t) + {c}sin({2*np.pi*5*f}t)"
    )
    print(f"The frequency obtained from Curve fitting method is: {f} units")


def main():
    # extracting the data from the specified file
    data = np.loadtxt(filename)
    xarr, yarr = data[:, 0], data[:, 1]

    # The frquency is estimated by observing the periodicity of the graph plotted from the given values in dataset
    freq_est = 1 / 2.5

    plt.plot(xarr, yarr, label="Observed plot with noise")
    plotLstSq(xarr, yarr, freq_est)
    plotCurveFit(xarr, yarr, freq_est)

    plt.legend(loc="best")
    plt.xlabel("Time")
    plt.ylabel("Y-axis")
    plt.savefig("dataset2")


main()
