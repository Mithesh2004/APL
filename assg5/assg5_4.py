# Set up the imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

xlim = [0, 2 * np.pi]
# Learning rate
lr = 0.1
initialx = 3


def f5(x):
    return np.cos(x) ** 4 - np.sin(x) ** 3 - 4 * np.sin(x) ** 2 + np.cos(x) + 1


def grad_desc(cfunc, xrange, initialx, lr):
    def cfuncd(x):
        h = 10 ** (-7)
        return (cfunc(x + h) - cfunc(x)) / h

    bestx = initialx

    xbase = np.linspace(xrange[0], xrange[-1], 100)
    ybase = cfunc(xbase)

    fig, ax = plt.subplots()
    ax.plot(xbase, ybase)
    ax.set_xlabel("x")
    ax.set_ylabel("f5(x)")
    xall, yall = [], []
    (lnall,) = ax.plot([], [], "ro-")
    (lngood,) = ax.plot([], [], "go", markersize=10)

    def onestepderiv(frame):
        nonlocal bestx, lr
        xall.append(bestx)
        yall.append(cfunc(bestx))
        x = bestx - cfuncd(bestx) * lr
        bestx = x
        y = cfunc(x)
        lngood.set_data(x, y)
        lnall.set_data(xall, yall)

    ani = FuncAnimation(fig, onestepderiv, frames=range(100), interval=10, repeat=False)

    print("Making animated plot.....")
    ani.save("animation4.gif", writer="pillow", fps=1)
    print("Plot saves as animation4.gif.")
    print(f"Best-x is = {bestx}\ny at best-x is = {cfunc(bestx)}")


grad_desc(f5, xlim, initialx, lr)
