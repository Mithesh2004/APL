import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


xlim4 = [-np.pi, np.pi]
ylim4 = xlim4.copy()
initial_vals = [-2, 0]
# Learning rate
lr = 0.1

def f4(x, y):
    return np.exp(-((x - y) ** 2)) * np.sin(y)


def df4_dx(x, y):
    return -2 * np.exp(-((x - y) ** 2)) * np.sin(y) * (x - y)


def df4_dy(x, y):
    return np.exp(-((x - y) ** 2)) * np.cos(y) + 2 * np.exp(-((x - y) ** 2)) * np.sin(
        y
    ) * (x - y)


def grad_desc(cfunc, cfuncd_x, cfuncd_y, xlim, ylim, initial_vals, lr):
    
    
    #randomly selecting 100 points with in the interval to plot the given function
    xbase = np.linspace(xlim[0], xlim[-1], 100)
    ybase = np.linspace(ylim[0], ylim[-1], 100)
    X, Y = np.meshgrid(xbase, ybase)
    Z = cfunc(X, Y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    
    bestx, besty = initial_vals[0] , initial_vals[-1]
    
    #initailising the plot for the animation
    xall, yall, zall = [], [], []
    lnall = ax.scatter([], [], [], c="red")
    lngood = ax.scatter([], [], [], c="green", marker="o", s=50)

    
    def onestepderiv(frame):
        nonlocal  bestx, besty
        
        #update bestx and besty using gradient descent function
        bestx = bestx - cfuncd_x(bestx, besty) * lr
        besty = besty - cfuncd_y(bestx, besty) * lr
        xall.append(bestx)
        yall.append(besty)
        zall.append(cfunc(bestx,besty))
        lngood._offsets3d = ([bestx], [besty], [cfunc(bestx,besty)])
        lnall._offsets3d = (xall, yall, zall)
    
    
    for i in range(1000):
        onestepderiv(i)
        
    #uncomemnt the below part and comment the above for-loop for animation(it will taken long time to run)
    """ #Animation part:
    
    ani = FuncAnimation(
        fig, onestepderiv, frames=range(1000), interval=1, repeat=False
    )
    print("Making animated plot.....")
    ani.save('animation2.gif', writer='pillow', fps=5)
    print("Plot saved as animation2.gif.")
    """
    
    print(f"Best-x = {bestx}\nBest-y = {besty}\nBest-cost at best-x and best-y is = {cfunc(bestx, besty)}")

grad_desc(f4, df4_dx, df4_dy, xlim4, ylim4,initial_vals, lr)
