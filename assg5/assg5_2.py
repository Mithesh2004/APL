import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

xlim3 = [-10, 10]
ylim3 = [-10, 10]
# Learning rate is chosen low to prevent overflow
lr = 0.001
initial_vals = [-7,7] 

def f3(x, y):
    return x**4 - 16 * x**3 + 96 * x**2 - 256 * x + y**2 - 4 * y + 262


def df3_dx(x, y):
    return 4 * x**3 - 48 * x**2 + 192 * x - 256


def df3_dy(x, y):
    return 2 * y - 4


def grad_desc(cfunc, cfuncd_x, cfuncd_y, xlim, ylim, initial_vals, lr):
    
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
    
    
    for i in range(1000000):
        onestepderiv(i)
      
    #uncomemnt the below part and comment the above for loop for animation(it will taken long time to run)
    """ #Animation part:
    
    ani = FuncAnimation(
        fig, onestepderiv, frames=range(10000), interval=1, repeat=False
    )
    print("Making animated plot.....")
    ani.save('animation2.gif', writer='pillow', fps=5)
    print("Plot saved as animation2.gif.")
    """
    
    print(f"Best-x = {bestx}\nBest-y = {besty}\nBest-cost at best-x and best-y is = {cfunc(bestx, besty)}")

grad_desc(f3, df3_dx, df3_dy, xlim3, ylim3,initial_vals, lr)
