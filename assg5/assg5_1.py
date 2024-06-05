# Set up the imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Learning rate 
lr = 0.1
xlim=[-5,5]
initialx = 4
def f1(x):
    return x ** 2 + 3 * x + 8

def grad_desc(cfunc,xrange,initialx,lr):
    
    
    #define the derivative function
    def cfuncd(x):
        h= 10**(-7) #choosing very low value for h so that it tends to 0
        return (cfunc(x + h) - cfunc(x)) /  h
    
    #randomy choose 100 inputs from the given interval
    xbase = np.linspace(xrange[0], xrange[-1], 100)
    ybase = cfunc(xbase)
    
     
    # Create a plot to visualize the function
    fig, ax = plt.subplots()
    ax.plot(xbase, ybase)
    ax.set_xlabel("x")
    ax.set_ylabel("f1(x)")
    
    xall, yall = [], []
    lnall,  = ax.plot([], [], 'ro-')
    lngood, = ax.plot([], [], 'go', markersize=10)
    
    bestx = initialx
    
    def onestepderiv(frame):
        
        nonlocal bestx
        
        xall.append(bestx)
        yall.append(cfunc(bestx)) #append the best cost
        
        bestx = bestx - cfuncd(bestx) * lr  
        lngood.set_data(bestx, cfunc(bestx))
        lnall.set_data(xall, yall)
    
    
    ani= FuncAnimation(fig, onestepderiv, frames=range(100), interval=1, repeat=False)
    print("Making animated plot.....")
    ani.save('animation1.gif', writer='pillow', fps=2)
    print("Plot saved as animation1.gif.")
    print(f"Best-x is = {bestx}\nBest-cost at best-x is = {cfunc(bestx)}")


grad_desc(f1,xlim,initialx,lr)

