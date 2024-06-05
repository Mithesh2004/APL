import numpy as np
import matplotlib.pyplot as plt

filename = "./a3-data/dataset1.txt"

# extracting the x and y points from the dataset
data = np.loadtxt(filename)
xarr, yarr = data[:, 0], data[:, 1]

# building M matrix with x points in one column and 1's in another column
M = np.column_stack([xarr, np.ones(len(xarr))])
# estimating the slope and intercept
m, c = np.linalg.lstsq(M, yarr, rcond=False)[0]

# finding the y coords from calculated m and c
fitted_yarr = np.array([m * x + c for x in xarr])

err = np.array([y1 - y2 for y1, y2 in zip(fitted_yarr, yarr)])
std_dev = np.std(err, axis=0)

plt.plot(xarr, yarr, label="Observed plot with noise")
plt.plot(xarr, fitted_yarr, label=f"Best fitted line: y = {m}x + {c}")

# plotting error bar for one in each 25 points
plt.errorbar(xarr[::25], yarr[::25], yerr=std_dev, fmt="o", label="Error Bar")
plt.legend(loc="best")
plt.xlabel("x-axis")
plt.ylabel("y-axis")

plt.savefig("dataset1")
