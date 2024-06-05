import numpy as np
import matplotlib.pyplot as plt


# Function to calculate the total distance of a given order of cities
def distance(cities, cityorder):
    num_cities = len(cities)
    totaldistance = 0
    # Calculate the distance between each pair of consecutive cities
    for i in range(num_cities):
        totaldistance += np.sqrt(
            (cities[cityorder[i]][0] - cities[cityorder[(i + 1) % num_cities]][0]) ** 2
            + (cities[cityorder[i]][1] - cities[cityorder[(i + 1) % num_cities]][1])
            ** 2
        )
    return totaldistance


# Function to solve the Traveling Salesman Problem using Simulated Annealing
def tsp(cities):
    num_cities = len(cities)
    # Generate a random initial order of cities
    order = np.random.permutation(num_cities)

    # Function to generate a neighbor order by swapping two cities
    def get_neighbor(current_order):
        neighbor = current_order.copy()
        i, j = np.random.randint(num_cities, size=2)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        return neighbor

    # Function to calculate the acceptance probability
    def acceptance_probability(current_cost, neighbor_cost, temperature):
        if neighbor_cost < current_cost:
            return 1
        else:
            return np.exp((current_cost - neighbor_cost) / temperature)

    current_order = order
    best_order = current_order
    current_cost = distance(cities, current_order)
    best_cost = current_cost
    initial_temperature = 1000
    cooling_rate = 0.995
    iterations = 100000

    # Main loop of simulated annealing
    for i in range(iterations):
        temperature = initial_temperature * (cooling_rate**i)
        neighbor_order = get_neighbor(current_order)
        neighbor_cost = distance(cities, neighbor_order)

        # If the new order is accepted, update the current order and cost
        if (
            acceptance_probability(current_cost, neighbor_cost, temperature)
            > np.random.rand()
        ):
            current_order = neighbor_order
            current_cost = neighbor_cost

        # If the new order is better than the best so far, update the best order and cost
        if current_cost < best_cost:
            best_order = current_order
            best_cost = current_cost

    return best_order.tolist()


# Read input from file and convert it to a list of tuples representing cities' coordinates
with open("cities.txt", "r") as file:
    num_cities = int(file.readline())
    cities = [tuple(map(float, line.strip().split())) for line in file]

# Solve the TSP and get the best order of cities
bestorder = tsp(cities)
# Calculate the total distance of the best order
bestcost = distance(cities, bestorder)

# Calculate the percentage improvement over a random order
random_order = list(range(len(cities)))
random_cost = distance(cities, random_order)
percent_imp = ((random_cost - bestcost) / random_cost) * 100

# Print the results
print(
    f"The Optimized Order is {bestorder} with distance = {bestcost:.2f}. It has {percent_imp:.2f}% improvement from the starting order {random_order}."
)

# Plot all cities as blue dots
plt.scatter(*zip(*cities), c="b", label="Cities")

# Mark the starting city as a green dot
plt.scatter(*cities[bestorder[0]], c="g", label="Starting City")

# Plot red lines between consecutive cities in the best order
for i in range(len(bestorder) - 1):
    plt.plot(*zip(cities[bestorder[i]], cities[bestorder[i + 1]]), c="r")

# Join the first and last city in the best order with a red line to complete the loop
plt.plot(*zip(cities[bestorder[-1]], cities[bestorder[0]]), c="r", label="path")

plt.legend()
plt.savefig("assg6")
