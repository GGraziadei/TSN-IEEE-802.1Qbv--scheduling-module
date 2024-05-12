import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the function
def func(E, T):
    return np.abs(E)**2 * (T )**2

# Generate sample data
E_values = np.linspace(-100, 100, 100)
T_values = np.linspace(-100, 1000000, 100)

E, T = np.meshgrid(E_values, T_values)
Z = func(E, T)

# Plot the function in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(E, T, Z, cmap='viridis')

ax.set_xlabel('E')
ax.set_ylabel('T')
ax.set_zlabel('Function Value')
ax.set_title('Graph of the Function in 3D')

plt.show()
