# draw a graph
from matplotlib import pyplot as plt
import numpy as np


x = [1,2,3,4,5,6,7,8,9,10]
delays = [42,29,19,18,20,19,39,23,19,18]

plt.plot(x, delays, label='delay', linewidth=2)
plt.xlabel('iteration')
plt.ylabel('delay (us)')
plt.title('Heuristic balancing and minimization of delay') 



# add line at 29
max_treshold = [42,42,42,42,42,42,42,42,42,42]
min_treshold = [42,29,19,18,18,18,18,18,18,18]

points = [5,6,7,8,9,10]
points_delays = [20,19,39,23,19,18]
plt.scatter(points, points_delays, label='Free', color='green')

points = [2,3,4]
points_delays = [29,19,18]
delta_j = [13,10,1]
plt.scatter(points, points_delays, label='Non-free', color='red')
for i, txt in enumerate(delta_j):
    plt.annotate(f"+{txt}", (points[i], points_delays[i]))

plt.step(x, max_treshold, label='max delay threshold', linestyle='--', linewidth=3)
plt.step(x, min_treshold, label='min delay threshold', linestyle='--', linewidth=3)


#color the area between the two lines

# y scale 400
plt.ylim(10, 50)

plt.legend()
plt.show()  

# print variance of the delay
variance = np.var(delays)
print(f"Variance of the delay: {variance}")