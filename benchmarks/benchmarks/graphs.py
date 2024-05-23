import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline

instances_be = ['1', '2', '3', '4', '5', '6']
instances_t = ['1', '7', '8', '9', '10', '11']

solving_be_ilp = [0.79, 3.71, 10.65, 13.5, 29.32, 30.91]
preprocessing_be_ilp = [0.33, 0.33, 0.33, 0.33, 0.38, 0.38]
solving_t_ilp = [0.79,3.05, 7.76, 11.5, 40.84, 59.41]
preprocessing_t_ilp = [0.33, 0.35, 0.35, 0.35, 0.38, 0.38]

solving_be_heuristic = [0.09,0.77,0.09,0.1,0.15,0.15]
preprocessing_be_heuristic = [0.4,0.3,0.04,0.04,0.06,0.05]
solving_t_heuristic = [0.09, 0.09, 0.08, 0.09, 0.14, 0.16]
preprocessing_t_heuristic = [0.4, 0.03, 0.03, 0.04, 0.05, 0.06]

# GRAPH 1 SOLVING TIME
fig, ax = plt.subplots(2, 2, figsize=(10, 5))

ax[0][0].plot(instances_be, solving_be_ilp, label='Solving ILP')
ax[0][0].plot(instances_be, solving_be_heuristic, label='Solving Heuristic')
#ax[0].plot(instances_be, preprocessing_be_ilp, label='Pre-processing ILP')
#ax[0].plot(instances_be, preprocessing_be_heuristic, label='Pre-processing Heuristic')

ax[0][0].set_ylabel('Time (s)')
ax[0][0].set_xlabel('Instance id')

ax[0][0].set_title('Processing Time factor max troughput')
ax[0][0].legend()

ax[0][1].plot(instances_t, solving_t_ilp, label='Solving ILP')
ax[0][1].plot(instances_t, solving_t_heuristic, label='Solving Heuristic')
#ax[1].plot(instances_t, preprocessing_be_ilp, label='Pre-processing ILP')
#ax[1].plot(instances_t, preprocessing_be_heuristic, label='Pre-processing Heuristic')

ax[0][1].set_ylabel('Time (s)')
ax[0][1].set_xlabel('Instance id')

ax[0][1].set_title('Processing Time factor granularity')

ax[0][1].legend()

ax[1][0].plot(instances_be, preprocessing_be_ilp, label='Pre-processing ILP')
ax[1][0].plot(instances_be, preprocessing_be_heuristic, label='Pre-processing Heuristic')

ax[1][0].set_ylabel('Time (s)')
ax[1][0].set_xlabel('Instance id')

ax[1][0].set_title('Pre-processing Time factor max troughput')
ax[1][0].legend()

ax[1][1].plot(instances_t, preprocessing_t_ilp, label='Pre-processing ILP')
ax[1][1].plot(instances_t, preprocessing_t_heuristic, label='Pre-processing Heuristic')

ax[1][1].set_ylabel('Time (s)')
ax[1][1].set_xlabel('Instance id')

ax[1][1].set_title('Pre-processing Time factor granularity')
ax[1][1].legend()


plt.tight_layout()
plt.savefig('benchmarks/executionTime.png')

# GRAPH 2 - feasibility
feasibility_be_ilp = [1,1,1,1,1,1]
feasibility_t_ilp = [1,1,1,1,1,1]
feasibility_be_heuristic = [0.7,0.6,0.7,0.7,1,1]
feasibility_t_heuristic = [0.7,0.7,0.6,0.7,1,1]

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].plot(instances_be, feasibility_be_ilp, label='ILP')
ax[0].plot(instances_be, feasibility_be_heuristic, label='Heuristic')
# plot vertical line
ax[0].axvline(x='5', color='r', linestyle='--',  label='B_e > 80 * 10e6')
ax[0].set_ylabel('Feasibility ratio')
ax[0].set_xlabel('Instance id')
ax[0].set_title('Feasibility factor max troughput')

ax[0].legend()
ax[0].set_ylim(0, 1.3)

ax[1].plot(instances_t, feasibility_t_ilp, label='ILP')
ax[1].plot(instances_t, feasibility_t_heuristic, label='Heuristic')

ax[1].set_ylabel('Feasibility ratio')
ax[1].set_xlabel('Instance id')
ax[1].set_title('Feasibility factor granularity')
ax[1].axvline(x='10', color='r', linestyle='--', label='T > 8 * 10e3')
ax[1].legend()
ax[1].set_ylim(0, 1.3)

plt.tight_layout()
plt.savefig('benchmarks/feasibility.png')

# GRAPH 3 - KPI

jitter_be_ilp = [0,0,0,0,0,0]
jitter_t_ilp = [0,0,0,0,0,0]

jitter_be_heuristic = [0,0,80,0,32,42]
jitter_t_heuristic = [0,0,144,0,32,192]

delay_be_ilp = [1536,864,704,624,456,422]
delay_t_ilp = [1536,1728,2112,2496,3648,4224]

delay_be_heuristic = [3076,1540,916,1012,388,311]
delay_t_heuristic = [3076,3076,3268,3076,3076,3076]

delay_be_gap = [delay_be_heuristic[i]-delay_be_ilp[i] for i in range(6)]
delay_t_gap = [delay_t_heuristic[i]-delay_t_ilp[i] for i in range(6)]

fig,ax = plt.subplots(2, 2, figsize=(10, 5))

ax[0][0].plot(instances_be, jitter_be_ilp, label='ILP')
ax[0][0].plot(instances_be, jitter_be_heuristic, label='Heuristic')
ax[0][0].set_ylabel('Jitter (t.u.)')
ax[0][0].set_xlabel('Instance id')
ax[0][0].set_title('Max Jitter factor max troughput')
ax[0][0].legend()

ax[0][1].plot(instances_t, jitter_t_ilp, label='ILP')
ax[0][1].plot(instances_t, jitter_t_heuristic, label='Heuristic')
ax[0][1].set_ylabel('Jitter (t.u.)')
ax[0][1].set_xlabel('Instance id')
ax[0][1].set_title('Max Jitter factor granularity')
ax[0][1].legend()

ax[1][0].plot(instances_be, delay_be_ilp, label='ILP')
ax[1][0].plot(instances_be, delay_be_heuristic, label='Heuristic')
ax[1][0].plot(instances_be, delay_be_gap, label='Delay opt. Gap')
ax[1][0].set_ylabel('Delay (t.u.)')
ax[1][0].set_xlabel('Instance id')
ax[1][0].set_title('Max Delay factor max troughput')
ax[1][0].legend()

ax[1][1].plot(instances_t, delay_t_ilp, label='ILP')
ax[1][1].plot(instances_t, delay_t_heuristic, label='Heuristic')
ax[1][1].plot(instances_t, delay_t_gap, label='Delay opt. Gap')

ax[1][1].set_ylabel('Delay (t.u.)')
ax[1][1].set_xlabel('Instance id')
ax[1][1].set_title('Max Delay factor granularity')
ax[1][1].legend()

plt.tight_layout()
plt.savefig('benchmarks/KPI.png')

