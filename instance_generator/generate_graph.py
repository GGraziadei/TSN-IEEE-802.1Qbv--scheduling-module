
from argparse import ArgumentParser
import csv

parser = ArgumentParser()
parser.add_argument("-f", "--filename", dest="filename", help="Output filename.", required=True)

args = parser.parse_args()

# generate a graph from csv file
import matplotlib.pyplot as plt
import numpy as np

with open(f'instance_generator/{args.filename}.csv', 'r') as f:
    # read the csv file, header is the first line, data is the rest, comma separated
    reader = csv.reader(f)
    header = next(reader)
    data = list(reader)
    # per each row define a dictionary with the header as key
    data = [{header[i]: row[i] for i in range(len(header))} for row in data]

from os import path

if not path.exists(f'instance_generator/graphs/{args.filename}'):
    from os import makedirs
    makedirs(f'instance_generator/graphs/{args.filename}')

# plot the data
# x axis is the id
n_requests = [int(row['instance']) for row in data]
solving_time = [float(row['solving_time']) for row in data]
pre_processing_time = [float(row['pre_processing']) for row in data]

delays = [float(row['delay']) for row in data]
jitters = [float(row['jitter']) for row in data]

cumulative_delay = [float(row['cumulative_max_delay']) for row in data]
cumulative_jitter = [float(row['cumulative_max_jitter']) for row in data]


# get request per app 
app1_requests = [row for row in data if row['app'] == 'App1']
app1_n_requests = range(1, len(app1_requests) + 1)
app2_requests = [row for row in data if row['app'] == 'App2']
app2_n_requests = range(1, len(app2_requests) + 1)

app1_delay = [min(float(row['delay']),1*10**6) for row in app1_requests]
app2_delay = [float(row['delay']) for row in app2_requests]
app1_solving_time = [float(row['solving_time']) for row in app1_requests]
app2_solving_time = [float(row['solving_time']) for row in app2_requests]
app1_pre_processing_time = [float(row['pre_processing']) for row in app1_requests]
app2_pre_processing_time = [float(row['pre_processing']) for row in app2_requests]

app1_jitter = [float(row['jitter']) for row in app1_requests]
app2_jitter = [float(row['jitter']) for row in app2_requests]
app1_cumulative_delay = [min(float(row['cumulative_max_delay_app']),1*10**6) for row in app1_requests]
app2_cumulative_delay = [float(row['cumulative_max_delay_app']) for row in app2_requests]
app1_cumulative_jitter = [float(row['cumulative_max_jitter_app']) for row in app1_requests]
app2_cumulative_jitter = [float(row['cumulative_max_jitter_app']) for row in app2_requests]

# GRAPH 1 - solving time - 
figure, ax = plt.subplots(figsize=(7,7))

# solving time + interpolation
ax.set_title('Solving and pre-processing time')
ax.plot(app1_n_requests, app1_solving_time, label='App1 Solving time')
ax.plot(app2_n_requests, app2_solving_time, label='App2 Solving time')
ax.plot(app1_n_requests, app1_pre_processing_time, label='App1 Pre processing time')
ax.plot(app2_n_requests, app2_pre_processing_time, label='App2 Pre processing time')

# interpolate the data
app1_solving_time_linear = np.polyfit(app1_n_requests, app1_solving_time, 3)
app2_solving_time_linear = np.polyfit(app2_n_requests, app2_solving_time, 3)

ax.plot(app1_n_requests, np.polyval(app1_solving_time_linear, app1_n_requests), label='App1 Solving time cubic interpolation', linestyle='--')
ax.plot(app2_n_requests, np.polyval(app2_solving_time_linear, app2_n_requests), label='App2 Solving time cubic interpolation', linestyle='--')
ax.legend()
#ax[0].setxlabel('Number of requests')
#ax[0].setylabel('Time (s)')
#ax.set_yscale('log')
ax.set_xlabel('Number of requests')
ax.set_ylabel('Time (s)')

plt.savefig(f'instance_generator/graphs/{args.filename}/time.png')

#GRAPH 2 - cumulative delay/jitter - boxplot
figure, ax = plt.subplots(1,2, figsize=(15,5))

ax[0].plot(n_requests, cumulative_delay, label='cumulative max delay')
ax[0].plot(n_requests, cumulative_jitter, label='cumulative max jitter')
ax[0].legend()
ax[0].set_xlabel('Number of requests')
ax[0].set_ylabel('Time (ns)')
ax[0].set_title('Cumulative maximum delay and jitter of network')

ax[1].boxplot([app1_delay, app2_delay, app1_jitter, app2_jitter])
ax[1].set_xticks([1, 2, 3, 4], ['App1 delay', 'App2 delay', 'App1 jitter', 'App2 jitter'])
ax[1].set_ylabel('Delay (ns)')
ax[1].set_title('Delay and jitter of apps variance distribution')
plt.savefig(f'instance_generator/graphs/{args.filename}/cumulative.png')

# GRAPH 3 - KPI 
figure, ax = plt.subplots(2,2, figsize=(15,15))
ax[0,0].plot(app1_n_requests, app1_delay, label='App1 delay')
ax[0,0].plot(app1_n_requests, app1_jitter, label='App1 jitter')
ax[0,0].legend()
ax[0,0].set_xlabel('Number of requests')
ax[0,0].set_ylabel('Time (ns)')
ax[0,0].set_title('Delay and jitter of App1')

ax[1,0].plot(app1_n_requests, app1_cumulative_delay, label='App1 cumulative max delay')
ax[1,0].plot(app1_n_requests, app1_cumulative_jitter, label='App1 cumulative max jitter')
ax[1,0].legend()
ax[1,0].set_xlabel('Number of requests')
ax[1,0].set_ylabel('Time (ns)')
ax[1,0].set_title('Cumulative maximum delay and jitter of App1')

ax[0,1].plot(app2_n_requests, app2_delay, label='App2 delay')
ax[0,1].plot(app2_n_requests, app2_jitter, label='App2 jitter')
ax[0,1].legend()
ax[0,1].set_xlabel('Number of requests')
ax[0,1].set_ylabel('Time (ns)')
ax[0,1].set_title('Delay and jitter of App2')

ax[1,1].plot(app2_n_requests, app2_cumulative_delay, label='App2 cumulative max delay')
ax[1,1].plot(app2_n_requests, app2_cumulative_jitter, label='App2 cumulative max jitter')
ax[1,1].legend()
ax[1,1].set_xlabel('Number of requests')
ax[1,1].set_ylabel('Time (ns)')
ax[1,1].set_title('Cumulativ emaximum delay and jitter of App2')

plt.savefig(f'instance_generator/graphs/{args.filename}/kpi.png')