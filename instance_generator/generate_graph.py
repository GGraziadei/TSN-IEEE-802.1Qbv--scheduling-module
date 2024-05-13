
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

# plot the data
# x axis is the id
n_requests = [int(row['instance']) for row in data]
solving_time = [float(row['solving_time']) for row in data]
pre_processing_time = [float(row['pre_processing']) for row in data]

fig, ax = plt.subplots()
ax.plot(n_requests, solving_time, label='solving time')
ax.plot(n_requests, pre_processing_time, label='pre processing time')
ax.legend()


plt.xlabel('Number of requests')
plt.ylabel('Time (s)')

plt.show()

delays = [float(row['delay']) for row in data]
jitters = [float(row['jitter']) for row in data]

cumulative_delay = [float(row['cumulative_max_delay']) for row in data]
cumulative_jitter = [float(row['cumulative_max_jitter']) for row in data]

fig, ax = plt.subplots()
ax.plot(n_requests, delays, label='delay')
ax.plot(n_requests, jitters, label='jitter')
ax.legend()

plt.xlabel('Number of requests')
plt.ylabel('Time (us)')
plt.show()

fig, ax = plt.subplots()
ax.plot(n_requests, cumulative_delay, label='cumulative max delay')
ax.plot(n_requests, cumulative_jitter, label='cumulative max jitter')
ax.legend()

plt.xlabel('Number of requests')
plt.ylabel('Time (us)')
plt.show()

# get request per app 
app1_requests = [row for row in data if row['app'] == 'App1']
app1_n_requests = range(1, len(app1_requests) + 1)
app2_requests = [row for row in data if row['app'] == 'App2']
app2_n_requests = range(1, len(app2_requests) + 1)

app1_delay = [float(row['delay']) for row in app1_requests]
app2_delay = [float(row['delay']) for row in app2_requests]
app1_solving_time = [float(row['solving_time']) for row in app1_requests]
app2_solving_time = [float(row['solving_time']) for row in app2_requests]
app1_pre_processing_time = [float(row['pre_processing']) for row in app1_requests]
app2_pre_processing_time = [float(row['pre_processing']) for row in app2_requests]

app1_jitter = [float(row['jitter']) for row in app1_requests]
app2_jitter = [float(row['jitter']) for row in app2_requests]
app1_cumulative_delay = [float(row['cumulative_max_delay_app']) for row in app1_requests]
app2_cumulative_delay = [float(row['cumulative_max_delay_app']) for row in app2_requests]
app1_cumulative_jitter = [float(row['cumulative_max_jitter_app']) for row in app1_requests]
app2_cumulative_jitter = [float(row['cumulative_max_jitter_app']) for row in app2_requests]

fig, ax = plt.subplots()
ax.plot(app1_n_requests, app1_delay, label='App1 delay')
ax.plot(app1_n_requests, app1_jitter, label='App1 jitter')
plt.legend()

plt.xlabel('Number of requests')
plt.ylabel('Time (us)')
plt.show()

fig, ax = plt.subplots()
ax.plot(app2_n_requests, app2_delay, label='App2 delay')
ax.plot(app2_n_requests, app2_jitter, label='App2 jitter')
plt.legend()

plt.xlabel('Number of requests')
plt.ylabel('Time (us)')
plt.show()

fig, ax = plt.subplots()
ax.plot(app1_n_requests, app1_cumulative_delay, label='App1 cumulative max delay')
ax.plot(app1_n_requests, app1_cumulative_jitter, label='App1 cumulative max jitter')
plt.legend()

plt.xlabel('Number of requests')
plt.ylabel('Time (us)')

plt.show()

fig, ax = plt.subplots()
ax.plot(app2_n_requests, app2_cumulative_delay, label='App2 cumulative max delay')
ax.plot(app2_n_requests, app2_cumulative_jitter, label='App2 cumulative max jitter')
plt.legend()

plt.xlabel('Number of requests')
plt.ylabel('Time (us)')
plt.show()


# boxplot with solving time
fig, ax = plt.subplots()
ax.boxplot([app1_solving_time, app2_solving_time, app1_pre_processing_time, app2_pre_processing_time])
plt.xticks([1, 2,3,4], ['App1 Solving', 'App2 Solving', 'App1 Pre Processing', 'App2 Pre Processing'])
plt.ylabel('Solving time (s)')
plt.show()

# solving time per app
fig, ax = plt.subplots()
ax.plot(app1_n_requests, app1_solving_time, label='App1 Solving time')
ax.plot(app2_n_requests, app2_solving_time, label='App2 Solving time')
plt.legend()

plt.xlabel('Number of requests')
plt.ylabel('Time (s)')
plt.show()
