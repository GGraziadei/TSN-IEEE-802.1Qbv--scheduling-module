
from argparse import ArgumentParser
import csv

parser = ArgumentParser()
parser.add_argument("-f", "--filename", dest="filename", help="Output filename.", required=True)

args = parser.parse_args()

# generate a graph from csv file
import matplotlib.pyplot as plt
import numpy as np

with open(f'{args.filename}.csv', 'r') as f:
    # read the csv file, header is the first line, data is the rest, comma separated
    reader = csv.reader(f)
    header = next(reader)
    data = list(reader)
    # per each row define a dictionary with the header as key
    data = [{header[i]: row[i] for i in range(len(header))} for row in data]

from os import path

args.filename = args.filename.replace('/', '_')   

if not path.exists(f'instance_generator/graphs/{args.filename}'):
    from os import makedirs
    makedirs(f'instance_generator/graphs/{args.filename}')

app2_delay = []
app1_delay = []

for row in data:
    row['pre_processing'] = float(row['pre_processing']) 
    row['solving_time'] = float(row['solving_time'])
    row['delay'] = float(row['delay']) / 10**6
    row['jitter'] = float(row['jitter']) / 10**6
    row['cumulative_max_delay'] = float(row['cumulative_max_delay']) / 10**6
    row['cumulative_max_jitter'] = float(row['cumulative_max_jitter']) / 10**6
    row['cumulative_max_delay_app'] = float(row['cumulative_max_delay_app']) / 10**6
    row['cumulative_max_jitter_app'] = float(row['cumulative_max_jitter_app']) / 10**6

    if row['app'] == 'App_1':
        if row['cumulative_max_delay'] > 1:
            row['cumulative_max_delay'] = 1

        if row['delay'] > 1:
            row['delay'] = 1
            print('delay > 1')

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
app1_requests = [row for row in data if row['app'] == 'App_1']
app1_delay = [float(row['delay']) for row in app1_requests]
app1_n_requests = range(1, len(app1_requests) + 1)
app2_requests = [row for row in data if row['app'] == 'App_2']
app2_delay = [float(row['delay']) for row in app2_requests]
app2_n_requests = range(1, len(app2_requests) + 1)


app1_delay_mean = []
app1_delay_variance = []
for d in app1_delay:
    val = (sum(app1_delay_mean) + d) / ( 1 + len(app1_delay_mean))
    app1_delay_mean.append(val)


assert len(app1_delay_mean) == len(app1_delay)
assert len(app1_delay_mean) == len(app1_n_requests)

app2_delay_mean = []
for d in app2_delay:
    val = (sum(app2_delay_mean) + d) / ( 1 + len(app2_delay_mean))
    app2_delay_mean.append(val)
assert len(app2_delay_mean) == len(app2_n_requests), f'{len(app2_delay_mean)} != {len(app2_n_requests)}'
                           
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

app1_max_delay = 0 
app2_max_delay = 0
app1_max_jitter = 0
app2_max_jitter = 0
mean_delay_app1 = 0
mean_jitter_app1 = 0
mean_delay_app2 = 0
mean_jitter_app2 = 0
app1_count = 0
app2_count = 0

mean10_delay_app1 = 0
mean10_jitter_app1 = 0
mean10_delay_app2 = 0
mean10_jitter_app2 = 0

cumlative_requests = {}
for row in data:
    if row['app'] == 'App_1':
        app1_count += 1
        mean_delay_app1 += float(row['delay'])
        mean_jitter_app1 += float(row['jitter'])
        mean10_delay_app1 += float(row['delay'])
        mean10_jitter_app1 += float(row['jitter'])

        if app1_count % 10 == 0:
            mean10_delay_app1 = 0
            mean10_jitter_app1 = 0

        if float(row['delay']) > app1_max_delay:
            app1_max_delay = float(row['delay'])
        if float(row['jitter']) > app1_max_jitter:
            app1_max_jitter = float(row['jitter'])
        
    else:
        app2_count += 1
        mean_delay_app2 += float(row['delay'])
        mean_jitter_app2 += float(row['jitter'])

        mean10_delay_app2 += float(row['delay'])
        mean10_jitter_app2 += float(row['jitter'])

        if app2_count % 10 == 0:
            mean10_delay_app2 = 0
            mean10_jitter_app2 = 0
            
        if float(row['delay']) > app2_max_delay:
            app2_max_delay = float(row['delay'])
        if float(row['jitter']) > app2_max_jitter:
            app2_max_jitter = float(row['jitter'])

    id = int(row['instance'])

    cumlative_requests[id] = {
        'app1_delay': app1_max_delay,
        'app2_delay': app2_max_delay,
        'app1_jitter': app1_max_jitter,
        'app2_jitter': app2_max_jitter,
        'app1_mean_delay': mean_delay_app1 / app1_count if app1_count > 0 else 0,
        'app2_mean_delay': mean_delay_app2 / app2_count if app2_count > 0 else 0,
        'app1_mean_jitter': mean_jitter_app1 / app1_count if app1_count > 0 else 0,
        'app2_mean_jitter': mean_jitter_app2 / app2_count if app2_count > 0 else 0,
        'app1_mean10_delay': mean10_delay_app1 / (app1_count % 10) if app1_count % 10 > 0 else 0,
        'app2_mean10_delay': mean10_delay_app2 / (app2_count % 10) if app2_count % 10 > 0 else 0,
        'app1_mean10_jitter': mean10_jitter_app1 / (app1_count % 10) if app1_count % 10 > 0 else 0,
        'app2_mean10_jitter': mean10_jitter_app2 / (app2_count % 10) if app2_count % 10 > 0 else 0
    }

#export to excel 
import pandas as pd
# specify index column
df = pd.DataFrame(data)
df.set_index('instance', inplace=True)

# df remove columns
df.drop(columns=['cumulative_max_delay_app', 'cumulative_max_jitter_app'], inplace=True)
df['max_delay_app1'] = [cumlative_requests[r]['app1_delay'] for r in cumlative_requests]
df['max_jitter_app1'] = [cumlative_requests[r]['app1_jitter'] for r in cumlative_requests]

df['max_delay_app2'] = [cumlative_requests[r]['app2_delay'] for r in cumlative_requests]
df['max_jitter_app2'] = [cumlative_requests[r]['app2_jitter'] for r in cumlative_requests]

df['mean_delay_app1'] = [cumlative_requests[r]['app1_mean_delay'] for r in cumlative_requests]
df['mean_jitter_app1'] = [cumlative_requests[r]['app1_mean_jitter'] for r in cumlative_requests]
df['mean_delay_app2'] = [cumlative_requests[r]['app2_mean_delay'] for r in cumlative_requests]
df['mean_jitter_app2'] = [cumlative_requests[r]['app2_mean_jitter'] for r in cumlative_requests]

app1_filtered = df[df['app'] == 'App_1']
app2_filtered = df[df['app'] == 'App_2']
app1_filtered['mean100_delay_app1'] = app1_filtered['delay'].rolling(100).mean()
app2_filtered['mean100_delay_app2'] = app2_filtered['delay'].rolling(100).mean()
app1_filtered['mean100_jitter_app1'] = app1_filtered['jitter'].rolling(100).mean()
app2_filtered['mean100_jitter_app2'] = app2_filtered['jitter'].rolling(100).mean()

df = pd.merge(df, app1_filtered[['mean100_delay_app1', 'mean100_jitter_app1']], on='instance', how='left')
df = pd.merge(df, app2_filtered[['mean100_delay_app2', 'mean100_jitter_app2']], on='instance', how='left')

# if the value is nan, replace it with the previous value
df.fillna(method='ffill', inplace=True)
# if the value is nan, replace it with the next value
df.fillna(method='bfill', inplace=True)

# Pandas mean so far 
df.to_excel(f'instance_generator/graphs/{args.filename}/{args.filename}.xlsx')

# GRAPH 1 - solving time - 
figure, ax = plt.subplots(figsize=(7,7))

# solving time + interpolation
ax.set_title('Solving and pre-processing time')
ax.plot(app1_n_requests, app1_solving_time, label='App1 Solving time')
ax.plot(app2_n_requests, app2_solving_time, label='App2 Solving time')
ax.plot(app1_n_requests, app1_pre_processing_time, label='App1 Pre processing time')
ax.plot(app2_n_requests, app2_pre_processing_time, label='App2 Pre processing time')

assert len(app1_n_requests) == len(app1_solving_time)
assert len(app2_n_requests) == len(app2_solving_time)
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
plt.tight_layout()
plt.savefig(f'instance_generator/graphs/{args.filename}/time.png')


# GRAPH 2 - KPI 
figure, ax = plt.subplots(2,2, figsize=(15,15))
ax[0,0].plot(app1_n_requests, app1_delay, label='App1 delay')
ax[0,0].plot(app1_n_requests, app1_jitter, label='App1 jitter')
ax[0,0].legend()
ax[0,0].set_xlabel('Number of requests')
ax[0,0].set_ylabel('Time (ms)')
ax[0,0].set_title('Delay and jitter of App1')

ax[0,1].plot(app1_n_requests, app1_cumulative_delay, label='App1 max delay')
ax[0,1].plot(app1_n_requests, app1_filtered['mean100_delay_app1'], label='App1 mobile mean delay (window size 100)')
ax[0,1].plot(app1_n_requests, app1_filtered['mean_delay_app1'], label='App1 mean delay')

ax[0,1].plot(app1_n_requests, app1_cumulative_jitter, label='App1 max jitter')
ax[0,1].legend(bbox_to_anchor=(1, 0.3), loc='center right')
ax[0,1].set_xlabel('Number of requests')
ax[0,1].set_ylabel('Time (ms)')
ax[0,1].set_title('Cumulative maximum delay and jitter of App1')

ax[1,0].plot(app2_n_requests, app2_delay, label='App2 delay')
ax[1,0].plot(app2_n_requests, app2_jitter, label='App2 jitter')
ax[1,0].legend()
ax[1,0].set_xlabel('Number of requests')
ax[1,0].set_ylabel('Time (ms)')
ax[1,0].set_title('Delay and jitter of App2')

ax[1,1].plot(app2_n_requests, app2_cumulative_delay, label='App2 max delay')
ax[1,1].plot(app2_n_requests, app2_filtered['mean100_delay_app2'], label='App2 mobile mean delay (window size 100)')
ax[1,1].plot(app2_n_requests, app2_filtered['mean_delay_app2'], label='App2 mean delay')

ax[1,1].plot(app2_n_requests, app2_cumulative_jitter, label='App2 max jitter')
ax[1,1].legend()
ax[1,1].set_xlabel('Number of requests')
ax[1,1].set_ylabel('Time (ms)')
ax[1,1].set_title('Cumulativ emaximum delay and jitter of App2')
plt.tight_layout()
plt.savefig(f'instance_generator/graphs/{args.filename}/kpi.png')

#GRAPH 3 - cumulative delay/jitter - boxplot
figure, ax = plt.subplots(1,2, figsize=(15,5))

app1_delay = [cumlative_requests[id]['app1_delay'] for id in cumlative_requests]
app2_delay = [cumlative_requests[id]['app2_delay'] for id in cumlative_requests]
app1_jitter = [cumlative_requests[id]['app1_jitter'] for id in cumlative_requests]
app2_jitter = [cumlative_requests[id]['app2_jitter'] for id in cumlative_requests]

ax[0].plot(n_requests, app1_delay, label='App1 cumulative max delay')
ax[0].plot(n_requests, app2_delay, label='App2 cumulative max delay')
ax[0].plot(n_requests, app1_jitter, label='App1 cumulative max jitter')
ax[0].plot(n_requests, app2_jitter, label='App2 cumulative max jitter')
ax[0].legend()
ax[0].set_xlabel('Number of requests')
ax[0].set_ylabel('Time (ms)')
ax[0].set_title('Cumulative maximum delay and jitter of network')

ax[1].boxplot([app1_delay, app2_delay, app1_jitter, app2_jitter])
ax[1].set_xticks([1, 2, 3, 4], ['App1 delay', 'App2 delay', 'App1 jitter', 'App2 jitter'])
ax[1].set_ylabel('Delay (ms)')
ax[1].set_title('Delay and jitter of apps variance distribution')
plt.tight_layout()
plt.savefig(f'instance_generator/graphs/{args.filename}/cumulative.png')
