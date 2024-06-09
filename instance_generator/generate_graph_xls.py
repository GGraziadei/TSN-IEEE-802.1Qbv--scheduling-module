
from argparse import ArgumentParser
import csv
import os

import pandas as pd

parser = ArgumentParser()
parser.add_argument("-f", "--filename", dest="filename", help="Output filename.", required=True)

args = parser.parse_args()

# generate a graph from csv file
import matplotlib.pyplot as plt
import numpy as np

if args.filename is None:
    print("Filename is required")
    exit()

if not args.filename.endswith(".xlsx"):
    print("Filename must be a xlsx file")
    exit()

df = pd.read_excel(args.filename)


n_requests = range(1, len(df)+1)

app1_requests = df[ df['app'] == 'App_1']
n_requests_app1 = range(1, len(app1_requests)+1)
app2_requests = df[ df['app'] == 'App_2']
n_requests_app2 = range(1, len(app2_requests)+1)

# GRAPH 2 - KPI 
figure, ax = plt.subplots(2,2, figsize=(15,15))
ax[0,0].plot(n_requests_app1, app1_requests['delay'], label='App1 delay')
ax[0,0].plot(n_requests_app1, app1_requests['jitter'], label='App1 jitter')
ax[0,0].legend()
ax[0,0].set_xlabel('Number of requests')
ax[0,0].set_ylabel('Time (ms)')
ax[0,0].set_title('Delay and jitter of App1')

ax[0,1].plot(n_requests_app1, app1_requests['max_delay_app1'], label='App1 cumulative max delay')
#ax[0,1].plot(app1_n_requests, app1_delay_mean, label='App1 cumulative mean delay')

ax[0,1].plot(n_requests_app1, app1_requests['max_jitter_app1'], label='App1 cumulative max jitter')
ax[0,1].legend()
ax[0,1].set_xlabel('Number of requests')
ax[0,1].set_ylabel('Time (ms)')
ax[0,1].set_title('Cumulative maximum delay and jitter of App1')

ax[1,0].plot(n_requests_app2, app2_requests['delay'], label='App2 delay')
ax[1,0].plot(n_requests_app2, app2_requests['jitter'], label='App2 jitter')
ax[1,0].legend()
ax[1,0].set_xlabel('Number of requests')
ax[1,0].set_ylabel('Time (ms)')
ax[1,0].set_title('Delay and jitter of App2')

ax[1,1].plot(n_requests_app2, app2_requests['max_delay_app2'], label='App2 cumulative max delay')
#ax[1,1].plot(app2_n_requests, app2_delay_mean, label='App2 cumulative mean delay')

ax[1,1].plot(n_requests_app2, app2_requests['max_jitter_app2'], label='App2 cumulative max jitter')
ax[1,1].legend()
ax[1,1].set_xlabel('Number of requests')
ax[1,1].set_ylabel('Time (ms)')
ax[1,1].set_title('Cumulativ emaximum delay and jitter of App2')
plt.tight_layout()
#extract the folder name
folder_name = os.path.dirname(args.filename)
plt.savefig(f'{folder_name}/kpi.png')

