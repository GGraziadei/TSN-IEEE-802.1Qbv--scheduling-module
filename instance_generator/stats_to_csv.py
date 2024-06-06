# load a pckl file and write stats in csv

from enum import Enum
import json
from math import ceil
from os import path
import pickle
import csv

from argparse import ArgumentParser
# generate a graph
import matplotlib.pyplot as plt
import numpy as np

parser = ArgumentParser()
parser.add_argument("-f", "--filename", dest="filename", help="Output filename.", required=True)
parser.add_argument("-r", "--requests", dest="requests", help="Requests filename.", required=False)
parser.add_argument("-n", "--network", dest="network", help="Network filename.", required=False)

args = parser.parse_args()

if not args.requests:
    args.requests = path.dirname(args.filename) + "/requests"
if not args.network:
    args.network = path.dirname(args.filename) + "/network"

class Complexity(Enum):
    LOW = 1
    LOWMEDIUM = 2
    MEDIUM = 3
    MEDIUMHIGH = 4
    HIGH = 5

    def get_delta_complexity(self):
        return {
            Complexity.LOW: 0.3,
            Complexity.LOWMEDIUM: 0.25,
            Complexity.MEDIUM: 0.2,
            Complexity.MEDIUMHIGH: 0.15,
            Complexity.HIGH: 0.5
        }[self]

    def get_instance_name(self):
        return {
            Complexity.LOW: "low",
            Complexity.LOWMEDIUM: "lowmedium",
            Complexity.MEDIUM: "medium",
            Complexity.MEDIUMHIGH: "mediumhigh",
            Complexity.HIGH: "high"
        }[self]

    def get_loading_amout(self):
        return {
            Complexity.LOW: 0.3,
            Complexity.LOWMEDIUM: 0.55,
            Complexity.MEDIUM: 0.75,
            Complexity.MEDIUMHIGH: 0.90,
            Complexity.HIGH: 0.95
        }[self]
        
with open(f'{args.filename}.pickle', 'rb') as f:
    stats = pickle.load(f)


# count the number of items
count = 0
for key in stats:
    count += 1
print(args.filename)
print("Number of items: ", 3000)
print('Accepted requests: ', count)
print("Feasibility: ", count/3000)

troughput_traffic = {}
size_network = {}
troughput_network = {}
with open(f'{args.network}.json', mode='r') as f_network:
    network = json.load(f_network)
    for idx,a_e in enumerate(network["a_e"]):
        size_network[idx + 1] = a_e

apps = {}
with open(f'{args.requests}.json', mode='r') as f_requests:
    requests = json.load(f_requests)
    for req in requests:
        apps[req["id"]] = req["name"]
        for e in req["path"]:
            if not e in troughput_traffic:
                troughput_traffic[e] = 0
            if not e in troughput_network:
                troughput_network[e] = 0
            troughput_traffic[e] += req["size"]
            size = size_network[e]
            w_fe = ceil(req["size"] / size)
            troughput_network[e] += w_fe * size

app_delay = {}
app_jitter = {}
pre_processing_average = 0
solving_time_average = 0

with open(f'{args.filename}.csv', mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(['instance', 'app', "pre_processing", "solving_time", "delay", "jitter", "cumulative_max_delay", "cumulative_max_jitter", "cumulative_max_delay_app", "cumulative_max_jitter_app"])
    for stat in stats:
        pre_processing = stats[stat]["pre_processing_time"][0] + stats[stat]["pre_processing_time"][1] * 10**-6
        solving_time = stats[stat]["solving_time"].total_seconds()
        sys_maxdelay = max(stats[stat]["delays"].values()) 
        sys_maxjitter = max(stats[stat]["jitter"].values()) 
        delay = 0
        for k,v in stats[stat]["delays"].items():
            if stat == k[0]:
                if delay < v:   
                    delay = v 
        jitter = 0
        for k,v in stats[stat]["jitter"].items():
            if stat == k:
                if jitter < v:   
                    jitter = v 

        app = apps[stat]
        
        if app not in app_delay:
            app_delay[app] = delay
        else:
            if app_delay[app] < delay:
                app_delay[app] = delay
        
        if app not in app_jitter:
            app_jitter[app] = jitter
        else:
            if app_jitter[app] < jitter:
                app_jitter[app] = jitter

        pre_processing_average += pre_processing
        solving_time_average += solving_time

        writer.writerow([stat, app, pre_processing,  solving_time, delay, jitter, sys_maxdelay, sys_maxjitter, app_delay[app], app_jitter[app]])

print("Average pre_processing time: ", pre_processing_average/count)
print("Average solving time: ", solving_time_average/count)

with open(f'{args.filename}_troughput.csv', mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(['interface', "traffic_troughput", "network_troughput", "waste_troughput"])
    # sort by troughput
    troughput_traffic = {k: v for k, v in sorted(troughput_traffic.items(), key=lambda item: item[1], reverse=True)}
    for id in troughput_traffic:
        writer.writerow([id, troughput_traffic[id] * 8 / 10**3, troughput_network[id] * 8 / 10**3, 100 * (troughput_network[id] - troughput_traffic[id]) / troughput_network[id] ])
