# load a pckl file and write stats in csv

from enum import Enum
import json
import pickle
import csv

from argparse import ArgumentParser
# generate a graph
import matplotlib.pyplot as plt
import numpy as np


parser = ArgumentParser()
parser.add_argument("-f", "--filename", dest="filename", help="Output filename.", required=True)

args = parser.parse_args()

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
print("Number of items: ", 10)
print('Accepted requests: ', count)
print("Feasibility: ", count/10)

# remove the last word
instance = args.filename.split("/")
#instance = instance[0:len(instance)-2]
# join the words
instance = "/".join(instance[0:len(instance)-1])
print("Instance: ", instance)

apps = {}
with open(f'{instance}/requests.json', mode='r') as f_requests:
    requests = json.load(f_requests)
    for req in requests:
        apps[req["id"]] = req["name"]

app_delay = {}
app_jitter = {}
sys_maxdelay = 0
sys_maxjitter = 0

pre_processing_average = 0
solving_time_average = 0

with open(f'{args.filename}.csv', mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(['instance', 'app', "pre_processing", "solving_time", "delay", "jitter", "cumulative_max_delay", "cumulative_max_jitter", "cumulative_max_delay_app", "cumulative_max_jitter_app"])
    for stat in stats:
        pre_processing = stats[stat]["pre_processing_time"][0] + stats[stat]["pre_processing_time"][1] * 10**-6
        solving_time = stats[stat]["solving_time"][0] + stats[stat]["solving_time"][1] * 10**-6
        delay = stats[stat]["delay"]
        jitter = stats[stat]["jitter"]

        app = apps[stat]

        if not app_delay.get(apps[stat]):
            app_delay[apps[stat]] = delay
        else:
            app_delay[apps[stat]] = max(app_delay[apps[stat]], delay)
        
        if not app_jitter.get(apps[stat]):
            app_jitter[apps[stat]] = jitter
        else:
            app_jitter[apps[stat]] = max(app_jitter[apps[stat]], jitter)

        sys_maxdelay = max(sys_maxdelay, delay)
        sys_maxjitter = max(sys_maxjitter, jitter)

        pre_processing_average += pre_processing
        solving_time_average += solving_time

        writer.writerow([stat, app, pre_processing,  solving_time, delay, jitter, sys_maxdelay, sys_maxjitter, app_delay[app], app_jitter[app]])

print("Average pre_processing time: ", pre_processing_average / count)
print("Average solving time: ", solving_time_average / count)