import copy
from enum import Enum
import json
import os
from problem.flow import Flow
from problem.greedy_model import Greedy
from problem.milp_model import ILP
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-i", "--instance", dest="folder", help="Instance folder.", required=True)

args = parser.parse_args()

'''
python3 dynamic_test.py -i instance_generator/instances/instance_1
'''

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
    

stats = {}
with open(f'{args.folder}/network.json', 'r') as f:
        flows = []
        with open(f'{args.folder}/requests.json', 'r') as f_requests:
            requests = json.load(f_requests)
            for request in requests:
                flow = Flow(
                    id=request["id"],
                    E_f=request["path"],
                    P_f=request["period"],
                    DELAY_f=request["delay"],
                    JITTER_f=request["jitter"],
                    size_f=request["size"])
                flows.append(flow)

        data = json.load(f)
        instance = Greedy(data)
        flow = flows.pop(0)
        instance.network.flows = [flow]
        instance.pre_processing()
        instance.solve()
        
        networks_load = {}
        stats[flow.get_id()] = instance.stat()
        stats[flow.get_id()]["delays"]=copy.deepcopy(instance.d_fi)
        stats[flow.get_id()]["jitter"]=copy.deepcopy(instance.j_f)
        #stats[flow.get_id()]["plan"]=copy.deepcopy(instance.x_feti)
        count = 1

        for flow in flows:
            sol = instance.new_request(flow)
            if sol:
                stats[flow.get_id()] = instance.stat()
                stats[flow.get_id()]["delays"]=copy.deepcopy(instance.d_fi)
                stats[flow.get_id()]["jitter"]=copy.deepcopy(instance.j_f)
                stats[flow.get_id()]["app"] = request["name"]
                #stats[flow.get_id()]["plan"]=copy.deepcopy(instance.x_feti)
            count += 1
            
            if count==200:
                networks_load[200] = copy.deepcopy(instance.network)
            elif count==400:
                networks_load[400] = copy.deepcopy(instance.network)
            elif count==600:
                networks_load[600] = copy.deepcopy(instance.network)
            elif count==800:
                networks_load[800] = copy.deepcopy(instance.network)
            elif count==1000:
                networks_load[1000] = copy.deepcopy(instance.network)
         
# save stats in pickle file
import pickle
with open(f'{args.folder}/out_heuristic.pickle', 'wb') as f:
    pickle.dump(stats, f)

with open(f'{args.folder}/load.pickle', 'wb') as f:
    pickle.dump(networks_load, f)
