from argparse import ArgumentParser
from enum import Enum
import json
import os
from problem.flow import Flow
from problem.gurobi_ilp import GurobiILP as ILP

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
    
parser = ArgumentParser()
parser.add_argument("-i", "--instance", dest="instance", help="Instance path", required=True)

args = parser.parse_args()

folder = args.instance

stats = {}
with open(f'{folder}/network.json', 'r') as network_file:
    data = json.load(network_file)
    instance = ILP(data)
    with open(f'{folder}/requests.json', 'r') as f_requests:
        requests = json.load(f_requests)
        for request in requests:
            flow = Flow(id=1, 
                    E_f=request["path"],
                    P_f=request["period"],
                    DELAY_f=request["delay"],
                    JITTER_f=request["jitter"],
                    size_f=request["size"])
            instance.network.flows = [flow]
            instance.pre_processing()
            result = instance.solve()
            if result['status'] != 'Infeasible':
                stat = instance.stat()
                stat["delay"] = instance.d
                stat["jitter"] = instance.j
                stat['id'] = flow.get_id()
                stats[flow.get_id()] = stat

                for k,v in instance.x_eti.items():
                    if v == 1:
                        e,t,i = k
                        w_fe = instance.w_fe[e]
                        link = instance.network.get_link_by_id(e)
                        for tt in range(t, t + w_fe):
                            link.remove_t(tt)
          
# save stats in pickle file
import pickle
with open(f'{folder}/out_ilp.pickle', 'wb') as f:
    pickle.dump(stats, f)
