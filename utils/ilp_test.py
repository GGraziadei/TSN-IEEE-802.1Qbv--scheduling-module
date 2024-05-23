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
parser.add_argument("-n", "--network", dest="network", help="Network configuration in json format.", required=True)
parser.add_argument("-r", "--requests", dest="requests", help="Requests configuration in json format.", required=True)
parser.add_argument("-f", "--filename", dest="filename", help="Output filename.", required=True)

args = parser.parse_args()

'''
python3 ilp_test.py -n network_fast.json -f ilp_fast_wifidc -r rquests_wifiToDC.json
python3 ilp_test.py -n network_storeAndForward.json -f storeAndForward_wifidc -r rquests_wifiToDC.json  
python3 ilp_test.py -n network_fast.json -f fast_wifiwifi -r rquests_wifiToWifi.json
python3 ilp_test.py -n network_storeAndForward.json -f storeAndForward_wifiwifi -r rquests_wifiToWifi.json 

python3 ilp_test.py -n T1000/network_fast.json -f T1000ilp_fast_wifidc -r T1000rquests_wifiToDC.json

'''

stats = {}
for complexity in Complexity:
    instance_folder = 'instance_generator/' + complexity.get_instance_name()
    stats[complexity.get_instance_name()] = []

    with open(f'instance_generator/{args.requests}', 'r') as f_requests:
        requests = json.load(f_requests)
        for request in requests:
            flow = Flow(id=1, E_f=request["path"], P_f=request["period"], DELAY_f=request["delay"], JITTER_f=request["jitter"], size_f=request["size"])
            with open(f"instance_generator/{args.network}", 'r') as f:
                
                data = json.load(f)
                instance = ILP(data)
                
                #acquire the load over the interfaces
                """
                for e in instance.network.links:
                    id = e.id
                    with open(f'instance_generator/{complexity.get_instance_name()}/{complexity.get_instance_name()}_directed_link_{id}.bin', 'rb') as file:
                        from bitarray.util import deserialize
                        array = deserialize(file.read())
                        e.set_T_e(array)
                """
                
                instance.network.flows = [flow]
                instance.pre_processing()
                print(instance.network)
                result = instance.solve()

                if result['status'] != 'Infeasible':
                    stat = instance.stat()
                    stat["delay"] = instance.d
                    stat["jitter"] = instance.j
                    stat['id'] = flow.get_id()
                    stats[complexity.get_instance_name()].append(stat)

                del instance

            
# save stats in pickle file
import pickle
with open(f'instance_generator/ilp_{args.filename}.pickle', 'wb') as f:
    pickle.dump(stats, f)
