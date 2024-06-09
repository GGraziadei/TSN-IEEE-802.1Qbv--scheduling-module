from argparse import ArgumentParser
from enum import Enum
import json
import os
import networkx as nx
import matplotlib.pyplot as plt
import random

# unt references Bytes and us 

parser = ArgumentParser()
parser.add_argument("-pm", "--processing", dest="procesisng", help="Network processing mode", required=True, choices=["storeAndForward", "fast"])
parser.add_argument("-t", "--duration", dest="duration", help="Parameeter T", required=True)
parser.add_argument("-p", "--path", dest="path", help="Path generation mode", required=True, choices=["wifi2dc", "wifi2wifi"])
# add size of the requests
parser.add_argument("-s", "--size", dest="size", help="Min size of the requests in byte", required=True)
parser.add_argument("-speed", "--speed", dest="speed", help="Speed of interfaces", required=False)
parser.add_argument("-n", "--number", dest="number", help="Number of requests", required=True)

parser.add_argument("-os", "--opticalsize", dest="opticalsize", help="Size of the optical interface", required=False)
parser.add_argument("-rs", "--radiosize", dest="radiosize", help="Size of the radio interface", required=False)
parser.add_argument("-a", "--application", dest="app", help="Application class", required=False)
# python3 instance_generator/generator.py -pm storeAndForward -t 100 -p wifi -s 80
parser.add_argument("-name", "--name", dest="name", help="Instance name", required=False)

args = parser.parse_args()

def grouped(iterable):
    return [(iterable[i], iterable[i+1]) for i in range(len(iterable)-1)]

class Netowrk:

    class ProcessingMode(Enum):
        STORE_AND_FORWARD = 1
        FAST = 2

        def get_name(self):
            return self.name
        
    def __init__(self):
        self.G = nx.DiGraph()

    def add_node(self, node):
        self.G.add_node(node)

    def add_edge(self, node1, node2, layer2, processing_mode : ProcessingMode, throughput, a_e, d_e):
        if not self.G.has_node(node1):
            self.add_node(node1)
        if not self.G.has_node(node2):
            self.add_node(node2)
        interface_id = len(self.G.edges) + 1
        tau_e = a_e * 8 *  10**9 / throughput
        self.G.add_edge(node1, node2,id=interface_id, layer2=layer2, processing_mode=processing_mode, throughput=throughput, a_e=a_e, tau_e=tau_e, d_e=d_e)

    def random_path(self, debug=False, paths=[], src_list = [], dest_list = []):
        
        random_path = [] 
        start = random.choice(src_list)
        dest = random.choice(dest_list)
        while not paths[start][dest]:
            start = random.choice(src_list)
            dest = random.choice(dest_list)
        random_path   = paths[start][dest]
        random_path_e = []
        for u,v in grouped(random_path):
            if debug:
                print( f"Edge {u} -> {v}", self.G[u][v])
            random_path_e.append(self.G[u][v]["id"])
        return random_path_e

    def remove_node(self, node):
        self.G.remove_node(node)

    def remove_edge(self, node1, node2):
        self.G.remove_edge(node1, node2)

    def draw(self):
        pos = nx.spring_layout(self.G)
        nx.draw(self.G, pos, with_labels=True)
        edge_labels = nx.get_edge_attributes(self.G, 'id')
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels)
        plt.show()
    
class Application:
    
    def __init__(self, name, period, delay, jitter, path, size):
        self.period = period
        self.delay = delay
        self.jitter = jitter
        self.name = name
        self.path = path
        self.size = size

    def __str__(self):
        return self.name

### Generate instance folder ###

if not os.path.exists("instance_generator/instances/"):
    os.makedirs("instance_generator/instances/")

count_instances = len(os.listdir("instance_generator/instances/"))
instance_id = count_instances + 1
if args.name:
    instance_id = args.name
folder_name = f"instance_generator/instances/instance_{instance_id}"

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

### Generate network ###
network = Netowrk()

# DC nodes
processing_mode = Netowrk.ProcessingMode.STORE_AND_FORWARD if args.procesisng == "storeAndForward" else Netowrk.ProcessingMode.FAST
duration = int(args.duration) # us

# fixed values
optical_throughput =  10 * 10**9 if args.speed == None else int(args.speed) * 10 # 10 Gbps
optical_delay = duration/100 # 100 us 1% of the duration of the SF
optical_size = 1 if args.opticalsize == None else int(args.opticalsize)  # 1 byte
radio_throughput = 48 * 10**6 if args.speed == None else int(args.speed) * 1# Mbps
radio_delay = 4000 # us
radio_size = 24 if args.radiosize == None else int(args.radiosize) #  byte

network.add_edge("DC1", "DC2", "optical", processing_mode, optical_throughput, optical_size, optical_delay)
network.add_edge("DC2", "DC3", "optical", processing_mode, optical_throughput, optical_size, optical_delay)
network.add_edge("DC3", "DC4", "optical", processing_mode, optical_throughput, optical_size, optical_delay)

#print edges
for u,v in network.G.edges:
    print(f"Edge {u} -> {v}", network.G[u][v])


optical_throughput_lan = 100 * 10**6 if args.speed == None else int(args.speed) * 10 # 10 Gbps
optical_delay_lan = 1000

### Add user equipments ###
for i in range(1,201):
    network.add_edge(f"UE_{i}", "DC1", "wifi6", processing_mode, radio_throughput, radio_size, radio_delay)
    network.add_edge("DC4", f"UE_{i+200}", "wifi6", processing_mode, radio_throughput, radio_size, radio_delay)
    network.add_edge("DC4", f"UE_{i+400}", "wired", processing_mode, optical_throughput_lan, optical_size, optical_delay_lan)
    #network.draw()

### Generate requests ###
dc_dest = [f"UE_{i}" for i in range(401,601)]
ue_dest = [f"UE_{i}" for i in range(201,401)]
ue_src = [f"UE_{i}" for i in range(1,201)]

dest = ue_dest if args.path == "wifi2wifi" else dc_dest

requests = []
# simulate the path generation module
PATHS  = dict(nx.all_pairs_shortest_path(network.G))
for id in range(1, int(args.number) + 1):
    path = network.random_path(src_list=ue_src, dest_list=dest, paths=PATHS, debug=False)
    print(path)
    size = random.randint(int(args.size), int(args.size) + 30)

    if args.app and int(args.app) == 1:
        app = Application(f"App_1", duration/10, delay=duration/10, jitter=duration/100, path=path, size=size)
    elif args.app and int(args.app) == 2:
        app = Application(f"App_2", duration, delay=duration, jitter=duration/10, path=path, size=size*10)
    else:
        if random.random() < 0.5:
            app = Application(f"App_1", duration/10, delay=duration/10, jitter=duration/100, path=path, size=size)
        else:
            app = Application(f"App_2", duration, delay=duration, jitter=duration/10, path=path, size=size*10)
        
    
    app.id = id
    requests.append(app)

### Output files ### 
b_e_max = max([network.G[u][v]["throughput"] for u,v in network.G.edges])
tau_min = min([network.G[u][v]["tau_e"] for u,v in network.G.edges])
instance_size = len(network.G.edges)**2 * (duration * b_e_max / tau_min)**2
# txt file description
with open(f"{folder_name}/description.txt", "w") as f:
    f.write(f"Processing mode: {processing_mode.get_name()}\n")
    f.write(f"Duration: {duration} us\n")
    f.write(f"Path generation mode: {args.path}\n")
    f.write(f"Number of requests: {len(requests)}\n")
    f.write(f"Min size of the requests: {args.size} bytes\n")
    f.write(f"Instance size: {instance_size}\n")

# json file network
with open(f"{folder_name}/network.json", "w") as f:
    G = network.G
    data = {
        "SF_duration" : duration,
        "E" : [G[u][v]["id"] for (u, v) in G.edges()],
        "B_e" : [G[u][v]["throughput"] for i, (u, v) in enumerate(G.edges())],
        "a_e" : [G[u][v]["a_e"] for i, (u, v) in enumerate(G.edges())],
        "tau_e" : [G[u][v]["tau_e"] for i, (u, v) in enumerate(G.edges())],
        "g_e" : [ [] for i, (u, v) in enumerate(G.edges())],
        "d_e" : [G[u][v]["d_e"] for i, (u, v) in enumerate(G.edges())],
        "mode" : [ processing_mode.get_name()  for i, (u, v) in enumerate(G.edges())],
    }
    json.dump(data, f, indent=4)  


# json file requests
with open(f"{folder_name}/requests.json", "w") as f:
    data = []
    for r in requests:
        data.append(r.__dict__)
    json.dump(data, f)







