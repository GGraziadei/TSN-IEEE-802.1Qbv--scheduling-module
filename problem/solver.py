import copy
import datetime
from math import ceil
import pickle
from bitarray import bitarray
from problem.network import Network
from problem.flow import Flow
from problem.directed_link import DirectedLink
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


class TSNScheduling:

    @classmethod
    def now(cls) -> classmethod:
        return datetime.datetime.now()
    
    @classmethod
    def format_delta(cls, begin, end) -> classmethod:
        delta = end - begin
        return delta.seconds, delta.microseconds

    @classmethod
    def grouped(cls, iterable) -> classmethod:
        return [(iterable[i], iterable[i+1]) for i in range(len(iterable)-1)]

    @classmethod
    def gcd(cls, p, q) -> classmethod:
        #  Euclid's algorithm to find the GCD.
        while q != 0:
            p, q = q, p % q
        return p

    @classmethod
    def is_coprime(cls, x, y) -> classmethod:
        # Check if the GCD of 'x' and 'y' is equal to 1.
        return cls.gcd(x, y) == 1
    
    @classmethod
    def lcm(cls, n : list) -> classmethod:
        # Compute the Least Common Multiple of 'x' and 'y'.
        lcm = n[0]
        for i in n[1:]:
            lcm = lcm*i//cls.gcd(lcm, i)
        return lcm
    
    
    def __init__(self):
        pass
    
    def __init__(self, request : list[dict], network_parameeter:dict, 
                incumbent : dict[(int,int,int,int):int] = {}) :
        
        # network is the dictionary with the network configuration
        network = Network( T=network_parameeter["SF_duration"])
        
        # request is the list of dictionary with all the flows to schedule
        flows = []
        for f in request:
            flows.append(Flow(f["id"], f["E_f"], f["P_f"], f["T_f"], f["DELAY_f"], f["JITTER_f"], f["#f"]))
        network.set_flows(flows)

        directed_links = []
        for id,speed in enumerate(network_parameeter["B_e"]):
            tau_e = network_parameeter["tau_e"][id-1]
            a_e = network_parameeter["a_e"][id-1]
            d_e = network_parameeter["d_e"][id-1]
            g_e = network_parameeter["g_e"][id-1]
            directed_links.append(DirectedLink(id, speed, a_e, tau_e, d_e, g_e))

        network.set_links(directed_links)

        self.network = network
        self.incumbent = incumbent

    def __init__(self, request : list[Flow], network_parameeter:dict, 
                incumbent : dict[(int,int,int,int):int] = {}) :
        
        # network is the dictionary with the network configuration
        network = Network( T=network_parameeter["SF_duration"])
        
        # request is the list of dictionary with all the flows to schedule
        network.set_flows(request)

        directed_links = []
        for id,speed in enumerate(network_parameeter["B_e"]):
            tau_e = network_parameeter["tau_e"][id-1]
            a_e = network_parameeter["a_e"][id-1]
            d_e = network_parameeter["d_e"][id-1]
            g_e = network_parameeter["g_e"][id-1]
            directed_links.append(DirectedLink(id, speed, a_e, tau_e, d_e, g_e))
        network.set_links(directed_links)

        self.network = network
        self.incumbent = incumbent

    def __init__(self, network_parameeter:dict, 
                incumbent : dict[(int,int,int,int):int] = {}) :
        
        # network is the dictionary with the network configuration
        network = Network( T=network_parameeter["SF_duration"])
        
        # request is the list of dictionary with all the flows to schedule
        flows = []
        if "E_f" in network_parameeter:
            for f in range(1, len(network_parameeter["E_f"])+1):
                flows.append(Flow(f, network_parameeter["E_f"][f-1], network_parameeter["P_f"][f-1], network_parameeter["DELAY_f"][f-1], network_parameeter["JITTER_f"][f-1], network_parameeter["#f"][f-1]))
    
        network.set_flows(flows)

        directed_links = []
        id = 1
        for speed in network_parameeter["B_e"]:
            tau_e = network_parameeter["tau_e"][id-1]
            a_e = network_parameeter["a_e"][id-1]
            d_e = network_parameeter["d_e"][id-1]
            g_e = network_parameeter["g_e"][id-1]
            if "mode" in network_parameeter:
                if network_parameeter["mode"][id-1] == "STORE_AND_FORWARD":
                    directed_links.append(DirectedLink(id, speed, a_e, tau_e, d_e, g_e, DirectedLink.TransmissionMode.STORE_AND_FORWARD))
                else:
                    directed_links.append(DirectedLink(id, speed, a_e, tau_e, d_e, g_e, DirectedLink.TransmissionMode.FAST))
            else:
                directed_links.append(DirectedLink(id, speed, a_e, tau_e, d_e, g_e))
            id += 1

        network.set_links(directed_links)

        self.network = network
        self.incumbent = incumbent

    def solve(self):
        pass

    def store_data_plane(self, file_name:str = 'data_plane.pkl'):
        if self.x_feti:
            # store pckl
            with open(file_name, 'wb') as f:
                pickle.dump(self.x_feti, f)

    def generate_gantt(self, file_name:str = 'gantt.png'):
        df = pd.DataFrame(columns=['flow', 'link', 'start', 'end', 'duration'])
        row_id = 0
        colors = {}
        for f in self.network.get_flows():
            for i in range(1, f.get_T_f()+1):
                color = np.random.rand(3,)
                colors[f.get_id(),i] = color

        for (f,e,t,i) in self.x_feti.keys():
            flow = self.network.get_flows()[f-1]
            link = self.network.get_links()[e-1]
            w_fe = flow.w_fe[e]
            new_row = {
                    'flow': (f,i),
                    'link': f'link_{e}',
                    'start': (t-1) * link.get_tau_e(),
                    'duration': w_fe * link.get_tau_e(),
                    # assign random color to each flow
            }
            df.loc[row_id] = new_row
            row_id += 1
        df.sort_values(by=['link', 'start'], inplace=True)
        # create gantt chart
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, row in df.iterrows():
            flow = row['flow']
            color = colors[flow]
            ax.barh(row['link'], row['duration'], left=row['start'], color=color, label=row['link'])
        
        for link in self.network.get_links():
            g_e = link.get_g_e()
            label = f'link_{link.get_id()}'
            print(label, g_e)
            for t in g_e:
                start = (t-1) * link.get_tau_e()
                ax.barh(label, link.get_tau_e(), left=start, color='red', label=label)

        ax.set_xlabel('Time (us)')
        ax.set_ylabel('Link')
        ax.set_title('T scheduling - Gantt Chart')
        ax.invert_yaxis()
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1))
        ax.grid(which='major', axis='x', linestyle='-')
    
        
        plt.show()
    
    def pipeline(self,e1,e2,flow_id) -> float:  
        link1 = self.network.get_link_by_id(e1) 
        link2 = self.network.get_link_by_id(e2)
        flow = self.network.get_flows()[flow_id-1]
        h_12 = self.h_ij[e1,e2]
        h_21 = self.h_ij[e2,e1]
        w_fe = flow.w_fe[e1]

        last_e1 = w_fe * link1.get_tau_e()
        first_e2 = ceil(last_e1/link2.get_tau_e()) * link2.get_tau_e()

        sigma = first_e2

        if link1.get_mode() == DirectedLink.TransmissionMode.STORE_AND_FORWARD:
            return sigma

        if link2.get_tau_e() <= h_12 * link1.get_tau_e():
            w_fe = flow.w_fe[e2]
            sigma = first_e2 - link2.get_tau_e() * (w_fe - ceil(h_21))
        else:
            sigma = ceil(link1.get_tau_e() / link2.get_tau_e()) * link2.get_tau_e()

        d = link1.get_d_e() 
        return sigma + d

    
    def is_apriori_feasible(self) -> bool: 
        tick = self.now()

        # check a-priori infeasibility

        # available timeslots for each link
        required_timeslots = {}
        for f in self.network.get_flows():
            for e_id in f.get_E_f():
                w_fe = f.w_fe[e_id]
                if e_id in required_timeslots:
                    required_timeslots[e_id] += w_fe
                else:
                    required_timeslots[e_id] = w_fe
        
        for e in self.network.get_links():
            if e.get_id() not in required_timeslots:
                continue
            t_e = e.get_t_e()
            required = required_timeslots[e.get_id()]
            if t_e < required:
                print(f"Link {e.get_id()} is infeasible, not enough timeslots for all flows")
                return False
            
        # check co-prime restrictions
        for e1,e2 in self.grouped(self.network.get_links()):   
            a_e1 = e1.get_a_e()
            a_e2 = e2.get_a_e()
            tau_e1 = e1.get_tau_e()
            tau_e2 = e2.get_tau_e()
            if self.is_coprime(a_e1,a_e2) and a_e1 != 1 and a_e2 != 1:
                print(f"Link {e1.get_id()} and {e2.get_id()} are co-prime in bits per timeslot")
                return False
            """
            if self.is_coprime(tau_e1,tau_e2) and tau_e1 != 1 and tau_e2 != 1:
                print(f"Link {e1.get_id()} and {e2.get_id()} are co-prime in timeslot duration")
                return False
            """

        T = self.network.get_T()
        for f in self.network.get_flows():
            if self.is_coprime(f.get_P_f(), T):
                print(f"Flow {f.get_id()} is infeasible, P_f and T are co-prime")
                return False
            if f.get_P_f() > T:
                print(f"Flow {f.get_id()} is infeasible, P_f is greater than T")
                return False
              
        tock = self.now()
        self.is_apriori_feasible_time = self.format_delta(tick, tock)
        return True
    
    def pre_processing(self):
        tick = self.now()
        
        for link in self.network.get_links():
            link.tau_e = link.get_tau_e()
            # define the number of timeslots per each link.
            link.t_e = int(float(self.network.T) / link.tau_e)
            a = bitarray('1'*link.t_e)
            for t in link.get_g_e():
                if 1<=t<=link.t_e:
                    a[t-1] = False
            link.set_T_e(a)
            for link_2 in self.network.get_links():
                
                a_e1 = link.get_a_e()
                a_e2 = link_2.get_a_e()

                h_12 = a_e2 / a_e1
                # define the transofrmation matrix for each pair of directed links.
                self.network.h_ij[link.get_id(),link_2.get_id()] = h_12

        for flow in self.network.get_flows():
            # define the number of iteration per each flow.
            T_f = ceil(self.network.get_T()/flow.get_P_f())
            flow.set_T_f(T_f)

            for e in flow.get_E_f():
                link = self.network.get_link_by_id(e)
                # define the window size per each directed link.
                flow.w_fe[e] = ceil(flow.size_f/link.a_e)

        
        
        self.d_fi = []
        self.j_f = []
        
        

        # get last t for each flow - remove the t from the link
        for (f,e,t,i) in self.incumbent.keys():
            flow = self.network.get_flows()[f-1]
            e_src = flow.get_E_f()[0]
            e_dest = flow.get_E_f()[-1]

            # remove allocated t
            self.network.get_links()[e-1].remove_t(t)    
        
            if e == e_src:
                last_i,last_t = self.network.get_flows()[f-1].get_last_t()
                if last_t < t or not last_i:
                    self.network.get_flows()[f-1].set_last_t((i,t))
            
            if e == e_dest:
                tau_dest = self.network.get_links()[e-1].get_tau_e()
                delay = t - (flow.get_DELAY_f() * (i-1)) + tau_dest
                if delay > flow.get_max_delay():
                    flow.set_max_delay(delay)
                if delay < flow.get_min_delay():
                    flow.set_min_delay(delay)

                self.d_fi[(f,i)] = delay
        
        if self.incumbent:
            for flow in self.network.get_flows():
                self.j_f[flow.get_id()] = flow.get_max_delay() - flow.get_min_delay()

        # min lantency for a flow
        for f in self.network.get_flows():
            min_latency = 0
            for e in f.get_E_f():
                min_latency += self.network.get_links()[e-1].get_tau_e()
            f.set_min_latency(min_latency)

        if self.incumbent:
            self.x_feti = copy(self.incumbent)

        self.h_ij = self.network.h_ij

        tock = self.now()
        self.pre_processing_time = self.format_delta(tick, tock)
