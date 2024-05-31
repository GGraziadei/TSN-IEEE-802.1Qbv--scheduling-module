import copy
from math import ceil
from bitarray import bitarray
from gurobipy import *

from problem.milp_model import ILP

class GurobiILP(ILP):

    def pre_processing(self, debug = False) -> None:
        tick = self.now()
        super().pre_processing()
        
        assert self.network is not None, "Network is not defined"
        assert len(self.network.get_flows()) >= 1, "At least one flow is required"
         
        self.T_f = self.network.get_T_f()
        self.E = range(1,len(self.network.get_links())+1)
        self.F = range(1,len(self.network.get_flows())+1)
        
        self.E_f = [f.get_E_f() for f in self.network.get_flows()]
        self.P_f = [f.get_P_f() for f in self.network.get_flows()]
        
        self.DELAY_f = [f.get_DELAY_f() for f in self.network.get_flows()]
        self.JITTER_f = [f.get_JITTER_f() for f in self.network.get_flows()]
        
        self.SF_duration = self.network.get_T()
        
        # in the preprocessing the not available timeslots are set to 0
        self.T_e = [e.get_T_e() + e.get_T_e() # bitarray of available timeslots
            for e in self.network.links]
        
        # b_feti pre-processing matrix of the available timeslots
        # first-level key is the tuple iteration, link
        # second level key is timeslot (id-1)
        # at each key of the dictionary correponds a bitarray of the available timeslots
        # for starting the specific iteration on the specific link
        self.W = {flow.get_id() : flow.get_w_fe()
                   for flow in self.network.get_flows()}

        self.N = {}

        for f in self.F:
            for i in range(1,self.T_f[f-1]+1):
                for e in self.E_f[f-1]:
                    w = self.W[f][e]
                    link = self.network.get_link_by_id(e)
                    mask = bitarray('1'*w)
                    p_fe = ceil(self.P_f[f-1]/link.get_tau_e())
                    first_assignable = p_fe * (i-1) + 1
                    iterator = link.get_T_e().search(mask)
                    first_assignable = first_assignable - 1
                    bit_array = bitarray('0'*link.get_t_e())

                    for t in iterator:
                        if t >= first_assignable:
                            bit_array[t] = True

                    self.N[(f,e,i)] = bit_array

        tock = self.now()
        self.pre_processing_time += self.format_delta(tick, tock)

        if debug:
            print("Network configuration")
            # write links
            for l in self.network.links: print(l)
            # write flows 
            for f in self.network.flows: print(f)
            
    def space(self):
        feti = [(f,e,t,i)  
                for f in self.F
                    for e in self.E_f[f-1]
                        for i in range(1,self.T_f[f-1]+1)
                            for t in ILP.bitarray_to_list(self.T_e[e-1]) ]
        
        fi = [(f,i) for f in self.F for i in range(1,self.T_f[f-1]+1)]
        f = [f for f in self.F]
        fet = [(f,e,t) for f in self.F for e in self.E_f[f-1] for t in self.T_e[e-1]]

        return feti, fet, fi, f
    
    def solve(self):
        tick = self.now()

        prob = Model("TSN - Scheduling Problem")

        feti, fet, fi, f = self.space()   

        x_feti = {} # space reduced with N matrix constriant 
        y_feti = {}
        for f,e,t,i in feti:
            t_e = self.network.get_link_by_id(e).get_t_e()
            source = self.E_f[f-1][0]

            x_feti[f,e,t,i] = prob.addVar(vtype=GRB.BINARY, name=f"x_feti_{f}_{e}_{t}_{i}")
            y_feti[f,e,t,i] = prob.addVar(vtype=GRB.BINARY, name=f"y_feti_{f}_{e}_{t}_{i}")

            if t <= t_e:
                prob.addConstr(x_feti[f,e,t,i] <= self.N[(f,e,i)][t-1])
            if e == source and t > t_e:
                prob.addConstr(x_feti[f,e,t,i] == 0)

        c_fet = {}
        for f,e,t in fet:
            c_fet[f,e,t] = prob.addVar(vtype=GRB.BINARY, name=f"c_fet_{f}_{e}_{t}")

        d_fi = {}
        for f,i in fi:
            d_fi[f,i] = prob.addVar(vtype=GRB.INTEGER, name=f"d_fi_{f}_{i}")

        j_f = {}
        for f in self.F:
            j_f[f] = prob.addVar(vtype=GRB.INTEGER, name=f"j_f_{f}")

        w = prob.addVar(vtype=GRB.INTEGER, name="w")
        z = prob.addVar(vtype=GRB.INTEGER, name="z")

        #C1
        for e in self.E:
            for t in ILP.bitarray_to_list(self.T_e[e-1]):
                F = [f for f in self.F if e in self.E_f[f-1]]
                prob.addConstr(quicksum(y_feti[f,e,t,i] for f in F for i in range(1,self.T_f[f-1])) <= 1)

        #C2
        for f in self.F:
            for e in self.E_f[f-1]:
                for i in range(1,self.T_f[f-1]+1):
                    T = ILP.bitarray_to_list(self.N[(f,e,i)])
                    prob.addConstr(quicksum(x_feti[f,e,t,i] for t in T) == 1)

        #C3
        for f in self.F:
            for e in self.E_f[f-1]:
                for i in range(1,self.T_f[f-1]+1):
                    w = self.W[f][e]
                    t_e = self.network.get_link_by_id(e).get_t_e()
                    for k in range(t,min(t+w,2*t_e)):
                        prob.addConstr(x_feti[f,e,t,i] <= y_feti[f,e,k,i])

        #Time constraints
        for f in self.F:
            for i in range(1,self.T_f[f-1]+1):
                e_dest = self.E_f[f-1][-1]
                tau_dest = self.network.get_link_by_id(e_dest).get_tau_e()
                d_e_dest = self.network.get_link_by_id(e_dest).get_d_e()
                p_f_dest = ceil(self.P_f[f-1]/tau_dest)
                prob.addConstr(d_fi[f,i] == quicksum((t-1) * x_feti[f,e_dest,t,i] for t in ILP.bitarray_to_list(self.T_e[e_dest-1])) - p_f_dest * (i-1))
                prob.addConstr(d_fi[f,i] * tau_dest + d_e_dest <= self.DELAY_f[f-1])
                prob.addConstr(w >= d_fi[f,i] )

            if self.T_f[f-1] > 1:
                for i1 in range(1,self.T_f[f-1]+1):
                    for i2 in range(1,self.T_f[f-1]+1):
                        if i1 != i2:
                            prob.addConstr(j_f[f] >= d_fi[f,i1] - d_fi[f,i2])
                prob.addConstr(j_f[f] * tau_dest <= self.JITTER_f[f-1])
            else: prob.addConstr(j_f[f] == 0)
            prob.addConstr(z >= j_f[f])

        
        for f in self.F:
            for i in range(1,self.T_f[f-1]+1):
                for e1,e2 in self.grouped(self.E_f):
                    T1 = ILP.bitarray_to_list(self.b_ie_t[(i,e1)])
                    T2 = ILP.bitarray_to_list(self.b_ie_t[(i,e2)])
                    tau_e2 = self.network.get_link_by_id(e2).get_tau_e()
                    tau_e1 = self.network.get_link_by_id(e1).get_tau_e()
                    sigma = self.pipeline(e1,e2,f)
                    prob.addConstr(quicksum(x_feti[f,e1,t,i] * (t-1) * tau_e1 for t in T1) + sigma <= quicksum(tau_e2 * (t-1) * x_feti[f,e2,t,i] for t in T2))
    
        # circular shift constraint
        for e in self.E:
            for t in ILP.bitarray_to_list(self.T_e[e-1]):
                t_e = self.network.get_link_by_id(e).get_t_e()
                if t <= t_e:
                    prob.addConstr(quicksum(y_feti[f,e,t,i] + y_feti[f,e,t+t_e,i]  for f in self.F for i in range(1,self.T_f[f-1]+1)) <= 1)

        prob.setObjective(z + 0.01 * w, GRB.MINIMIZE)

        tock = self.now()
        self.loading_time = self.format_delta(tick, tock)

        self.tick = self.now()
        print("Solving...")
        prob.optimize()
        self.tock = self.now()
        self.solving_time = self.format_delta(self.tick, self.tock)

        if prob.status !=  GRB.Status.OPTIMAL:
            return {"status": prob.status}
        self.y_feti = {}
        for f,e,t,i in feti:
            self.y_feti[f,e,t,i] = y_feti[f,e,t,i].X
            if y_feti[f,e,t,i].X == 1:
                print(f,e,t,i)
        self.d = w
        self.j = z
        print(self.d,self.j)
        
        
        self.objective = prob.getObjective().getValue()
        return {
            "status": True, 
            "objective": self.objective,
        }
    
    def generate_gantt(self, file_name:str = 'gantt.png'):
        import matplotlib
        from matplotlib import pyplot as plt
        import numpy as np
        import pandas as pd

        df = pd.DataFrame(columns=['flow', 'link', 'start', 'end', 'duration'])
        row_id = 0
        colors = {}

        for i in range(1, self.T_f + 1):
            color = np.random.rand(3,)
            colors[i] = color

        eti_space, i_space, etk_space = self.space()

            
        for k,v in self.x_eti.items():
            e,t,i = k
            if v == 0:
                continue
            link = self.network.get_link_by_id(e)
            w_fe = self.w_fe[e]
            new_row = {
                    'flow': i,
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
        for link in self.network.get_links():
            g_e = link.get_g_e()
            label = f'link_{link.get_id()}'
            for t in g_e:
                start = (t-1) * link.get_tau_e()
                ax.barh(label, link.get_tau_e(), left=start, color='grey', label=label)
            tau_e = link.get_tau_e()
            
        for i, row in df.iterrows():
            iteration = row['flow']
            color = colors[iteration]
            ax.barh(row['link'], row['duration'], left=row['start'], color=color, label=row['link'])
        
        ax.set_xlabel('Time (us)')
        ax.set_ylabel('Link')
        ax.set_title('T scheduling - Gantt Chart')
        ax.invert_yaxis()
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1))
        ax.grid(which='major', axis='x', linestyle='-')
    
        
        plt.show()