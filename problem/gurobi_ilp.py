import copy
from math import ceil
from gurobipy import *

from problem.milp_model import ILP

class GurobiILP(ILP):

    def solve(self):
        tick = self.now()

        prob = Model("TSN - Scheduling Problem")

        eti_space, i_space, ei_space = self.space()   

        #x_eti   = prob.addVars(ei_space, vtype=GRB.BINARY, name="x_eti")
        x_eti = {}
        for e,t,i in eti_space:
            x_eti[e,t,i] = prob.addVar(vtype=GRB.BINARY, name=f"x_eti_{e}_{t}_{i}")

        #y_p     = prob.addVars(i_space, vtype=GRB.INTEGER, name="y_p")
        y_p = {}
        for i in i_space:
            y_p[i] = prob.addVar(vtype=GRB.INTEGER, name=f"y_p_{i}")

        y_max   = prob.addVar(vtype=GRB.INTEGER, name="y_max")
        z       = prob.addVar(vtype=GRB.INTEGER, name="z")

        # constraint 3
        # all the iterations have to scheduled over each link of the path
        for i in range(1,self.T_f + 1):
            for e in self.E:
                T = ILP.bitarray_to_list(self.b_ie_t[(i,e)])
                #prob += lpSum([x_eti[e,t,i] for t in T]) == 1
                prob.addConstr(quicksum(x_eti[e,t,i] for t in T) == 1)
        
        # constraint 2
        # at most one iteration can starts in one timeslot 
        for e in self.E:
            link = self.network.get_link_by_id(e)
            t_e = link.get_t_e()
            w_fe = self.w_fe[e]
            for t in range(1,t_e+1):
                t_a = t + t_e 
                t_b = min(t_a + w_fe - 1, 2*t_e)
                #prob += lpSum([x_eti[e,t,i] for i in range(1,self.T_f + 1)]) + lpSum([x_eti[e,tt,i] for i in range(1,self.T_f + 1) for tt in range(t_a,t_b+1)]) <= 1
                prob.addConstr(quicksum(x_eti[e,t,i] for i in range(1,self.T_f + 1)) + quicksum(x_eti[e,tt,i] for i in range(1,self.T_f + 1) for tt in range(t_a,t_b+1)) <= 1)
        
        '''
        # constraint 4
        # each iteration can start only in one available timeslot
        for i in range(1,self.T_f + 1):
            for e in self.flow.get_E_f():
                for t_index, value in enumerate(self.b_ie_t[(i,e)]):
                    if value == False:
                        #print(f"Flow {self.flow.get_id()} - iteration {i} is not schedulable on link {e} at time {t_index+1}")
                        prob += x_eti[e,t_index+1,i] == 0
        '''
        
        dest_link = self.network.get_link_by_id(self.E_f[-1])
        p_f_dest = ceil(self.P_f/dest_link.get_tau_e())
        tau_dest = dest_link.get_tau_e()
        d_e_dest = dest_link.get_d_e()
        for i in range(1,self.T_f + 1):
            e = dest_link.get_id()

            T2 = ILP.bitarray_to_list(self.b_ie_t[(i,e)])

            #prob += y_p[i]  == lpSum([(t-1) *  x_eti[e,t,i] for t in T2]) - p_f_dest * (i-1)
            prob.addConstr(y_p[i]  == quicksum((t-1) *  x_eti[e,t,i] for t in T2) - p_f_dest * (i-1))
            #prob += y_max >= y_p[i]
            prob.addConstr(y_max >= y_p[i])
            
        #prob+= y_max * tau_dest + d_e_dest <= self.DELAY_f
        prob.addConstr(y_max * tau_dest + d_e_dest <= self.DELAY_f)

        # jitter constraint
        if self.T_f > 1:
            for i1 in range(1,self.T_f + 1):
                for i2 in range(1,self.T_f +1):
                    if i1 != i2 : 
                        #prob+= z >= y_p[i1] - y_p[i2]
                        prob.addConstr(z >= y_p[i1] - y_p[i2])
        else:
            prob.addConstr(z == 0)
        #prob += z * tau_dest <= self.JITTER_f
        prob.addConstr(z * tau_dest <= self.JITTER_f)
        
        for i in range(1,self.T_f+1):
            for e1,e2 in self.grouped(self.E_f):
                T1 = ILP.bitarray_to_list(self.b_ie_t[(i,e1)])
                T2 = ILP.bitarray_to_list(self.b_ie_t[(i,e2)])
                tau_e2 = self.network.get_link_by_id(e2).get_tau_e()
                tau_e1 = self.network.get_link_by_id(e1).get_tau_e()
                sigma = self.pipeline(e1,e2,self.flow.get_id())
                #prob += lpSum([x_eti[e1,t,i] * (t-1) * tau_e1 for t in T1]) + sigma <= lpSum([tau_e2 * (t-1) * x_eti[e2,t,i] for t in T2])
                prob.addConstr(quicksum(x_eti[e1,t,i] * (t-1) * tau_e1 for t in T1) + sigma <= quicksum(tau_e2 * (t-1) * x_eti[e2,t,i] for t in T2))

        W1 = 100/self.JITTER_f
        W2 = 10/self.DELAY_f
        #prob += W1 * z + W2 * y_max
        prob.setObjective(W1 * z + W2 * y_max, GRB.MINIMIZE)

        tock = self.now()
        self.loading_time = self.format_delta(tick, tock)

        self.tick = self.now()
        print("Solving...")
        prob.optimize()
        self.tock = self.now()
        self.solving_time = self.format_delta(self.tick, self.tock)

        if prob.status !=  GRB.Status.OPTIMAL:
            return {"status": prob.status}

        tau_dest = self.network.get_link_by_id(self.E_f[-1]).get_tau_e()
        self.x_eti = {}
        self.y_p = {}
        tau_dest = self.network.get_link_by_id(self.E_f[-1]).get_tau_e()

        for var in prob.getVars():
            if var.varName.startswith("x_eti"):
                e,t,i = var.varName.split("_")[2:]
                e,t,i = int(e),int(t),int(i)
                self.x_eti[e,t,i] = var.x
                w_fe = self.w_fe[e]
                if var.x == 1:
                    for t in range(t, t + w_fe):
                        self.network.get_link_by_id(e).remove_t(t)
            elif var.varName.startswith("y_p"):
                i = var.varName.split("_")[2]
                i = int(i)
                self.y_p[i] = var.x * tau_dest
            elif var.varName == "y_max":
                self.d = var.x * tau_dest
            elif var.varName == "z":
                self.j = var.x 
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