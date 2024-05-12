import copy
from math import ceil
import pickle
import random
import matplotlib
import matplotlib.pyplot as plt
from pulp import *
import pandas as pd
import numpy as np
import datetime
from bitarray import bitarray


from problem.directed_link import DirectedLink
from problem.flow import Flow
from problem.network import Network
from problem.solver import TSNScheduling

class ILP(TSNScheduling):

    path_to_cplex = r'/home/ggraziadei/opt/ibm/ILOG/CPLEX_Studio2211/cplex/bin/x86-64_linux/cplex'
    solver = CPLEX_CMD(path=path_to_cplex,threads=8, timeLimit=3600, gapRel=0.05)

    @classmethod
    def bitarray_to_list(cls, bit_array:bitarray):
        for index, value in enumerate(bit_array):
            if value == True:
                yield index+1

    def __str__(self) -> str:  
        return f"TSNScheduling({self.network}, {self.incumbent})"
    
    def space(self):
        eti = [(e,t,i) for e in self.E
                                    for t in ILP.bitarray_to_list(self.T_e[e-1])
                                        for i in range(1,self.T_f+1)]
        i   = [ i for i in range(1,self.T_f + 1)]
        ei  = [(e,i) for e in self.E for i in range(1,self.T_f + 1)]

        return eti, i, ei
    
    def pre_processing(self, debug = False) -> None:
        tick = self.now()
        super().pre_processing()
        
        assert self.network is not None, "Network is not defined"
        assert len(self.network.get_flows()) == 1, "Only one flow is allowed"
        self.flow = self.network.get_flows()[0]
         
        #self.T_f = self.network.get_T_f()
        #self.E = range(1,len(self.network.get_links())+1)
        # self.F = range(1,len(self.network.get_flows())+1)
        
        self.E = self.flow.get_E_f()
        self.T_f = self.flow.get_T_f()

        #self.E_f = [f.get_E_f() for f in self.network.get_flows()]
        self.E_f = self.flow.get_E_f()
        #self.P_f = [f.get_P_f() for f in self.network.get_flows()]
        self.P_f = self.flow.get_P_f()

        #self.DELAY_f = [f.get_DELAY_f() for f in self.network.get_flows()]
        self.DELAY_f = self.flow.get_DELAY_f()
        #self.JITTER_f = [f.get_JITTER_f() for f in self.network.get_flows()]
        self.JITTER_f = self.flow.get_JITTER_f()
        self.SF_duration = self.network.get_T()
        
        # in the preprocessing the not available timeslots are set to 0
        self.T_e = [
            bitarray('1' * 2 * len(e.get_T_e())) # bitarray of available timeslots
            for e in self.network.links ]
        
        # b_eti pre-processing matrix of the available timeslots
        # first-level key is the tuple iteration, link
        # second level key is timeslot (id-1)
        # at each key of the dictionary correponds a bitarray of the available timeslots
        # for starting the specific iteration on the specific link
        self.w_fe = self.flow.get_w_fe()
        b_ie_t = {}

        for i in range(1,self.T_f + 1):
            for e1 in range(len(self.E_f)):
                e1 = self.E_f[e1]
                link = self.network.get_link_by_id(e1)
                bit_array = bitarray('0'*link.get_t_e())
                p_fe = ceil(self.P_f/link.get_tau_e())

                # set to 0 the time slots that are not assignable 
                # according to period  constraint.
                
                first_assignable = p_fe * (i-1) + 1
                w_size = self.flow.get_w_fe()[e1]

                bit_mask = bitarray('1'*w_size)
                # find all the fesible starting slot for the flow on the link
                iterator = link.get_T_e().search(bit_mask)

                # convert first assignable to the index of the timeslot
                first_assignable = first_assignable - 1
                required_time = 0

                for e2 in  range(e1 + 1, len(self.E)):
                    e2 = self.E_f[e2]
                    link2 = self.network.get_link_by_id(e2)
                    required_time += self.flow.get_w_fe()[e2] * link2.get_tau_e()
                
                for t in iterator:
                    if t >= first_assignable and (t+1) * link.get_tau_e() <= self.network.get_T() - required_time:
                        bit_array[t] = True

                if e1 != self.E_f[0]:
                    # enable the window
                    for t in range(1,w_size+1):
                        bit_array[-t] = True


                b_ie_t[(i,link.get_id())] = bit_array 
            
        self.b_ie_t = b_ie_t

        tock = self.now()
        self.pre_processing_time += self.format_delta(tick, tock)

        if debug:
            print("Network configuration")
            # write links
            for l in self.network.links: print(l)
            # write flows 
            for f in self.network.flows: print(f)
 
    def solve(self) -> dict:
        tick = self.now()

        prob = LpProblem("TSN - Scheduling Problem", LpMinimize)

        eti_space, i_space, ei_space = self.space()    
        x_eti   = LpVariable.dicts("x_eti", eti_space, cat="Binary")
        y_p     = LpVariable.dicts("y_p", i_space, cat="Integer")
        y_max       = LpVariable("y_max", cat="Integer")
        z       = LpVariable("z", cat="Integer")

        # constraint 3
        # all the iterations have to scheduled over each link of the path
        for i in range(1,self.T_f + 1):
            for e in self.E:
                T = ILP.bitarray_to_list(self.b_ie_t[(i,e)])
                prob += lpSum([x_eti[e,t,i] for t in T]) == 1

        # constraint 2
        # at most one iteration can starts in one timeslot 
        for e in self.E:
            link = self.network.get_link_by_id(e)
            t_e = link.get_t_e()
            w_fe = self.w_fe[e]
            for t in range(1,t_e+1):
                t_a = t + t_e 
                t_b = min(t_a + w_fe - 1, 2*t_e)
                prob += lpSum([x_eti[e,t,i] for i in range(1,self.T_f + 1)]) + lpSum([x_eti[e,tt,i] for i in range(1,self.T_f + 1) for tt in range(t_a,t_b+1)]) <= 1
        
        '''
        # constraint 4
        Reduced the space of the search
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

            prob += y_p[i]  == lpSum([(t-1) *  x_eti[e,t,i] for t in T2]) - p_f_dest * (i-1)
            prob += y_max >= y_p[i]
        
        prob+= y_max * tau_dest + d_e_dest <= self.DELAY_f
        
        # jitter constraint
        if self.T_f > 1:
            for i1 in range(1,self.T_f + 1):
                for i2 in range(1,self.T_f +1):
                    if i1 != i2 : 
                        prob+= z >= y_p[i1] - y_p[i2]
        else:
            prob+= z == 0
        prob += z * tau_dest <= self.JITTER_f
        
        for i in range(1,self.T_f+1):
            for e1,e2 in self.grouped(self.E_f):
                T1 = ILP.bitarray_to_list(self.b_ie_t[(i,e1)])
                T2 = ILP.bitarray_to_list(self.b_ie_t[(i,e2)])
                tau_e2 = self.network.get_link_by_id(e2).get_tau_e()
                tau_e1 = self.network.get_link_by_id(e1).get_tau_e()
                sigma = self.pipeline(e1,e2,self.flow.get_id())
                prob += lpSum([x_eti[e1,t,i] * (t-1) * tau_e1 for t in T1]) + sigma <= lpSum([tau_e2 * (t-1) * x_eti[e2,t,i] for t in T2])

        W1 = 100/self.JITTER_f
        W2 = 10/self.DELAY_f
        prob += W1 * z + W2 * y_max

        tock = self.now()
        self.loading_time = self.format_delta(tick, tock)

        self.tick = self.now()
        #prob.writeLP("scheduling.lp")
        #prob.writeMPS("scheduling.mps")
        print("Solving...")
        prob.solve(self.solver)
        #prob.solve()
  
        self.tock = self.now()
        self.solving_time = self.format_delta(self.tick, self.tock)

        if LpStatus[prob.status] == "Infeasible":
            return {"status": LpStatus[prob.status]}

        tau_dest = self.network.get_link_by_id(self.E_f[-1]).get_tau_e()
        self.x_eti = copy.deepcopy(x_eti)
        self.d_i = copy.deepcopy(y_p)
        self.d = copy.deepcopy(y_max.value() * tau_dest)
        self.j = copy.deepcopy(z.value() * tau_dest)
        print(self.d,self.j)
        self.objective = prob.objective

        return {
            "status": LpStatus[prob.status], 
            "objective": prob.objective
        }

    def post_processing(self) -> None:
        self.x_feti_opt = {}
        self.y_feti_opt = {}
        tick = self.now()
        for f in self.F:
            for e in self.E_f[f-1]:
                bound = int(0.5 * self.T_e[e-1][-1])
                for t in self.T_e[e-1]:  
                    t_opt = t
                    if t > bound : t_opt -= bound # circular shifting 
                    for i in range(1,self.T_f[f-1] + 1):
                        
                        if value(self.x_feti[f,e,t,i]) > 0:
                            self.x_feti_opt[f,e,t_opt,i] = 1
                        if value(self.y_feti[f,e,t,i]) > 0:
                            self.y_feti_opt[f,e,t_opt,i] = 1
                    

        tock = self.now()
        self.post_processing_time = self.format_delta(tick, tock)

    def store_data_plane(self, filename:str= "scheduling_results.pkl") -> None:
        y_feti = {}
        for f in self.F:
            for e in self.E_f[f-1]:
                for t in self.T_e[e-1]:
                    for i in range(1,self.T_f[f-1] + 1):
                        y_feti[f,e,t,i] = value(self.x_feti[f,e,t,i])

        with open(filename, 'wb') as f:
            pickle.dump(y_feti, f)

    def export_results(self, filename:str = "scheduling_results.xlsx") -> None:

        # xls export
        if not filename.endswith(".xlsx"): filename += ".xlsx"
        
        max_t = max([len(e.get_T_e()) for e in self.network.get_links() ])
        SF = range(1,2*max_t+1)
        df = pd.DataFrame(columns=SF, index=self.E)
        df_data = pd.DataFrame(columns=SF, index=self.E)
        for f in self.F:
            for i in range(1,self.T_f[f-1] + 1):
                for e in self.E_f[f-1]:
                    for t in self.T_e[e-1]:
                        if value(self.x_feti[f,e,t,i]) > 0:
                            df.loc[e,t] = f"{f}_{i}"

        for f in self.F:
            for i in range(1,self.T_f[f-1] + 1):
                for e in self.E_f[f-1]:
                    for t in self.T_e[e-1]:
                        if value(self.y_feti[f,e,t,i]) > 0:
                            df_data.loc[e,t] = f"{f}_{i}"

        # optimization
        SF = range(1,max_t+1)                    
        if self.x_feti_opt:
            df_opt = pd.DataFrame(columns=SF, index=self.E)
            for f in self.F:
                for i in range(1,self.T_f[f-1] + 1):
                    for e in self.E_f[f-1]:
                        for t in self.T_e[e-1]:
                            if (f,e,t,i) in self.x_feti_opt and value(self.x_feti_opt[f,e,t,i]) > 0:
                                df_opt.loc[e,t] = f"{f}_{i}"
                            

        if self.y_feti_opt:
            df_opt_y = pd.DataFrame(columns=SF, index=self.E)
            for f in self.F:
                for i in range(1,self.T_f[f-1] + 1):
                    for e in self.E_f[f-1]:
                        for t in self.T_e[e-1]:
                            if (f,e,t,i) in self.y_feti_opt and value(self.y_feti_opt[f,e,t,i]) > 0:
                                df_opt_y.loc[e,t] = f"{f}_{i}"
                            

        max_i = max([self.T_f[f-1] for f in self.F])
        df_kpi = pd.DataFrame(columns=range(1,max_i+1), index=self.F)
        for f in self.F:
            variances = []
            tau_dest = self.tau_e[self.E_f[f-1][-1]-1]
            for i in range(1,self.T_f[f-1] + 1):
                df_kpi.loc[f,i] = value(self.d_fi[f,i]) * tau_dest
                variances.append(value(self.d_fi[f,i]) * tau_dest)
            df_kpi.loc[f,max_i+1] = max(variances) - min(variances)
            df_kpi.loc[f,max_i+2] = np.var(variances) 

        df_changes = pd.DataFrame(columns=SF, index=self.E)
        for f in self.F:
            for e in self.E_f[f-1]:
                for t in self.T_e[e-1]:
                    if value(self.c_fet[f,e,t]) and value(self.c_fet[f,e,t]) >0 :
                        df_changes.loc[e,t] = f'c_{f}*'

        with pd.ExcelWriter(filename) as writer:
            df.to_excel(writer, sheet_name="Scheduling")
            df_data.to_excel(writer, sheet_name="DataPlane")
            if self.x_feti_opt:
                df_opt.to_excel(writer, sheet_name="SchedulingOptimized")
            if self.y_feti_opt:
                df_opt_y.to_excel(writer, sheet_name="DataPlaneOptimized")
            df_kpi.to_excel(writer, sheet_name="KPI")
            df_changes.to_excel(writer, sheet_name="Changes")

    def stat(self) -> dict:
        return {"pre_processing_time": self.pre_processing_time, "loading_time": self.loading_time, "solving_time": self.solving_time}
    
    def is_valid(self) -> bool:

        for link in self.network.get_links():
            bit_array = bitarray('1'*link.t_e)
            link.set_T_e(bit_array)

        for flow in self.network.get_flows():
            for iteration in range(1,flow.get_T_f()+1):
                for e_id in flow.get_E_f():
                    scheduled = False,None
                    for (f,e,t,i) in self.x_feti_opt.keys():
                        if f == flow.get_id() and e == e_id and i == iteration:
                            if self.x_feti[f,e,t,i] == 1:
                                if scheduled[0] == False:
                                    scheduled = True,t
                                elif scheduled[0] == True:
                                    print("Flow is scheduled twice", f,e,t,i)
                                    return False
                    
                    if scheduled[0] == False:
                        print("Flow is not scheduled (f,e,i)", flow.get_id(),e_id,iteration)
                        return False
                    
                    t = scheduled[1]

                    link = self.network.get_links()[e_id-1]
                    if link.get_T_e()[t-1] == 0:
                        print("Flow is scheduled on a non available time slot (f,e,i)", f.get_id(),e_id,iteration)
                        return False
                    else:
                        link.get_T_e()[t-1] = 0
                print(f"Flow {flow.get_id()} - iteration {iteration} is scheduled correctly")    
        return True
        
    def generate_gantt(self, file_name:str = 'gantt.png'):
        df = pd.DataFrame(columns=['flow', 'link', 'start', 'end', 'duration'])
        row_id = 0
        colors = {}

        for i in range(1, self.T_f + 1):
            color = np.random.rand(3,)
            colors[i] = color

        eti_space, i_space, etk_space = self.space()

        for (e,t,i) in eti_space:
            if ( self.x_eti[e,t,i].value() > 0.3):

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






