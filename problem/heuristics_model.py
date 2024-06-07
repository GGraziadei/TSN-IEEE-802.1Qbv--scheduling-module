from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
import copy
import concurrent.futures
from math import ceil
from bitarray import bitarray

from .flow import Flow
from .solver import TSNScheduling
import pandas as pd

class Heuristics(TSNScheduling):
    
    def pre_processing(self):
        return super().pre_processing()

    def find_available_t(self, e_id, step, t_min=None) -> int|None:
        bin_array = self.network.get_links()[e_id-1].get_T_e()[t_min-1:]
        bin_search = bitarray(step * '1')
        # example: f with 1/4 of SF mask 1000 = 0x08
        # left logical shift of the mask  = all multiple of 8 (8,16, ...., SF_duration)
        iterator = bin_array.search(bin_search, 1)
        if iterator: 
            return [i + t_min - 1 for i in iterator]

        bin_array = self.network.get_links()[e_id-1].get_T_e()[:t_min-1]
        iterator = bin_array.search(bin_search,1)
        return iterator

    def find_first_t(self, e_id, t_min, step) -> int|None:

        iterator = self.find_available_t(e_id, step, t_min)
        # shift mask to optimize the reasearch of the first t
        
        circular_corresponding = []
        for i in iterator:
            t = 1 + i
            if t >= t_min: return t
            else : circular_corresponding.append(t)
        
        if circular_corresponding: 
            return circular_corresponding[0]
                                        
        return None
    
    def candidate(self, flow, path, t_min, p_f_t_start, max_d) -> dict:
        t_start = t_min
        e_src = path.pop(0)
        tau_e_src = self.network.get_link_by_id(e_src).get_tau_e()

        candidate = [(e_src, t_min)]
        t_e = self.network.get_link_by_id(e_src).get_t_e()
        if t_min > p_f_t_start:
                delay = (t_min - p_f_t_start - 1) * self.network.get_link_by_id(e_src).get_tau_e()
        else: 
            delay = (t_min + t_e - p_f_t_start - 1) * self.network.get_link_by_id(e_src).get_tau_e()
        t_prev = (t_min -1) * self.network.get_link_by_id(e_src).get_tau_e()
        
        if len(path) > 0:
            e_dest = path[0]
            tau_e_dest = self.network.get_link_by_id(e_dest).get_tau_e()
            sigma = self.pipeline(e_src,e_dest,flow_id=flow["flow"].get_id())
            t_min = ceil(((t_min-1) * tau_e_src + sigma) / tau_e_dest) + 1
        
        d_e_last = self.network.get_link_by_id(e_src).get_d_e()
            
        while path:
            e_id = path.pop(0)
            step = flow["flow"].get_w_fe()[e_id]
            t = self.find_first_t(e_id, t_min, step)
            if t:
                
                #check delay constraint 
                # the max_d value is the maximum delay admitted according to balance the jitter
                t_complement = (t-1) * self.network.get_link_by_id(e_id).get_tau_e()
                if t_complement > t_prev: delay += (t_complement - t_prev)
                else: delay += (t_complement + self.network.T - t_prev)

                # if the constraint is violated reject the candidate
                if  delay > max_d:
                    return None

                if path:
                    e_dest = path[0]
                    tau_e_dest = self.network.get_link_by_id(e_dest).get_tau_e()
                    tau_e_src = self.network.get_link_by_id(e_id).get_tau_e()
                    sigma = self.pipeline(e_id,e_dest,flow_id=flow["flow"].get_id())
                    t_min = ceil(((t_min-1) * tau_e_src + sigma) / tau_e_dest) + 1

                if t_min > len(self.network.get_link_by_id(e_id).get_T_e()):
                    # circular buffer optimization
                    t_min = 1

                candidate.append((e_id, t))
                t_prev = (t -1) * self.network.get_link_by_id(e_id).get_tau_e()
                d_e_last = self.network.get_link_by_id(e_id).get_d_e()
                if delay + d_e_last > max_d: return None
                
            else: continue
        
        if len(candidate) == len(flow["flow"].get_E_f()):
            
            delay += d_e_last            
            last_i, last_t = flow["flow"].get_last_t()
            
            new_jitter = 0
            if last_i:
                old_max_delay = flow["flow"].get_max_delay()
                old_min_delay = flow["flow"].get_min_delay()
                new_max_delay = max(old_max_delay, delay)
                new_min_delay = min(old_min_delay, delay)
                
                if (new_max_delay-new_min_delay) > flow["flow"].get_JITTER_f():
                    return None

                # variation of the jitter
                new_jitter = (new_max_delay - new_min_delay) - (old_max_delay - old_min_delay)
            
            return {
                "candidate" : candidate,
                "delay" : delay,
                "jitter" : new_jitter,
                "t_start" : t_start
            }
        
    def candidates(self, flow):
        
        max_d = flow["flow"].get_DELAY_f()

        min_latency = flow["flow"].get_min_latency()
        e_src = flow["flow"].get_E_f()[0]
        tau_e_src = self.network.get_links()[e_src-1].get_tau_e()
        p_f = flow["flow"].get_P_f()

        # mapping of the period to the time division over the e_src
        p_f_t = ceil(p_f/tau_e_src)
        p_f_t_start = ceil(p_f_t * (flow["iteration"]-1)) 
        
        t_e = self.network.get_links()[e_src-1].get_t_e()
        step = flow["flow"].get_w_fe()[e_src]

        # the starting point is the last t allocated after the period 
        # starting point according to the time division
        t_min = flow["t_min"] + p_f_t * (flow["iteration"]-1)
        last_i, last_t = flow["flow"].get_last_t()
        if last_i: t_min = max(last_t, t_min)

        # heuristics over the starting scheduling point 
       
        # the starting point cannot execed the infeasibility of other iteration 
        # of the same flow
        flow_required = p_f_t * (flow["flow"].get_T_f() - flow["iteration"]) 
        min_required_f = t_e - t_min - flow_required
        if min_required_f < 0: min_required_f = 0

        # the starting point cannot execed the infeasibility of the current iteration
        # given the minimum latency and the maximum delay admitted 
        # the starting point cannot be start in an infeasibility region
        max_d_t = int(max_d/tau_e_src)
        min_latency_t = ceil(min_latency/tau_e_src)
        min_required_i = max_d_t - min_latency_t
        if min_required_i < 0: min_required_i = 0

        candidates = {}

        # find the restriction of the starting point
        t_min_delta = 1 + min(min_required_i,min_required_f)
        lowerbound = t_min 

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            futures = []
            for t_start in range(t_min, t_min + t_min_delta, step):
                if t_start < lowerbound: continue
                t_min = self.find_first_t(e_src, t_start, step)
                if not t_min: continue
                lowerbound = t_min

                path = copy.deepcopy(flow["flow"].get_E_f())
                #c = self.candidate(flow, path, t_min, p_f_t_start, max_d)
                future = executor.submit(self.candidate, flow, path, t_min, p_f_t_start, max_d)
                futures.append(future)

        for c in futures:
            result = c.result()
            if result:
                candidate = result["candidate"]
                delay = result["delay"]
                new_jitter = result["jitter"]
                start = result["t_start"]

                #if delay <  max_d:  max_d = delay

                if new_jitter not in candidates:
                    candidates[new_jitter] = {delay: {start: candidate}}
                elif delay not in candidates[new_jitter]:
                    candidates[new_jitter][delay] = {start: candidate}
                else:
                    candidates[new_jitter][delay][start] = candidate    

        return candidates
    
    def select_candidate(self, candidates:dict, ch= None) -> tuple:
        best_jitter = min(candidates.keys())
        best_delay = min(candidates[best_jitter].keys())
        t_min = min(candidates[best_jitter][best_delay].keys())

        best_candidate = candidates[best_jitter][best_delay][t_min]

        return best_candidate, best_jitter, best_delay

    def solve(self, ch = None):

        flows_iterations = []
        for f in self.network.flows:
            for i in range(1, f.get_T_f() + 1):
                flows_iterations.append({   
                             "flow" : f, 
                             "iteration" : i, 
                             "t_min" : 1, 
                             "partioned_delay" : f.get_DELAY_f() / len(f.get_E_f()),
                             "scritcality" : f.get_DELAY_f() / f.get_min_latency(),
                            })
                
        #print([f'{f["flow"].get_id()} -- {f["iteration"]}' for f in flows_iterations])
        return self._solve(flows_iterations, ch)

    def _solve(self, flows_iterations, ch = None) -> bool|None: 
        tick = self.now()

        x_feti = {}
        d_fi = {}
        j_f = {}
        
        flows_iterations.sort(key=lambda x: x["scritcality"] * x["partioned_delay"])


        for flow in flows_iterations:
            candidates = self.candidates(flow)
            if not candidates:
                print(f"Flow {flow['flow'].get_id()}_{flow['iteration']} has no candidates")
                return False
            
            best_candidate, best_jitter, best_delay = self.select_candidate(candidates, ch)
            print(f"Flow {flow['flow'].get_id()}_{flow['iteration']} -- {best_jitter} -- {best_delay} -- {best_candidate}")
                  
            if best_delay < flow["flow"].get_min_delay():
                flow["flow"].set_min_delay(best_delay)
            if best_delay > flow["flow"].get_max_delay():
                flow["flow"].set_max_delay(best_delay)
            
            for e_id, t in best_candidate:
                e = self.network.get_link_by_id(e_id)
                t_e = e.get_t_e()
                step = flow["flow"].get_w_fe()[e_id]
                for k in range(0, step):
                    _t = t + k
                    if _t > t_e: _t = _t - t_e
                    self.network.get_links()[e_id-1].remove_t(_t)
                x_feti[(flow["flow"].get_id(), e_id, t, flow["iteration"])] = 1
                d_fi[(flow["flow"].get_id(), flow["iteration"])] = best_delay
            
            t_start = best_candidate[0][1]
            flow["flow"].set_last_t((flow["iteration"], t_start))

        for flow in self.network.flows: 
            j_f[flow.get_id()] = flow.get_max_delay() - flow.get_min_delay()

        tock = self.now()
        
        self.soling_time = tock - tick
        self.x_feti = x_feti
        self.d_fi = d_fi
        self.j_f = j_f
        
        return True

    def remove(self, flow : Flow, iteration : int):
        removing = []
        for (f, e, t, i) in self.x_feti.keys():
            if flow.get_id() == f and iteration == i:
                step = flow.get_w_fe()[e]
                t_e = self.network.get_link_by_id(e).get_t_e()
                for k in range(0, step):
                    if k + t > t_e:
                        self.network.get_links()[e-1].add_t(t+k-t_e)
                    else:
                        self.network.get_links()[e-1].add_t(t+k)
                removing.append((f,e,t,i))
        for r in removing:
            del self.x_feti[r]

    def ls(self):
        tick = self.now()
        for f in self.network.flows:
            #while not j_f or  j_f[f.get_id()] > self.network.flows[f.get_id()-1].get_JITTER_f():
            for iteration in range(1, f.get_T_f() + 1):
            
                if self.d_fi[f.get_id(), iteration] < f.get_max_delay():

                    network = copy.deepcopy(self.network)
                    x_feti = copy.deepcopy(self.x_feti)
                    d_fi = copy.deepcopy(self.d_fi)
                    j_f = copy.deepcopy(self.j_f)
                    
                    self.remove(flow=f, iteration=iteration)
                    constructive = {   
                             "flow" : f, 
                             "iteration" : iteration, 
                             "t_min" : 1, 
                             "partioned_delay" : f.get_DELAY_f() / len(f.get_E_f()),
                             "scritcality" : f.get_DELAY_f() / f.get_min_latency(),
                            }
                    
                    f.set_last_t((iteration, 1))
                    if not self._solve([constructive]):
                        self.network = network
                        self.x_feti = x_feti
                        self.d_fi = d_fi
                        self.j_f = j_f
                        return False
                    else:
                        for k in d_fi.keys():
                            if not k in self.d_fi:
                                self.d_fi[k] = d_fi[k]
                        for k in j_f.keys():
                            if not k in self.j_f:
                                self.j_f[k] = j_f[k]
                        for k in x_feti.keys():
                            if not k in self.x_feti:
                                self.x_feti[k] = x_feti[k]
                                
        tock = self.now()
        self.ls_time = tock - tick
        return True


    def find_overlapping(self, t_min, flow) -> list:
        e_src = flow.get_E_f()[0]
        step = flow.get_w_fe()[e_src]
        overlapping = []
        for (f,e,t,i) in self.x_feti.keys():
            if e == e_src and t_min <= t < t_min + step:
                overlapping.append((f,e,t,i))

        return overlapping

    def remove_overlapping(self, overlap) -> list:
        _f,e,t,_i = overlap

        overlap_schedule = []
        remove = []
        for (f,e,t,i) in self.x_feti.keys():
            if f == _f and i == _i:
                overlap_schedule.append((f,e,t,i))
                remove.append((f,e,t,i))
                w_fe = self.network.get_flows()[f-1].get_w_fe()[e]
                for k in range(0, w_fe):
                    self.network.get_links()[e-1].add_t(t+k)
        for r in remove:
            del self.x_feti[r]
        return overlap_schedule
    
    def restore_overlapping(self, overlap_schedule) -> None:
        for (f,e,t,i) in overlap_schedule:
            self.x_feti[f,e,t,i] = 1
            for tt in range(t, t + self.network.get_flows()[f-1].get_w_fe()[e-1]):
                self.network.get_links()[e-1].remove_t(tt)
    
    def overlap_to_flow(self, overlap) -> dict:
        f,e,t,i = overlap
        flow = self.network.get_flows()[f-1]
        return {
            "flow" : flow,
            "iteration" : i,
            "t_min" : 1, 
            "partioned_delay" : flow.get_DELAY_f() / len(flow.get_E_f()),
            "scritcality" : flow.get_min_latency(),
        }
    
    def local_search(self, to_allocate, allowed_steps) -> tuple|None:
        step = 0 # avoid circular depdency
        to_allocate.sort(key=lambda x: x["scritcality"] * x["partioned_delay"])
        to_allocate_dep = []
        while to_allocate:
            if step > allowed_steps: return None
            flow = to_allocate.pop(0)
            # search overlapping
            t_min = flow["t_min"]

            flow_overlapping = self.find_overlapping(t_min, flow["flow"])

            # search for the best candidate
            # per each overlapp try to allocate the flow
            # if not possible the flow cannot be accepted
            for overlap in flow_overlapping:
                self.remove_overlapping(overlap)
                to_allocate_dep.append(self.overlap_to_flow(overlap))
            to_allocate_dep.append(flow)
            step += 1
        
        sol, x_feti, d_fi, j_f = self._solve(to_allocate_dep)
        if sol is not None:
            self.x_feti = x_feti
            self.d_fi = d_fi
            self.j_f = j_f
        return sol is not None

    def flow_prepare(self, flow : Flow) -> None:
        flow_id = len(self.network.flows) + 1
        flow.set_id(flow_id) # expected with id -1
        flow.set_T_f(int(self.network.get_T() / flow.get_P_f() ))
        w_fe = {}
        for e in flow.get_E_f():
            w_fe[e] = ceil(flow.get_size_f() / self.network.get_link_by_id(e).get_a_e())
        flow.set_w_fe(w_fe)
        print(flow)
        return flow
    
    def new_request(self, flow : Flow, ch = None) :

        flow = self.flow_prepare(flow)
        tick = self.now()
        network = copy.deepcopy(self.network)
        incumbent = copy.deepcopy(self.x_feti)
        self.network.flows.append(flow)
        
        flows_iterations = [{
                             "flow" : flow, 
                             "iteration" : i, 
                             "t_min" : 1,
                             "partioned_delay" : flow.get_DELAY_f() / len(flow.get_E_f()),
                             "scritcality" : flow.get_min_latency(),
                            } for i in range(1, flow.get_T_f() + 1)]        

        flows_iterations.sort(key=lambda x: x["scritcality"] * x["partioned_delay"])
        
        #print([f'{f["flow"].get_id()} -- {f["iteration"]}' for f in flows_iterations])
        to_allocate = []
        for flow in flows_iterations:
            candidates = self.candidates(flow)
            
            if not candidates:
                print(f"Flow {flow['flow'].get_id()} has no candidates - local search needed")
                to_allocate.append(flow)
                return False
                
            best_candidate, best_jitter, best_delay = self.select_candidate(candidates, ch)
            print(f"Flow {flow['flow'].get_id()}_{flow['iteration']} -- {best_jitter} -- {best_delay} -- {best_candidate}")
                  
            if best_delay < flow["flow"].get_min_delay():
                flow["flow"].set_min_delay(best_delay)
            if best_delay > flow["flow"].get_max_delay():
                flow["flow"].set_max_delay(best_delay)

            step = flow["flow"].get_w_fe()[flow["flow"].get_E_f()[0]]
            for e_id, t in best_candidate:
                e = self.network.get_link_by_id(e_id)
                t_e = e.get_t_e()
                for k in range(0, step):
                    _t = t + k
                    if _t > t_e: _t = _t - t_e
                    self.network.get_links()[e_id-1].remove_t(_t)
                self.x_feti[(flow["flow"].get_id(), e_id, t, flow["iteration"])] = 1
                self.d_fi[(flow["flow"].get_id(), flow["iteration"])] = best_delay
            
            t_start = best_candidate[0][1]
            flow["flow"].set_last_t((flow["iteration"], t_start))

        # local search
        if to_allocate:
            max_steps = sum([f.get_T_f() for f in self.network.flows]) - len(to_allocate)
            ls_res = self.local_search(to_allocate, max_steps)
        
            if not ls_res:
                print(f"The request is not feasible, id {flow['flow'].get_id()} released")
                self.network = network
                self.x_feti = incumbent
                return False
        
        for flow in self.network.flows: 
            self.j_f[flow.get_id()] = flow.get_max_delay() - flow.get_min_delay()

        tock = self.now()
        
        self.soling_time = tock - tick
        print(f"New request {flow.get_id()} solved in {self.soling_time} seconds")
        return True
    
    def stat(self):
        return {
            "pre_processing_time" : self.pre_processing_time,
            "solving_time" : self.soling_time
        }

    def export_results(self, filename:str = "scheduling_results.xlsx") -> None:

        # xls export
        if not filename.endswith(".xlsx"): filename += ".xlsx"

        max_t = max([len(e.get_T_e()) for e in self.network.get_links() ])
        SF = range(1,max_t+1)
        df = pd.DataFrame(columns=SF, index=range(1,len(self.network.get_links())+1))
        for (f,e,t,i) in self.x_feti.keys():
            w_fe = self.network.get_flows()[f-1].get_w_fe()[e]
            for j in range(0, w_fe):
                df.loc[e,(t+j)] = f"{f}_{i}"

        df_kpi = pd.DataFrame(columns=range(1,max([f.get_T_f() for f in self.network.flows])+2 ),index=range(1,len(self.network.flows)+1))
        for (f,i) in self.d_fi.keys():
            df_kpi.loc[f, i] =  self.d_fi[(f,i)]
            df_kpi.loc[f, max([f.get_T_f() for f in self.network.flows])+1] =  self.j_f[f]
          
        with pd.ExcelWriter(filename) as writer:
            df.to_excel(writer, sheet_name="Scheduling")
            df_kpi.to_excel(writer, sheet_name="KPI")

            
            

                

                




                        
                    
            
                
