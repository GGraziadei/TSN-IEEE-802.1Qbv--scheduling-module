import copy
import random

from bitarray import bitarray
from .directed_link import DirectedLink
from .flow import Flow
from .network import Network
from .solver import TSNScheduling
from .heuristics_model import Heuristics
import pandas as pd

# define a constant 
ALPHA = 0.3

class GRASP(Heuristics):

    def set_alpha(self, alpha:float):
        self.alpha = alpha
    
    def get_alpha(self):
        return self.alpha
    
    def set_execution_time(self, time:int): 
        self.time = time
    
    def get_execution_time(self):
        return self.time
    
    def select_candidate(self, candidates:dict, ch) -> tuple:
        
        alpha = self.alpha if self.alpha else ALPHA

        min_jitter = min(candidates.keys())
        max_jitter = max(candidates.keys())
        
        threshold = min_jitter + alpha * (max_jitter - min_jitter)
        threshold_candidates = {k: v for k, v in candidates.items() if k <= threshold}
        
        if threshold_candidates:
            candidate_jitter = random.choice(list(threshold_candidates.keys()))
            
            min_delay = min(threshold_candidates[candidate_jitter].keys())
            max_delay = max(threshold_candidates[candidate_jitter].keys())
            threshold = min_delay + alpha * (max_delay - min_delay)

            threshold_candidates = {k: v for k, v in threshold_candidates[candidate_jitter].items() if k <= threshold}
            if threshold_candidates:
                candidate_delay = random.choice(list(threshold_candidates.keys()))
                
                t_start_min = min(threshold_candidates[candidate_delay].keys())
                t_start_max = max(threshold_candidates[candidate_delay].keys())
                threshold = t_start_min + alpha * (t_start_max - t_start_min)

                threshold_candidates = {k: v for k, v in threshold_candidates[candidate_delay].items() if k <= threshold}
                
                if threshold_candidates:
                    best_candidate = random.choice(list(threshold_candidates.keys()))
                    return threshold_candidates[best_candidate], candidate_jitter, candidate_delay
                
        return None, None, None

    class LocalOptimal:
        def __init__(self, network, x_feti, d_fi, j_f) -> None:
            self.network = network
            self.x_feti = x_feti
            self.d_fi = d_fi
            self.j_f = j_f
        
        def get_max_jitter(self):
            return max(self.j_f.values())
        
        def get_max_delay(self):
            return max(self.d_fi.values())
        
        @staticmethod
        def compare(a, b):
            return a.get_max_jitter() < b.get_max_jitter() or (a.get_max_jitter() == b.get_max_jitter() and a.get_max_delay() < b.get_max_delay())
    
    def solve(self, d_fi=None, j_f=None, x_feti=None):
        
        optimal = None
        tick = self.now()
        _network = copy.deepcopy(self.network)

        if x_feti and d_fi and j_f:
            optimal = self.LocalOptimal(
                copy.deepcopy(self.network), 
                x_feti, 
                d_fi, 
                j_f)

        while True:
            # deep copy of the network
            
            
            sol = super().solve()
            
            if sol:
                local_solution = self.LocalOptimal(
                    copy.deepcopy(self.network), 
                    self.x_feti, 
                    self.d_fi, 
                    self.j_f)
                if not optimal or self.LocalOptimal.compare(local_solution, optimal):
                    optimal = local_solution

            tock = self.now()
            self.network = _network

            if (tock - tick).seconds > self.time:
                if optimal:
                    self.x_feti = optimal.x_feti
                    self.d_fi = optimal.d_fi
                    self.j_f = optimal.j_f
                    return True
                return False

    