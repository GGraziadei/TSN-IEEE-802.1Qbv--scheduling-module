from .BRKGA_fwk.decoder import _Decoder
from .heuristics_model import Heuristics
from copy import deepcopy

class BRKGADecoder(_Decoder,Heuristics):

    def get_number_genes(self):
        genes = 0
        for e in self.network.links:
            genes += e.get_t_e()
        return genes
    
    def select_candidate(self, candidates:dict, ch) -> tuple:
        best_jitter = min(candidates.keys())
        best_delay = min(candidates[best_jitter].keys())
        
        t_dict = { t : t*ch[t-1] for t in candidates[best_jitter][best_delay].keys() }
        min_val = max(t_dict.values())
        
        for k,v in t_dict.items():
            if v < min_val:
                min_val = v
                t_min = k

        best_candidate = candidates[best_jitter][best_delay][t_min]

        return best_candidate, best_jitter, best_delay
    
    def decodeIndividual(self, chromosome):
        
        network = deepcopy(self.network)

        sol, x_feti, d_fi, j_f = super().solve(chromosome)
        
        if sol:
            j = max(j_f.values()) 
            d = max(d_fi.values())
            solution = x_feti
        else:
            j = None
            d = None
            solution = None

        self.network = network

        return solution,j,d,d_fi,j_f