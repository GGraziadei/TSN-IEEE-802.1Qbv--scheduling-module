
from enum import Enum


class Flow:

    def __init__(self, id, E_f, P_f, DELAY_f, JITTER_f, size_f, T_f = None):
        self.id = id
        self.E_f = E_f
        self.P_f = P_f
        self.T_f = T_f
        self.DELAY_f = DELAY_f
        self.JITTER_f = JITTER_f
        self.size_f = size_f
        self.w_fe = {}
        
        # the last allocation for the flow - heuristics required
        self.last_t = None,1
        # minimum latency for the flow
        self.min_latency = 0
    
        self.max_delay = 0
        self.min_delay = self.DELAY_f
        
    def __str__(self):
        # return a table per flow with table header
        msg = f"Flow {self.id}\n\tE_f: {self.E_f}\n\tP_f(us): {self.P_f}\n\tDELAY_f(us): {self.DELAY_f}\n\tJITTER_f(us): {self.JITTER_f}\n\tsize_f (bit): {self.size_f}\n\tT_f: {self.T_f}\n\t"
        msg += "Window size:\n"
        for w in self.w_fe:
            msg += f"\t{self.w_fe[w]}"
        return msg
    
    def get_id(self):
        return self.id
    
    def get_E_f(self):
        return self.E_f
    
    def get_P_f(self):
        return self.P_f
    
    def get_T_f(self):
        return self.T_f
    
    def get_DELAY_f(self):
        return self.DELAY_f
    
    def get_JITTER_f(self):
        return self.JITTER_f
    
    def set_id(self, id):
        self.id = id

    def set_E_f(self, E_f):
        self.E_f = E_f

    def set_P_f(self, P_f):
        self.P_f = P_f
    
    def set_T_f(self, T_f):
        self.T_f = T_f

    def set_DELAY_f(self, DELAY_f):
        self.DELAY_f = DELAY_f

    def set_JITTER_f(self, JITTER_f):
        self.JITTER_f = JITTER_f

    def get_last_t(self):
        return self.last_t

    def set_last_t(self, last_t):
        self.last_t = last_t

    def get_min_latency(self):
        return self.min_latency
    
    def set_min_latency(self, min_latency):
        self.min_latency = min_latency
    
    def get_max_delay(self):
        return self.max_delay
    
    def set_max_delay(self, max_delay):
        self.max_delay = max_delay
    
    def get_min_delay(self):
        return self.min_delay
    
    def set_min_delay(self, min_delay):
        self.min_delay = min_delay

    def get_size_f(self):
        return self.size_f
    
    def set_size_f(self, size_f):
        self.size_f = size_f

    def get_w_fe(self):
        return self.w_fe
    
    def set_w_fe(self, w_fe):
        self.w_fe = w_fe
    

