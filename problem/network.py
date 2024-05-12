from .flow import Flow
from .directed_link import DirectedLink

class Network:

    def __init__(self, T: int, links: list[DirectedLink] = [],  flows: list[Flow] = []):
        self.links = links
        self.T = T
        self.flows = flows
        self.h_ij = {}
    
    def __str__(self):
        return f"Network with SF_duration: {self.T}"
    
    def get_links(self):
        return self.links
    
    def get_T(self):
        return self.T
    
    def set_T(self, T):
        self.T = T
    
    def set_links(self, links):
        self.links = links

    def get_flows(self):
        return self.flows
    
    def set_flows(self, flows):
        self.flows = flows
    
    def get_T_f(self) -> list[int]:
        return [ int(self.T/f.get_P_f()) for f in self.flows ]
    
    def get_link_by_id(self, id):
        for l in self.links:
            if l.get_id() == id:
                return l
        return None
    
    def links_stats(self):
        stats = {}
        for l in self.links:
            stats[l.get_id()] = l.stats()
        return stats