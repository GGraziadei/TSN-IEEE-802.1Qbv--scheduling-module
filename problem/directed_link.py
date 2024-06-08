from enum import Enum

class DirectedLink:

    class TransmissionMode(Enum):
        FAST = 1
        STORE_AND_FORWARD = 2

        def __str__(self):
            return self.name.tolower()
        
        def serialize(self):
            return self.name
        
        def deserialize(self, data):
            return self(data)
        
    def __init__(self, id, B_e, a_e, tau_e, d_e, g_e,mode=TransmissionMode.FAST,  T_e = []):
        self.id = id
        self.B_e = B_e
        self.tau_e = tau_e
        self.T_e = T_e
        self.a_e = a_e
        self.d_e = d_e
        self.g_e = g_e
        self.t_e = 0
        self.mode = mode # Transmission mode
    
    def __str__(self):
        return f"- DirectedLink {self.id}\n\tB_e(bit/s): {self.B_e}\n\ttau_e(us): {self.tau_e}\n\ta_e(bit): {self.a_e}\n\tt_e(time slot): {self.t_e}"
    
    def get_id(self):
        return self.id
    
    def get_B_e(self):
        return self.B_e
    
    def set_id(self, id):
        pass

    def set_tau_e(self, tau_e):
        self.tau_e = tau_e

    def get_tau_e(self):
        return self.tau_e
    
    def set_B_e(self, B_e):
        self.B_e = B_e

    def get_T_e(self):
        return self.T_e
    
    def set_T_e(self, T_e):
        self.T_e = T_e
        
    def add_t(self, t):
        self.T_e[t-1] = True
        
    def remove_t(self, t):
        if 1 <= t <= len(self.T_e):
            self.T_e[t-1] = False

    def get_a_e(self):
        return self.a_e

    def set_a_e(self, a_e):
        self.a_e = a_e

    def get_t_e(self):
        return self.t_e
    
    def set_t_e(self, t_e):
        self.t_e = t_e

    def get_d_e(self):
        return self.d_e
    
    def set_d_e(self, d_e):
        self.d_e = d_e

    def get_g_e(self):
        return self.g_e
    
    def set_g_e(self, g_e):
        self.g_e = g_e

    def get_mode(self):
        return self.mode
    
    def set_mode(self, mode):
        self.mode = mode

    def stats(self):
        from bitarray import bitarray
        length = len(self.get_T_e())
        ones = self.get_T_e().count(0)
        return { 'load' : ones/length, 'buffer_info': self.get_T_e().buffer_info() }
        
    def fragmentation(self):
        import bitarray
        return bitarray.intervals(self.get_T_e())