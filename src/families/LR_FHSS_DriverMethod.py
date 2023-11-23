import random
import numpy as np
from src.families.FHSfamily import FHSfamily

lr_fhss_lfsr_poly1 = [33, 45, 48, 51, 54, 57]
lr_fhss_lfsr_poly2 = [65, 68, 71, 72]

initial_state_EU = 6
initial_state_US = 56

frequency_gap_EU = 8
frequency_gap_US = 52

# get hop frequency in grid space and next lfsr state
def lr_fhss_get_next_state(lfsr_state, polynomial, xoring_seed, n_grid):
    
    hop = 0
    while 1:

        lsb = lfsr_state & 1
        lfsr_state >>= 1
        if lsb:
            lfsr_state ^= polynomial

        hop = xoring_seed
        if hop != lfsr_state:
            hop ^= lfsr_state

        if hop <= n_grid:
            break

    return lfsr_state, hop - 1


"""
Frequency hopping sequence generator based on lr_fhss driver implementation
"""
class LR_FHSS_DriverFamily(FHSfamily):

    def __init__(self, q, regionDR) -> None:
        super().__init__(q)
        self.init_params(regionDR)
        self.FHSfam = self.set_family()


    def init_params(self, regionDR):

        if regionDR == "EU137":
            self.grid_gap = frequency_gap_EU
            self.initial_state = initial_state_EU
            self.polynomials = lr_fhss_lfsr_poly1
            self.numFHS = 384
            self.numgrid = 35
            self.mask = 0x3F
            self.shift = 6

        elif regionDR == "EU336":
            self.grid_gap = frequency_gap_EU
            self.initial_state = initial_state_EU
            self.polynomials = lr_fhss_lfsr_poly2
            self.numFHS = 512
            self.numgrid = 86
            self.mask = 0x7F
            self.shift = 7

        elif regionDR == "US1523":
            self.grid_gap = frequency_gap_US
            self.initial_state = initial_state_US
            self.polynomials = lr_fhss_lfsr_poly1
            self.numFHS = 384
            self.numgrid = 60
            self.mask = 0x3F
            self.shift = 6
        
        else:
            raise Exception(f"Region '{regionDR}' not supported.") 


    # get lr fhss sequence of length q
    def get_lr_fhss_seq(self, id):

        lfsr_state = self.initial_state
        polynomial = self.polynomials[id >> self.shift]
        xoring_seed = id & self.mask

        fhs = []
        for _ in range(self.q):
            lfsr_state, hop = lr_fhss_get_next_state(lfsr_state, polynomial, xoring_seed, self.numgrid)
            fhs.append(hop)

        return np.array(fhs)


    def set_family(self):
        fam = []
        for id in range(self.numFHS):
            fhs = (self.get_lr_fhss_seq(id) * self.grid_gap) + random.randrange(0, self.grid_gap)
            fam.append(fhs)

        return np.array(fam)
    

    def get_random_sequence(self):
        seq_id = random.randrange(0, len(self.FHSfam))
        return seq_id, self.FHSfam[seq_id]
    
