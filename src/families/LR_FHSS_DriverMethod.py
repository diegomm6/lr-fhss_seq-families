import numpy as np
from src.families.FHSfamily import FHSfamily

n_grid = 35
lr_fhss_lfsr_poly1 = [33, 45, 48, 51, 54, 57]
initial_state = 6

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


# get lr fhss sequence of length q
def get_lr_fhss_seq(id, q):

    fhs = []
    lfsr_state = initial_state
    polynomial = lr_fhss_lfsr_poly1[id >> 6]
    xoring_seed = id & 0x3F

    for _ in range(q):
        lfsr_state, hop = lr_fhss_get_next_state(lfsr_state, polynomial, xoring_seed, n_grid)
        fhs.append(hop)

    return fhs


"""
Frequency hopping sequence generator based on lr_fhss driver implementation
this model is currently limited for the case with 280 obw and 8 grids, so 35 channels per grid
which allows 384 different sequences of a yet undetermined period
The only parameter is the length of the sequences to be generated
"""
class LR_FHSS_DriverFamily(FHSfamily):

    def __init__(self, q) -> None:
        super().__init__(q)
        self.m = 384
        self.FHSfam = []

    def set_family(self, numGrids):
        fam = []
        for id in range(self.m):
            fam.append(get_lr_fhss_seq(id, self.q))

        self.FHSfam = (np.array(fam) * numGrids) + np.random.randint(numGrids)

    def get_random_sequence(self):
        seq_id = np.random.randint(len(self.FHSfam))
        return self.FHSfam[seq_id]
    