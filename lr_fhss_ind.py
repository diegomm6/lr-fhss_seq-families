import numpy as np
from multiprocessing import Pool

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

def lfrs_get_next_state(lfsr_state, polynomial, xoring_seed, n_grid):
    
    lsb = lfsr_state & 1
    lfsr_state >>= 1

    if lsb:
        lfsr_state ^= polynomial

    return lfsr_state


# get lr fhss sequence of length 31
def get_lr_fhss_seq(id):
    n_grid = 35
    lr_fhss_lfsr_poly1 = [33, 45, 48, 51, 54, 57]

    lfsr_state = 6
    fhs = []

    polynomial = lr_fhss_lfsr_poly1[id >> 6]
    xoring_seed = id & 0x3F

    for _ in range(31):
        lfsr_state, hop = lr_fhss_get_next_state(lfsr_state, polynomial, xoring_seed, n_grid)
        fhs.append(hop)

    # print(f"id={id}\tpoly={polynomial}\tseed={xoring_seed}\nseq = {fhs}\n")

    return fhs

def get_lr_fhss_family():
    fam = []
    for id in range(384):
        fam.append(get_lr_fhss_seq(id))

    return np.array(fam)


lr_fhss_family = get_lr_fhss_family()*8

# array representoing the number of transmissions
# on each channed OCW and sub-channel OBW
numOCW = 7
numOBW = 280
numGrids = 8
seq_length = 31
startLimit = 500


def get_seq_time(family, startLimit):
    startTime = np.random.randint(startLimit)
    seq_id = np.random.randint(len(family))
    return family[seq_id], startTime


# default FHS generator with hash function
def transmissions_simulation(nodes, family, useGrid):
    transmissions = np.zeros((numOCW, numOBW, startLimit + seq_length))

    for n in range(nodes):
        ocw = np.random.randint(numOCW)
        grid = 0
        if useGrid:
            grid = np.random.randint(numGrids)

        # choose random sequence and starting time
        seq, t0 = get_seq_time(family, startLimit)

        for t, obw in enumerate(seq):
            transmissions[ocw][obw + grid][t0 + t] += 1

    return transmissions


def get_avg_collisions(v):
    nodes, runs, family, useGrid = v
    avg = 0
    for r in range(runs):
        tx = transmissions_simulation(nodes, family, useGrid)
        avg += (tx > 1).sum()
    return avg / runs


def main():
    runs = 30
    pool = Pool(processes=4)  # set the processes max number 4
    result = pool.map(get_avg_collisions,
                      [(1500, runs, lr_fhss_family, True),
                       (300, runs, lr_fhss_family, True),
                       (300, runs, lr_fhss_family, True)])
    pool.close()
    pool.join()

    n5000, n50000, n500000 = result

    print(f"n1500 collisions = {n5000}")
    print(f"n300 collisions = {n50000}")
    print(f"n300 collisions = {n500000}")



if __name__ == "__main__":
    main()


