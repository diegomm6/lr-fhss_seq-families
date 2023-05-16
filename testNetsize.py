from multiprocessing import Pool
import numpy as np
from base import *
import galois
from LempelGreenbergMethod import LempelGreenbergFamily
from LR_FHSS_DriverMethod import LR_FHSS_DriverFamily
from LiFanMethod import LiFanFamily
from HashMethod import HashFamily
from simulation import Simulation

# this script is designed to test multiple values for the network size for a 
# sinlge family, using parallel computation

def get_family():
    p = 2
    k = 5
    n = 5
    polys = galois.primitive_polys(p, n)
    poly1 = next(polys)
    poly2 = next(polys)

    lempelGreenbergGenerator = LempelGreenbergFamily(p=p, n=n, k=k, poly=poly1)
    lempelGreenberg_fam = lempelGreenbergGenerator.get_optimal_family()
    lempelGreenberg_fam = lempelGreenberg_fam *8

    lempelGreenbergGenerator2 = LempelGreenbergFamily(p=p, n=n, k=k, poly=poly2)
    lempelGreenberg_fam2 = lempelGreenbergGenerator2.get_optimal_family()
    lempelGreenberg_fam2 = lempelGreenberg_fam2 *8

    lempelGreenberg_fam3 = np.concatenate((lempelGreenberg_fam, lempelGreenberg_fam2))
    return lempelGreenberg_fam3


# test a single method for several number of nodes
def get_avg_collisions(v):

    runs = 30
    numOCW = 7
    numOBW = 280
    numGrids = 8
    seq_length = 31
    startLimit = 500
    useGrid = True
    family = get_family()

    nodes = int(v)

    simulation = Simulation(nodes=nodes, family=family, useGrid=useGrid, numOCW=numOCW,
                            numOBW=numOBW, numGrids=numGrids, startLimit=startLimit, seq_length=seq_length)

    avg = 0
    for r in range(runs):
        txData = simulation.run()
        avg += (txData > 1).sum()

    return avg / runs


if __name__ == "__main__":

    netSizes = [1e3, 2e3, 3e3, 5e3, 1e4, 2e4, 3e4, 5e4, 1e5, 2e5, 3e5, 5e5, 1e6]

    pool = Pool(processes = len(netSizes))
    result = pool.map(get_avg_collisions, netSizes)
    pool.close()
    pool.join()

    for i in range(len(netSizes)):
        print(f"nodes = {netSizes[i]}, collided slots = {result[i]}")

    print('\n', result)
