from multiprocessing import Pool
import numpy as np
import galois
from src.base import *
from src.families.LempelGreenbergMethod import LempelGreenbergFamily
from src.families.LR_FHSS_DriverMethod import LR_FHSS_DriverFamily
from src.families.LiFanMethod import LiFanFamily
from src.families.HashMethod import HashFamily
from src.families.WangMethod import WangFamily
from src.models.simulation import Simulation

# this script is designed to test multiple values for the network size for a 
# sinlge family, using parallel computation

def get_family():
    p = 37
    q = 31
    w = 8
    d = 8
    wangGenerator = WangFamily(p=p, q=q, w=w, d=d)
    wangFamily = wangGenerator.get_family()

    return wangFamily


# test a single method for several number of nodes
def get_avg_collisions(v):

    runs = 30
    numOCW = 7
    numOBW = 280
    numGrids = 8
    seq_length = 31
    startLimit = 500
    useGrid = False
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
