from multiprocessing import Pool
import numpy as np
from base import *
from WangMethod import WangFamily
from simulationCR import SimulationCR


# this script is designed to test multiple values for the network size for a 
# sinlge family, using parallel computation

def get_family():
    p = 37
    q = 31
    w = 8
    d = 8
    wangGenerator = WangFamily(p=p, q=q, w=w, d=d)
    wangFamily = wangGenerator.get_OCWGFHSfamily()

    return wangFamily


# test a single method for several number of nodes
def get_avg_packet_collision_rate(v):

    runs = 30
    numOCW = 7
    numOBW = 280
    numGrids = 8
    seq_length = 31
    startLimit = 500
    useGrid = False
    CR = 1
    family = get_family()

    nodes = int(v)

    simulation = SimulationCR(nodes=nodes, family=family, useGrid=useGrid, numOCW=numOCW, numOBW=numOBW,
                              numGrids=numGrids, startLimit=startLimit, seq_length=seq_length, CR=CR)
    
    return simulation.get_packet_collision_rate(runs)


if __name__ == "__main__":

    netSizes = [1e3, 2e3, 3e3, 5e3, 1e4, 2e4, 3e4, 5e4, 1e5, 2e5, 3e5, 5e5, 1e6]

    pool = Pool(processes = len(netSizes))
    result = pool.map(get_avg_packet_collision_rate, netSizes)
    pool.close()
    pool.join()

    for i in range(len(netSizes)):
        print(f"nodes = {netSizes[i]}, collided packet rate = {result[i]}")

    print('\n', result)
