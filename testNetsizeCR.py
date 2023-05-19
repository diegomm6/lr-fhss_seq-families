from multiprocessing import Pool
import numpy as np
from base import *
import galois
from LempelGreenbergMethod import LempelGreenbergFamily
from HashMethod import HashFamily
from LR_FHSS_DriverMethod import LR_FHSS_DriverFamily
from LiFanMethod import LiFanFamily
from WangMethod import WangFamily
from simulationCR import SimulationCR


# this script is designed to test multiple values for the network size for a 
# sinlge family, using parallel computation

def get_family():
    q = 31
    m = 384
    hashGenerator = HashFamily(q=q)
    hash_fam = hashGenerator.get_family(m)

    return hash_fam


# test a single method for several number of nodes
def get_avg_packet_collision_rate(v):

    runs = 30
    numOCW = 7
    numOBW = 280
    numGrids = 8
    seq_length = 31
    startLimit = 500
    CR = 2
    useGrid = True
    family = get_family()

    nodes = int(v)

    simulation = SimulationCR(nodes=nodes, family=family, useGrid=useGrid, numOCW=numOCW, numOBW=numOBW,
                              numGrids=numGrids, startLimit=startLimit, seq_length=seq_length, CR=CR)
    
    return simulation.get_packet_collision_rate(runs)


if __name__ == "__main__":

    netSizesCR1 = [1e3, 2e3, 5e3, 7.5e3, 1e4, 1.33e4, 1.66e4, 2e4, 2.33e4, 2.66e4,
                   3e4, 3.5e4, 4e4, 4.5e4, 5e4, 6e4, 7e4, 8e4, 9e4, 1e5]

    netSizesCR2 = [1e3, 2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 8e3, 9e3, 1e4, 1.2e4,
                   1.4e4, 1.6e4, 1.8e4, 2e4, 2.33e4, 2.66e4, 3e4, 5e4, 1e5]
    
    netSizes = netSizesCR2

    print('CR = 2; hash_fam')

    pool = Pool(processes = len(netSizes))
    result = pool.map(get_avg_packet_collision_rate, netSizes)
    pool.close()
    pool.join()

    #for i in range(len(netSizes)):
        #print(f"nodes = {netSizes[i]}, collided packet rate = {result[i]}")

    print('\n', result)
