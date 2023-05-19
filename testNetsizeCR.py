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
    d = 8
    q = 31
    maxfreq = 280
    mingap = 8
    liFanGenerator = LiFanFamily(q=q, maxfreq=maxfreq, mingap=mingap)
    liFan_fam1 = liFanGenerator.get_3l_family(277, d)
    liFan_fam2 = liFanGenerator.get_3l_family(281, d)
    liFan_fam3 = liFanGenerator.get_3l_family(283, d)
    liFan_fam4 = liFanGenerator.get_3l_family(287, d)

    liFan_fam5 = np.concatenate((liFan_fam1, liFan_fam2, liFan_fam3, liFan_fam4))

    return liFan_fam5


# test a single method for several number of nodes
def get_avg_packet_collision_rate(v):

    runs = 30
    numOCW = 7
    numOBW = 280
    numGrids = 8
    seq_length = 31
    startLimit = 500
    CR = 2
    useGrid = False
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

    print('CR = 2; liFan_fam5')

    pool = Pool(processes = len(netSizes))
    result = pool.map(get_avg_packet_collision_rate, netSizes)
    pool.close()
    pool.join()

    #for i in range(len(netSizes)):
        #print(f"nodes = {netSizes[i]}, collided packet rate = {result[i]}")

    result2 = [round(i,6) for i in result]
    print('\n', result2)
