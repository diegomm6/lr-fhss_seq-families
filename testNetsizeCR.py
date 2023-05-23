from multiprocessing import Pool
import numpy as np
import galois
from src.base.base import *
from src.families.LempelGreenbergMethod import LempelGreenbergFamily
from src.families.HashMethod import HashFamily
from src.families.LR_FHSS_DriverMethod import LR_FHSS_DriverFamily
from src.families.LiFanMethod import LiFanFamily
from src.families.WangMethod import WangFamily
from src.simulationCR import SimulationCR


# this script is designed to test multiple values for the network size for a 
# sinlge family, using parallel computation

def get_family():
    q = 31
    l = 281
    d = 8
    liFanGenerator = LiFanFamily(q=q, maxfreq=280, mingap=8)
    liFan_fam1 = liFanGenerator.get_family(l, d, '3l')

    return liFan_fam1


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
    

    lifan_netSizes = [1e2, 2e2, 3e2, 4e2, 5e2, 6e2, 7e2, 8e2, 9e2, 1e3,
                         1.2e3, 1.4e3, 1.6e3, 1.8e3, 2e3, 2.25e3, 2.5e3,
                         2.75e3, 3e3, 3.33e3, 3.66e3, 4e3, 4.33e3, 4.66e3,
                         5e3, 5.5e3, 6e3, 6.5e3, 7e3, 7.5e3, 8e3, 9e3, 1e4]
    
    netSizes = netSizesCR2

    print('CR = 2; liFan_fam')

    pool = Pool(processes = len(netSizes))
    result = pool.map(get_avg_packet_collision_rate, netSizes)
    pool.close()
    pool.join()

    result2 = [round(i,6) for i in result]
    print('\n', result2)
