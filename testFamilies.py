from multiprocessing import Pool
import numpy as np
import galois
from src.base import *
from src.families.LempelGreenbergMethod import LempelGreenbergFamily
from src.families.LR_FHSS_DriverMethod import LR_FHSS_DriverFamily
from src.families.LiFanMethod import LiFanFamily
from src.families.HashMethod import HashFamily
from src.families.WangMethod import WangFamily
from src.simulation import Simulation

# this script is designed to test multiple families a single network size
# using parallel computation


# test several methods for a single number of nodes
# get average collided slots
# parameter vector v contains the family and useGrid signal
def get_avg_collisions(v):

    nodes = 100
    runs = 30
    numOCW = 7
    numOBW = 280
    numGrids = 8
    seq_length = 31
    startLimit = 500

    family, useGrid = v

    simulation = Simulation(nodes=nodes, family=family, useGrid=useGrid, numOCW=numOCW,
                            numOBW=numOBW, numGrids=numGrids, startLimit=startLimit, seq_length=seq_length)

    avg = 0
    for r in range(runs):
        txData = simulation.run()
        avg += (txData > 1).sum()

    return avg / runs


if __name__ == "__main__":

    #########################################################################################
    #########################      LEMPEL GREENBERG METHOD      #############################
    #########################################################################################
    p = 2
    k = 5
    n = 5
    polys = galois.primitive_polys(p, n)
    poly1 = next(polys)
    poly2 = next(polys)

    lempelGreenbergGenerator = LempelGreenbergFamily(p=p, n=n, k=k, poly=poly1)
    lempelGreenberg_fam = lempelGreenbergGenerator.get_family()
    lempelGreenberg_fam = lempelGreenberg_fam *8

    lempelGreenbergGenerator2 = LempelGreenbergFamily(p=p, n=n, k=k, poly=poly2)
    lempelGreenberg_fam2 = lempelGreenbergGenerator2.get_family()
    lempelGreenberg_fam2 = lempelGreenberg_fam2 *8

    lempelGreenberg_fam3 = np.concatenate((lempelGreenberg_fam, lempelGreenberg_fam2))
    #print(lempelGreenberg_fam3.shape)
    #for i,s in enumerate(lempelGreenberg_fam3):
    #    print(f"seq{i}: {s}")


    #########################################################################################
    ##############################      LI FAN METHOD       #################################
    #########################################################################################
    d = 8
    q = 31
    method = '3l'
    liFanGenerator = LiFanFamily(q=q, maxfreq=280, mingap=8)
    liFan_fam1 = liFanGenerator.get_family(277, d, method)
    liFan_fam2 = liFanGenerator.get_family(281, d, method)
    liFan_fam3 = liFanGenerator.get_family(283, d, method)
    liFan_fam4 = liFanGenerator.get_family(287, d, method)

    liFan_fam5 = np.concatenate((liFan_fam1, liFan_fam2, liFan_fam3, liFan_fam4))


    #########################################################################################
    ##############################        HASH METHOD       #################################
    #########################################################################################
    m = 384
    hashGenerator = HashFamily(q=q)
    hash_fam = hashGenerator.get_family(m)


    #########################################################################################
    ##############################      DRIVER METHOD       #################################
    #########################################################################################
    lr_fhssGenerator = LR_FHSS_DriverFamily(q)
    lr_fhss_family = lr_fhssGenerator.get_family()
    lr_fhss_family = lr_fhss_family *8


    #########################################################################################
    ##############################       WANG METHOD        #################################
    #########################################################################################
    p = 37
    q = 31
    w = 8
    d = 8
    wangGenerator = WangFamily(p=p, q=q, w=w, d=d)
    wangFamily = wangGenerator.get_family()


    #########################################################################################
    ##############################        SIMULATION          ###############################
    #########################################################################################

    pool = Pool(processes = 4)
    result = pool.map(get_avg_collisions,
                      [(hash_fam, True),
                       (lempelGreenberg_fam, True),
                       (liFan_fam1, False),
                       (lr_fhss_family, True)])
    pool.close()
    pool.join()

    hash_avg, lempelGreenberg_avg, liFan_avg, lr_fhss_avg= result

    print(f"           hash collisions = {hash_avg}")
    print(f"lempelGreenberg collisions = {lempelGreenberg_avg}")
    print(f"          liFan collisions = {liFan_avg}")
    print(f"        lr_fhss collisions = {lr_fhss_avg}")

    print('\n', result)
