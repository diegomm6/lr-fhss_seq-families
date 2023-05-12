from multiprocessing import Pool
import numpy as np
from base import *
from LempelGreenbergMethod import LempelGreenbergFamily
from LR_FHSS_DriverMethod import LR_FHSS_DriverFamily
from LiFanMethod import LiFanFamily
from HashMethod import HashFamily
from simulation import Simulation

p = 2
k = 5
n = 5
lempelGreenbergGenerator = LempelGreenbergFamily(p=p, n=n, k=k)
lempelGreenberg_fam = lempelGreenbergGenerator.get_optimal_family()
lempelGreenberg_fam = lempelGreenberg_fam *8

l = 277
d = 8
q = 31
maxfreq = 280
mingap = 8
liFanGenerator = LiFanFamily(q=q, maxfreq=maxfreq, mingap=mingap)
liFan_fam = liFanGenerator.get_3l_family(l,d)

m = 512
hashGenerator = HashFamily(q=q)
hash_fam = hashGenerator.get_family(m)

lr_fhssGenerator = LR_FHSS_DriverFamily(q)
lr_fhss_family = lr_fhssGenerator.get_lr_fhss_family()
lr_fhss_family = lr_fhss_family *8


# get average collision rate
# parameter vector v contains the family and useGrid signal
def get_avg_collisions(v):

    nodes = 1000
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


def main():

    pool = Pool(processes=3)  # set the processes max number 4
    result = pool.map(get_avg_collisions,
                      [(hash_fam, True),
                       (lempelGreenberg_fam, True),
                       (liFan_fam, False),
                       (lr_fhss_family, True)])
    pool.close()
    pool.join()

    hash_avg, lempelGreenberg_avg, liFan_avg, lr_fhss_avg= result

    print(f"           hash collisions = {hash_avg}")
    print(f"lempelGreenberg collisions = {lempelGreenberg_avg}")
    print(f"          liFan collisions = {liFan_avg}")
    print(f"        lr_fhss collisions = {lr_fhss_avg}")


if __name__ == "__main__":
    main()
