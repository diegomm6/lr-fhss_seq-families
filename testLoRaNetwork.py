from multiprocessing import Pool
import numpy as np
import seaborn as sns
import galois
import matplotlib.pyplot as plt
from src.families.LiFanMethod import LiFanFamily
from src.families.LempelGreenbergMethod import LempelGreenbergFamily
from src.families.LR_FHSS_DriverMethod import LR_FHSS_DriverFamily
from src.base.LoRaNetwork import LoRaNetwork


def get_family():

    polys = galois.primitive_polys(2, 5)
    poly1 = next(polys)
    lempelGreenbergGenerator = LempelGreenbergFamily(p=2, n=5, k=5, poly=poly1)
    lempelGreenberg_fam = lempelGreenbergGenerator.get_family()*8

    liFanGenerator = LiFanFamily(q=31, maxfreq=280, mingap=8)
    lifan_family = liFanGenerator.get_family(281, 8, '2l')

    lr_fhssGenerator = LR_FHSS_DriverFamily(q=31)
    driver_family = lr_fhssGenerator.get_family()*8

    return driver_family


def get_avg_decoding_rate(v):
    runs = 30
    simTime = 531
    numOCW = 7
    numOBW = 280
    numGrids = 8
    max_seq_length = 31
    granularity = 4
    numDecoders = 1
    decodeCapacity = 16
    CR = 2
    useGrid = True
    family = get_family()

    numNodes = int(v)

    network = LoRaNetwork(numNodes, family, useGrid, numOCW, numOBW, numGrids,
                          CR, granularity, max_seq_length, simTime, numDecoders, decodeCapacity)

    avg_decode_rate = 0
    for r in range(runs):
        network.run()
        avg_decode_rate += network.get_decoded_transmissions()
        network.restart()

    return avg_decode_rate / numNodes / runs


if __name__ == "__main__":

    netSizesCR1 = np.logspace(1.0, 4.0, num=50)    

    netSizes = netSizesCR1

    print('CR = 2; driver_family')

    pool = Pool(processes = len(netSizes))
    result = pool.map(get_avg_decoding_rate, netSizes)
    pool.close()
    pool.join()

    result2 = [round(i,6) for i in result]
    print('\n', result2)
