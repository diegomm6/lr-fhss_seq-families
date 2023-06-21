from multiprocessing import Pool
import numpy as np
import galois
from src.families.LiFanMethod import LiFanFamily
from src.families.LempelGreenbergMethod import LempelGreenbergFamily
from src.families.LR_FHSS_DriverMethod import LR_FHSS_DriverFamily
from src.base.LoRaNetwork import LoRaNetwork


def get_family():

    polys = galois.primitive_polys(2, 5)
    poly1 = next(polys)
    lempelGreenbergGenerator = LempelGreenbergFamily(p=2, n=5, k=5, poly=poly1)
    lempelGreenberg_fam = lempelGreenbergGenerator.get_family()*8


    lr_fhssGenerator = LR_FHSS_DriverFamily(q=31)
    driver_family = lr_fhssGenerator.get_family()*8


    liFanGenerator = LiFanFamily(q=31, maxfreq=280, mingap=8)
    lifan_family = liFanGenerator.get_family(281, 8, '2l')

    liFan_fam1 = liFanGenerator.get_family(277, 8, '2l')
    liFan_fam2 = liFanGenerator.get_family(281, 8, '2l')
    liFan_fam3 = liFanGenerator.get_family(283, 8, '2l')
    liFan_fam4 = liFanGenerator.get_family(287, 8, '2l')
    liFan_fam5 = np.concatenate((liFan_fam1, liFan_fam2, liFan_fam3, liFan_fam4))

    return lifan_family


def get_avg_decoding_rate(v):
    runs = 10
    simTime = 992
    numOCW = 7
    numOBW = 280
    numGrids = 8
    max_seq_length = 31
    granularity = 8
    numDecoders = 1
    decodeCapacity = 4096
    CR = 1
    useGrid = False
    family = get_family()

    numNodes = int(v)

    network = LoRaNetwork(numNodes, family, useGrid, numOCW, numOBW, numGrids,
                          CR, granularity, max_seq_length, simTime, numDecoders, decodeCapacity)

    avg_decoded_bytes = 0
    avg_decoded_packets = 0
    avg_collided_packets = 0
    for r in range(runs):

        network.run()
        avg_decoded_bytes += network.get_decoded_bytes()
        avg_decoded_packets += network.get_decoded_packets()
        avg_collided_packets += network.get_collided_packets()
        network.restart()

    x = [avg_decoded_bytes / runs, avg_decoded_packets / runs, avg_collided_packets/ runs]

    print(f"{numNodes}", x)
    return x


if __name__ == "__main__":

    print('lifan_family\tCR = 1\tprocessors = 4096')

    netSizes = np.logspace(1.0, 4.0, num=50)

    pool = Pool(processes = 15)
    result = pool.map(get_avg_decoding_rate, netSizes)
    pool.close()
    pool.join()

    print('avg_decoded_bytes\n', [round(i[0],6) for i in result])
    print('avg_decoded_packets\n', [round(i[1],6) for i in result])
    print('avg_collided_packets\n', [round(i[2],6) for i in result])

