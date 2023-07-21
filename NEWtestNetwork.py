from multiprocessing import Pool
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.base.LoRaNetwork import LoRaNetwork


def get_simdata(v):

    runs = 10
    simTime = 1000
    numOCW = 7
    numOBW = 280
    numGrids = 8
    granularity = 6
    numDecoders = 100
    CR = 1
    use_earlydecode = False
    use_earlydrop = False
    familyname = "lifan"

    numNodes = int(v)

    network = LoRaNetwork(numNodes, familyname, numOCW, numOBW, numGrids, CR, granularity,
                          simTime, numDecoders, use_earlydecode, use_earlydrop)

    avg_decoded_packets = 0
    avg_decoded_payloads = 0
    avg_decoded_bytes = 0
    avg_collided_payloads = 0
    for _ in range(runs):

        network.run()
        avg_decoded_packets += network.get_decoded_packets()
        avg_decoded_payloads += network.get_decoded_payloads()
        avg_decoded_bytes += network.get_decoded_bytes()
        avg_collided_payloads += network.get_collided_payloads()
        network.restart()

    x = [avg_decoded_packets / runs, avg_decoded_payloads / runs, avg_decoded_bytes / runs, avg_collided_payloads/ runs]

    print(f"{numNodes}", x)
    return x


if __name__ == "__main__":

    print('lifan\tCR = 1\tprocessors = 100\tearly d/d = NO')

    netSizes = np.logspace(1.0, 4.0, num=30)

    pool = Pool(processes = 10)
    result = pool.map(get_simdata, netSizes)
    pool.close()
    pool.join()

    print('avg_decoded_packets\n', [round(i[0],6) for i in result])
    print('avg_decoded_payloads\n', [round(i[1],6) for i in result])
    print('avg_decoded_bytes\n', [round(i[2],6) for i in result])
    print('avg_collided_payloads\n', [round(i[3],6) for i in result])
    
