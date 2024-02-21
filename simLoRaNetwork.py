import csv
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import Pool
from src.models.LoRaNetwork import LoRaNetwork
from src.base.base import cornerdetect, dBm2mW, mW2dBm, get_FS_pathloss
import time

def get_RXmatrix():

    simTime = 500
    numOCW = 1
    numOBW = 280
    numGrids = 8
    timeGranularity = 6
    freqGranularity = 25
    numDecoders = 100
    CR = 1
    use_earlydecode = True
    use_earlydrop = True
    use_headerdrop = False
    familyname = "driver"
    numNodes = 5

    random.seed(0)

    network = LoRaNetwork(numNodes, familyname, numOCW, numOBW, numGrids, CR, timeGranularity,
                          freqGranularity, simTime, numDecoders, use_earlydecode, use_earlydrop,
                          use_headerdrop)

    transmissions = network.get_transmissions()

    staticdoppler_matrix = network.get_rcvM(transmissions, power=False, dynamic=False)
    dynamicdoppler_matrix = network.get_rcvM(transmissions, power=False, dynamic=True)
    RXpower_matrix_dB = network.get_rcvM(transmissions, power=True, dynamic=True)

    network.gateway.run(transmissions, RXpower_matrix_dB)
    decoded_m = network.get_decoded_matrix(binary=False)

    # binary
    binary_matrix = dynamicdoppler_matrix.copy()[0]
    binary_matrix[binary_matrix > 1] = 1

    # 3-value
    value3_matrix = dynamicdoppler_matrix.copy()[0]
    value3_matrix[value3_matrix > 2] = 2

    # corner
    cornerm = cornerdetect(staticdoppler_matrix[0])
    cornerhighlight = np.add(staticdoppler_matrix[0], cornerm)

    # detected/received difference matrix
    #diff = np.subtract(staticdoppler_matrix, decoded_m)

    fslots, tslots = staticdoppler_matrix[0].shape
    tmax =  round(tslots * (102.4/timeGranularity) / 1000)
    fmax = round(fslots * (488.28125/freqGranularity) / 1000)
    fig = plt.figure(figsize=(18,12))
    im = plt.imshow(mW2dBm(RXpower_matrix_dB[0] /488), extent =[0, tmax, 0, fmax], interpolation ='none', aspect='auto') # mW2dBm(RXpower_matrix_dB[0] /488)
    fig.colorbar(im)
    plt.title(f'Spectogram of received signals [dB/Hz], {numNodes} txs, 1 OCW channel')
    plt.xlabel('s')
    plt.ylabel('kHz')
    plt.show()
    #plt.savefig('transmissions-driver.png')
    plt.close('all')


def get_simdata(v):

    power = True
    dynamic = False

    runs = 10
    simTime = 500
    numOCW = 1
    numOBW = 280
    numGrids = 8
    timeGranularity = 6
    freqGranularity = 25
    numDecoders = 800
    CR = 1
    use_earlydecode = True
    use_earlydrop = True
    use_headerdrop = False
    familyname = "driver"

    numNodes = int(v)

    network = LoRaNetwork(numNodes, familyname, numOCW, numOBW, numGrids, CR, timeGranularity,
                          freqGranularity, simTime, numDecoders, use_earlydecode, use_earlydrop,
                          use_headerdrop)

    avg_tracked_txs = 0
    avg_decoded_bytes = 0
    avg_header_drop_packets = 0
    avg_decoded_hrd_pld = 0
    avg_decoded_hdr = 0
    avg_decodable_pld = 0
    avg_collided_hdr_pld = 0
    avg_tp = 0
    avg_fp = 0
    avg_fn = 0
    avg_time = 0
    for r in range(runs):
        random.seed(2*r)

        network.run(power, dynamic)
        avg_tracked_txs += network.get_tracked_txs()
        avg_header_drop_packets += network.get_header_drop_packets()
        avg_decoded_bytes += network.get_decoded_bytes()
        avg_decoded_hrd_pld += network.get_decoded_hrd_pld()
        avg_decoded_hdr += network.get_decoded_hdr()
        avg_decodable_pld += network.get_decodable_pld()
        avg_collided_hdr_pld += network.get_collided_hdr_pld()
        tp, fp, fn, _time = 0,0,0,0 #network.exhaustive_search(power, dynamic)
        avg_tp += tp
        avg_fp += fp
        avg_fn += fn
        avg_time += _time 
        network.restart()

    x = [avg_tracked_txs / runs, avg_header_drop_packets / runs, avg_decoded_bytes / runs,
         avg_decoded_hrd_pld / runs, avg_decoded_hdr / runs, avg_decodable_pld / runs,
         avg_collided_hdr_pld / runs, avg_tp / runs, avg_fp / runs, avg_fn / runs, avg_time / runs]

    print(f"{numNodes}", x)
    return x


def runsim():

    print('driver \tCR = 1\tprocessors = 800\tearly d/d = YES\thdr drop = NO')

    netSizes = np.logspace(1.0, 4.0, num=15) # np.logspace(1.0, 3.0, num=40)
    #netSizes = [50]

    #pool = Pool(processes = 20)
    #result = pool.map(get_simdata, netSizes)
    #pool.close()
    #pool.join()

    result = []
    for nodes in netSizes:
        result.append(get_simdata(nodes))

    basestr = 'nodrop-cr1-500p-'
    print(basestr+'tracked_txs,', [round(i[0],6) for i in result])
    print(basestr+'header_drop_packets,', [round(i[1],6) for i in result])
    print(basestr+'decoded_bytes,', [round(i[2],6) for i in result])
    print(basestr+'decoded_hrd_pld,', [round(i[3],6) for i in result])
    print(basestr+'decoded_hdr,', [round(i[4],6) for i in result])
    print(basestr+'decodable_pld,', [round(i[5],6) for i in result])
    print(basestr+'collided_hdr_pld,', [round(i[6],6) for i in result])
    print(basestr+'tp,', [round(i[7],6) for i in result])
    print(basestr+'fp,', [round(i[8],6) for i in result])
    print(basestr+'fn,', [round(i[9],6) for i in result])
    print(basestr+'time,', [round(i[10],6) for i in result])
    

if __name__ == "__main__":

    #get_RXmatrix()
    runsim()
