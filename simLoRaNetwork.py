import csv
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import Pool
from src.models.LoRaNetwork import LoRaNetwork

def get_decoded_m():

    simTime = 500
    numOCW = 1
    numOBW = 280
    numGrids = 8
    timeGranularity = 6
    freqGranularity = 7
    numDecoders = 1000
    CR = 1
    use_earlydecode = True
    use_earlydrop = True
    use_headerdrop = False
    familyname = "driver"
    numNodes = 250

    random.seed(0)

    network = LoRaNetwork(numNodes, familyname, numOCW, numOBW, numGrids, CR, timeGranularity,
                          freqGranularity, simTime, numDecoders, use_earlydecode, use_earlydrop,
                          use_headerdrop)

    transmissions = network.get_transmissions()
    collision_matrix = network.get_collision_matrix(transmissions)
    network.gateway.run(transmissions, collision_matrix)

    decoded_m = network.get_decoded_matrix(binary=False)

    #collision_matrix[collision_matrix > 1] = 1

    diff = np.subtract(collision_matrix, decoded_m)

    fslots, tslots = decoded_m[0].shape
    tmax =  round(tslots* 102.4/timeGranularity / 1000)
    fmax = round(fslots * 488.28125/freqGranularity / 1000)

    network.milp_solve()
    return

    fig = plt.figure(figsize=(18,12))
    im = plt.imshow(diff[0], extent =[0, tmax, 0, fmax], interpolation ='none', aspect='auto')
    fig.colorbar(im)
    plt.title('diff m using 1 OCW channel')
    plt.xlabel('s')
    plt.ylabel('kHz')
    plt.show()
    #plt.savefig('transmissions-driver.png')
    plt.close('all')


def get_simdata(v):

    runs = 1
    simTime = 500
    numOCW = 1
    numOBW = 280
    numGrids = 8
    timeGranularity = 6
    freqGranularity = 7
    numDecoders = 1000
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
    for r in range(runs):
        random.seed(2*r)

        network.run()
        avg_tracked_txs += network.get_tracked_txs()
        avg_header_drop_packets += network.get_header_drop_packets()
        avg_decoded_bytes += network.get_decoded_bytes()
        avg_decoded_hrd_pld += network.get_decoded_hrd_pld()
        avg_decoded_hdr += network.get_decoded_hdr()
        avg_decodable_pld += network.get_decodable_pld()
        avg_collided_hdr_pld += network.get_collided_hdr_pld()
        network.milp_solve()
        network.restart()

    x = [avg_tracked_txs / runs, avg_header_drop_packets / runs, avg_decoded_bytes / runs,
         avg_decoded_hrd_pld / runs, avg_decoded_hdr / runs, avg_decodable_pld / runs,
         avg_collided_hdr_pld / runs]

    print(f"{numNodes}", x)
    return x


if __name__ == "__main__":


    print('driver\tCR = 1\tprocessors = 1000\tearly d/d = YES\thdr drop = NO')

    netSizes = np.logspace(1.0, 3.0, num=20) # np.logspace(1.0, 4.0, num=50)
    #netSizes = [200]#, 1000, 2000, 5000, 10000]

    pool = Pool(processes = 15)
    result = pool.map(get_simdata, netSizes)
    pool.close()
    pool.join()

    basestr = 'nodrop-cr1-1000p-'

    #with open('path/to/csv_file', 'w') as f:
    #    writer = csv.writer(f)
    #    writer.writerow(row)

    print(basestr+'tracked_txs,', [round(i[0],6) for i in result])
    print(basestr+'header_drop_packets,', [round(i[1],6) for i in result])
    print(basestr+'decoded_bytes,', [round(i[2],6) for i in result])
    print(basestr+'decoded_hrd_pld,', [round(i[3],6) for i in result])
    print(basestr+'decoded_hdr,', [round(i[4],6) for i in result])
    print(basestr+'decodable_pld,', [round(i[5],6) for i in result])
    print(basestr+'collided_hdr_pld,', [round(i[6],6) for i in result])
    