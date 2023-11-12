from multiprocessing import Pool
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.models.LoRaNetwork import LoRaNetwork


def get_m():
    
    simTime = 2000
    numOCW = 1
    numOBW = 280
    numGrids = 8
    timeGranularity = 6
    freqGranularity = 7
    numDecoders = 100
    CR = 1
    use_earlydecode = True
    use_earlydrop = True
    use_headerdrop = True
    familyname = "driver"
    numNodes = 500

    network = LoRaNetwork(numNodes, familyname, numOCW, numOBW, numGrids, CR, timeGranularity,
                          freqGranularity, simTime, numDecoders, use_earlydecode, use_earlydrop,
                          use_headerdrop)

    m = network.get_m()

    fslots, tslots = m[0].shape
    tmax =  round(tslots* 102.4/timeGranularity / 1000)
    fmax = round(fslots * 488.28125/freqGranularity / 1000)

    fig = plt.figure(figsize=(18,12))
    im = plt.imshow(m[0], extent =[0, tmax, 0, fmax], interpolation ='none', aspect='auto')
    fig.colorbar(im)
    plt.title('transmissions using 1 OCW channel')
    plt.xlabel('s')
    plt.ylabel('kHz')
    plt.show()
    #plt.savefig('transmissions-driver.png')
    plt.close('all')


def get_simdata(v):

    runs = 1
    simTime = 1000
    numOCW = 1
    numOBW = 280
    numGrids = 8
    timeGranularity = 6
    freqGranularity = 7
    numDecoders = 100
    CR = 2
    use_earlydecode = True
    use_earlydrop = True
    use_headerdrop = True
    familyname = "driver"

    numNodes = int(v)

    network = LoRaNetwork(numNodes, familyname, numOCW, numOBW, numGrids, CR, timeGranularity,
                          freqGranularity, simTime, numDecoders, use_earlydecode, use_earlydrop,
                          use_headerdrop)

    avg_decoded_packets = 0
    avg_decoded_payloads = 0
    avg_decoded_bytes = 0
    avg_collided_payloads = 0
    avg_header_drop_packets = 0
    for _ in range(runs):

        network.run2()
        avg_decoded_packets += network.get_decoded_packets()
        avg_decoded_payloads += network.get_decoded_payloads()
        avg_decoded_bytes += network.get_decoded_bytes()
        avg_collided_payloads += network.get_collided_payloads()
        avg_header_drop_packets += network.get_header_drop_packets()
        network.restart()

    x = [avg_decoded_packets / runs, avg_decoded_payloads / runs, avg_decoded_bytes / runs,
         avg_collided_payloads/ runs, avg_header_drop_packets / runs]

    print(f"{numNodes}", x)
    return x


if __name__ == "__main__":

    get_m()
    
    """
    print('driver\tCR = 2\tprocessors = 100\tearly d/d = YES\thdr drop = YES')

    netSizes = np.logspace(1.0, 4.0, num=50)

    #netSizes = [500, 1000, 2000, 5000, 10000]

    pool = Pool(processes = 5)
    result = pool.map(get_simdata, netSizes)
    pool.close()
    pool.join()

    print('avg_decoded_packets\n', [round(i[0],6) for i in result])
    print('avg_decoded_payloads\n', [round(i[1],6) for i in result])
    print('avg_decoded_bytes\n', [round(i[2],6) for i in result])
    print('avg_collided_payloads\n', [round(i[3],6) for i in result])
    print('avg_header_drop_packets\n', [round(i[4],6) for i in result])
    """
    