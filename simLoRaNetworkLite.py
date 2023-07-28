from multiprocessing import Pool
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.models.LoRaNetworkLite import LoRaNetworkLite



def get_simdata(v):

    runs = 10
    simTime = 912
    numOCW = 7
    numOBW = 280
    numGrids = 8
    granularity = 6
    numFragments = 31
    familyname = "driver"
    CR = 2

    nodes = int(v)

    simulation = LoRaNetworkLite(simTime, familyname, numGrids, numOCW,
                                 numOBW, nodes, numFragments, CR, granularity)
    
    #decoded = simulation.run(runs)

    occupancy = simulation.channel_occupancy(runs)

    return np.mean(occupancy)


if __name__ == "__main__":

    netSizes = np.logspace(1.0, 4.0, num=50)

    print('driver\tCR = 2')

    pool = Pool(processes = len(netSizes))
    result = pool.map(get_simdata, netSizes)
    pool.close()
    pool.join()

    #print('avg_decoded_payloads\n', [round(i[0],6) for i in result])
    #print('avg_decoded_packets\n', [round(i[1],6) for i in result])

    print('avg_channel_occupancy\n', [round(i,6) for i in result])


    """
    transmissions = simulation.get_collision_matrix()
    plt.figure(figsize=(18,12))
    sns.heatmap(transmissions[0])
    plt.title('transmissions using 1 OCW channel')
    plt.xlabel(f'slots (w granularity={granularity})')
    plt.ylabel('OBW')
    plt.show()
    #plt.savefig('transmissions-driver.png')
    plt.close('all')
    """
    

