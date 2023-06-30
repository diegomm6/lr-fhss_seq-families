from multiprocessing import Pool
import numpy as np
import seaborn as sns
import galois
import matplotlib.pyplot as plt
from src.families.LiFanMethod import LiFanFamily
from src.families.LempelGreenbergMethod import LempelGreenbergFamily
from src.families.LR_FHSS_DriverMethod import LR_FHSS_DriverFamily
from src.models.simulationCRgranularity import SimulationCRgranularity


def get_family():

    polys = galois.primitive_polys(2, 5)
    poly1 = next(polys)
    lempelGreenbergGenerator = LempelGreenbergFamily(p=2, n=5, k=5, poly=poly1)
    lempelGreenberg_fam = lempelGreenbergGenerator.get_family()*8

    liFanGenerator = LiFanFamily(q=31, maxfreq=280, mingap=8)
    lifan_family = liFanGenerator.get_family(281, 8, '2l')

    lr_fhssGenerator = LR_FHSS_DriverFamily(q=31)
    driver_family = lr_fhssGenerator.get_family()*8

    return lifan_family


# test a single method for several number of nodes
def get_avg_packet_collision_rate(v):

    runs = 30
    numOCW = 7
    numOBW = 280
    numGrids = 8
    seq_length = 31
    startLimit = 93
    granularity = 8
    CR = 2

    useGrid = False
    family = get_family()

    nodes = int(v)

    simulation = SimulationCRgranularity(nodes=nodes, family=family, useGrid=useGrid, numOCW=numOCW, numOBW=numOBW,
                              numGrids=numGrids, startLimit=startLimit, seq_length=seq_length, CR=CR, granularity=granularity)
    
    colrate = simulation.get_packet_collision_rate(runs)

    #print(f"finished for n={nodes}")

    return colrate


if __name__ == "__main__":

    netSizesCR1 = [1e3, 2e3, 5e3, 7.5e3, 1e4, 1.25e4, 1.5e4, 1.75e4, 2e4, 2.25e4, 2.5e4, 2.75e4,
                   3e4, 3.33e4, 3.66e4, 4e4, 4.5e4, 5e4, 5.5e4, 6e4, 7e4, 8e4, 9e4, 1e5]

    netSizesCR2 = [1e3, 2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 8e3, 9e3, 1e4, 1.2e4, 1.4e4, 1.6e4,
                   1.8e4, 2e4, 2.33e4, 2.66e4, 3e4, 3.5e4, 4e4, 4.5e4, 5e4, 6e4, 1e5]
    

    netSizes = np.logspace(1.0, 4.0, num=50)

    print('CR = 1; lifan_family')

    pool = Pool(processes = len(netSizes))
    result = pool.map(get_avg_packet_collision_rate, netSizes)
    pool.close()
    pool.join()

    result2 = [round(i,6) for i in result]
    print('\n', result2)


    """
    transmissions = simulation.run()
    plt.figure(figsize=(18,12))
    sns.heatmap(transmissions[0])
    plt.title('transmissions using 1 OCW channel')
    plt.xlabel(f'slots (w granularity={granularity})')
    plt.ylabel('OBW')
    plt.show()
    #plt.savefig('transmissions-driver.png')
    plt.close('all')
    """
    

