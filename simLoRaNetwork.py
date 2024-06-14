import csv
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import Pool
from src.models.LoRaNetwork import LoRaNetwork
from src.base.base import *
import time

def plot_rcvM():

    numNodes = 1000

    simTime = 1000
    numOCW = 1
    numOBW = 280
    numGrids = 8
    timeGranularity = 6
    freqGranularity = 25
    numDecoders = 500
    CR = 1
    use_earlydecode = True
    use_earlydrop = True
    use_headerdrop = False
    familyname = "driver" # driver - lifan
    collision_method = "strict" # strict - SINR

    random.seed(0)
    network = LoRaNetwork(numNodes, familyname, numOCW, numOBW, numGrids, CR, timeGranularity,
                          freqGranularity, simTime, numDecoders, use_earlydecode, use_earlydrop,
                          use_headerdrop, collision_method)

    transmissions = network.TXset
    count_static_rcvM = network.get_rcvM(transmissions, power=False, dynamic=False)
    count_dynamic_rcvM = network.get_rcvM(transmissions, power=False, dynamic=True)
    power_dynamic_rcvM = network.get_rcvM(transmissions, power=True, dynamic=True)

    network.gateway.predecode(transmissions, count_dynamic_rcvM, dynamic=True)
    decoded_headers = network.gateway.get_decoded_headers()
    decoded_m = network.get_rcvM(decoded_headers, power=False, dynamic=True)

    network.gateway.run(transmissions, count_dynamic_rcvM, dynamic=True)


    def print_m(m, save='a.png'):
        fslots, tslots = count_static_rcvM[0].shape
        tmax =  round(tslots * (FRG_TIME/timeGranularity))
        fmax = round(fslots * (OBW_BW/freqGranularity) / 1000)
        fig = plt.figure(figsize=(18,12))
        im = plt.imshow(m, extent =[0, tmax, 0, fmax], interpolation ='none', aspect='auto') # mW2dBm(power_dynamic_rcvM[0] /488)
        fig.colorbar(im)
        plt.title(f'Spectogram of received signals [dB/Hz], {numNodes} txs, 1 OCW channel')
        plt.xlabel('s')
        plt.ylabel('kHz', fontsize=18)
        plt.show()
        #plt.savefig(save)
        plt.close('all')

    def print_m2(m, save='spectogram.pdf'):
        fslots, tslots = count_static_rcvM[0].shape
        tmax =  round(tslots * (FRG_TIME/timeGranularity))
        fmax = round(fslots * (OBW_BW/freqGranularity) / 1000)
        fig = plt.figure(figsize=(14,10))
        im = plt.imshow(m, extent =[0, tmax, 0, fmax], interpolation ='none', aspect='auto')
        cbar = fig.colorbar(im)
        cbar.ax.tick_params(labelsize=18)
        plt.title(f'Spectogram of received signals [dB/Hz]', fontsize=16)
        plt.xlabel('s', fontsize=18)
        plt.ylabel('kHz', fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.clim(-135,-115)
        plt.show()
        #plt.savefig(save, bbox_inches='tight')
        plt.close('all')

    # binary - noise (0), signal/interference (1)
    binary_matrix = count_dynamic_rcvM.copy()[0]
    binary_matrix[binary_matrix > 1] = 1

    # 3-value - noise (0), signal (1), interference (2)
    value3_matrix = count_dynamic_rcvM.copy()[0]
    value3_matrix[value3_matrix > 2] = 2

    # interference, noise/signal (0), interference (2)
    interference = count_dynamic_rcvM.copy()[0]
    interference[interference > 2] = 2
    interference[interference < 2] = 0

    # 3-value decoded headers signals
    decoded_m[decoded_m > 2] = 2

    # corner
    cornerm = cornerdetect(count_static_rcvM[0])
    cornerhighlight = np.add(count_static_rcvM[0], cornerm)

    # detected/received difference matrix
    diff = np.subtract(value3_matrix, decoded_m[0])
    diff = np.add(diff, interference)
    diff[diff > 2] = 2

    # received power
    spec_density = mW2dBm(power_dynamic_rcvM[0] /488)

    #print(network.get_OCWchannel_occupancy())

    print_m(count_dynamic_rcvM[0], 'counts.png')
    #print_m(binary_matrix, 'bin.png')
    #print_m(value3_matrix, '3value.png')
    #print_m(decoded_m[0], 'dcdd.png')
    #print_m(diff, 'diff.png')
    #print_m2(spec_density)


def get_simdata(v):

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
    familyname = "driver" # driver - lifan

    power = False
    dynamic = False # NO SUPPORT FOR STATIC DOPPLER IN exhaustive search
    collision_method = "strict" # strict - SINR

    numNodes = int(v)

    network = LoRaNetwork(numNodes, familyname, numOCW, numOBW, numGrids, CR, timeGranularity,
                          freqGranularity, simTime, numDecoders, use_earlydecode, use_earlydrop,
                          use_headerdrop, collision_method)

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
    avg_lenmatch = 0 
    avg_minlenerr = 0
    ch_occ = 0
    for r in range(runs):
        random.seed(2*r)
    
        collided_TXset, diffM = network.get_predecoded_data()
        
        network.run(power, dynamic)
        avg_tracked_txs += network.get_tracked_txs()
        avg_header_drop_packets += network.get_header_drop_packets()
        avg_decoded_bytes += network.get_decoded_bytes()
        avg_decoded_hrd_pld += network.get_decoded_hrd_pld()
        avg_decoded_hdr += network.get_decoded_hdr()
        avg_decodable_pld += network.get_decodable_pld()
        avg_collided_hdr_pld += network.get_collided_hdr_pld()
        ch_occ += network.get_OCWchannel_occupancy()

        tp, fp, fn, _time, lenmatch, minlenerr = 0,0,0,0,0,0
        #if len(collided_TXset):
        #    tp, fp, fn, _time, lenmatch, minlenerr  = network.exhaustive_search(collided_TXset, diffM) 

        avg_tp += tp
        avg_fp += fp
        avg_fn += fn
        avg_time += _time
        avg_lenmatch += lenmatch
        avg_minlenerr += minlenerr
        
        network.restart()


    x = [avg_tracked_txs / runs, avg_header_drop_packets / runs, avg_decoded_bytes / runs,
         avg_decoded_hrd_pld / runs, avg_decoded_hdr / runs, avg_decodable_pld / runs,
         avg_collided_hdr_pld / runs, avg_tp / runs, avg_fp / runs, avg_fn / runs, avg_time / runs, ch_occ / runs,
         avg_lenmatch / runs, avg_minlenerr / runs]
    
    print(f"{numNodes}", x)
    return x


def runsim():

    print('driver \tCR = 1\tprocessors = 800\tearly d/d = YES\thdr drop = NO')

    netSizes = np.logspace(1.0, 3.0, num=40) # np.logspace(1.0, 3.0, num=40)
    #netSizes = [100000]

    # parallel simulation available when NOT USING parallel FHSlocator
    #pool = Pool(processes = 10)
    #result = pool.map(get_simdata, netSizes)
    #pool.close()
    #pool.join()

    result = [get_simdata(nodes) for nodes in netSizes]
    
    basestr = 'driver-'
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
    print(basestr+'chocc,', [round(i[11],6) for i in result])
    print(basestr+'lenmatch,', [round(i[12],6) for i in result])
    print(basestr+'minlenerr,', [round(i[13],6) for i in result])



if __name__ == "__main__":

    plot_rcvM()
    #runsim()
