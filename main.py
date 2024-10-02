import csv
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import Pool
from src.models.LoRaNetwork import LoRaNetwork
from src.base.base import *

def get_simdata(v):

    runs = 10                   # number of repetitions
    simTime = 500               # simulation time in timeslots
    numOCW = 1                  # number of OCW channels
    numOBW = 280                # number of OBW channels
    numGrids = 8                # number of grids
    timeGranularity = 6         # number of timeslots per LRFHSS fragment 
    freqGranularity = 25        # number of frequency slots per OBW channel
    numDecoders = 800           # number of decoders available at gateway
    CR = 1                      # coding rate, CR=1 for 1/3 and CR=2 for 2/3
    use_earlydecode = True      # early decode mechanisms flag
    use_earlydrop = True        # early drop mechanisms flag
    use_headerdrop = False      # drop frame after unsucceful header reception flag
    familyname = "driver"       # FHS family selection, 2 options: "driver" / "lifan"

    power = False               # select power based model
    dynamic = False             # dynamic doppler model (NO SUPPORT FOR STATIC DOPPLER IN exhaustive search)
    collision_method = "strict" # collision determination model, 2 options "strict" / "SINR"

    numNodes = int(v)           # number of LRFHSS nodes on ground

    network = LoRaNetwork(numNodes, familyname, numOCW, numOBW, numGrids, CR, timeGranularity,
                          freqGranularity, simTime, numDecoders, use_earlydecode, use_earlydrop,
                          use_headerdrop, collision_method)

    avg_tracked_txs = 0         # tracked LRFHSS frames by gateway's decoders
    avg_decoded_bytes = 0       # decoded bytes
    avg_header_drop_packets = 0 # packets dropped due to missing header
    avg_decoded_hrd_pld = 0     # decoded LRFHSS frames
    avg_decoded_hdr = 0         # decoded headers with unsuccesful header part
    avg_decodable_pld = 0       # decoded payloads with failes header reception
    avg_collided_hdr_pld = 0    # failed frames with both header and payload collided
    avg_tp = 0                  # FHS Locator true positives
    avg_fp = 0                  # FHS Locator false positives
    avg_fn = 0                  # FHS Locator false neagtives
    avg_time = 0                # avg FHS Locator processing time
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

        tp, fp, fn, _time = 0,0,0,0
        if len(collided_TXset):
            tp, fp, fn, _time, _, _  = network.exhaustive_search(collided_TXset, diffM) 

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

    # define vector of network sizes
    netSizes = np.logspace(1.0, 3.0, num=40) 

    result = [get_simdata(nodes) for nodes in netSizes]
    
    basestr = ''
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
    runsim()
