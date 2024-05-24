import csv
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import Pool
from src.models.LoRaNetwork import LoRaNetwork
from src.base.base import *
import time
import os
from PIL import Image


def createdataset():

    runs = 100
    simTime = 500
    numOCW = 1
    numOBW = 280
    numGrids = 8
    timeGranularity = 6
    freqGranularity = 1
    numDecoders = 800
    CR = 1
    use_earlydecode = True
    use_earlydrop = True
    use_headerdrop = False
    familyname = "driver" # driver - lifan

    power = False
    dynamic = True # NO SUPPORT FOR STATIC DOPPLER IN exhaustive search
    collision_method = "strict" # strict - SINR


    for r in range(runs):
        random.seed(2*r)

        numNodes = (r+1) * 10
        network = LoRaNetwork(numNodes, familyname, numOCW, numOBW, numGrids, CR, timeGranularity,
                        freqGranularity, simTime, numDecoders, use_earlydecode, use_earlydrop,
                        use_headerdrop, collision_method)
    
        transmissions = network.TXset
        count_dynamic_rcvM = network.get_rcvM(transmissions, power=False, dynamic=True)

        binary_matrix = count_dynamic_rcvM.copy()[0]
        binary_matrix[binary_matrix > 1] = 255

        image = Image.fromarray(binary_matrix)
        image = image.convert('RGB')

        image.save(os.path.join(os.getcwd(), f"imgdataset\img{r:03d}.png"))

        del network

if __name__ == "__main__":
    createdataset()
    