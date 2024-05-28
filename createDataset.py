import random, os
import numpy as np
from PIL import Image
from src.base.base import *
from src.base.DatasetGenerator import DatasetGenerator

def getLabel(TXSeqiIds):
    label = ['0'] * 384
    for id in TXSeqiIds:
        label[id] = '1'
    return ''.join(label)

def createdataset():

    runs = 100
    CR = 1
    numOCW = 1
    numOBW = 280
    simTime = 280
    timeGranularity = 6
    freqGranularity = 1
    numFragments = 0
    
    numTX = 200
    labels = ""
    for id in range(runs):

        datasetGenerator = DatasetGenerator(CR, numOCW, numOBW, simTime, freqGranularity, timeGranularity)

        if numTX==1:
            transmissions = [datasetGenerator.get_transmission(id, numFragments)]
        else:
            transmissions = datasetGenerator.get_TXlist(numTX, numFragments)

        rcvM = datasetGenerator.get_rcvM(transmissions)
        rcvM[rcvM > 0] = 255
        rcvM = np.array(rcvM[0], dtype=np.uint8)
        image = Image.fromarray(rcvM, mode='L')

        imgname = f"{numTX}txs_{id:03d}"
        image.save(os.path.join(os.getcwd(), f"imgdataset\img_v2_{imgname}.png"))

        TXSeqiIds = [tx.seqid for tx in transmissions]
        labels += f"v2_{imgname},{getLabel(TXSeqiIds)}\n"

        del datasetGenerator

    with open('data.csv', 'a') as file:
        file.write(labels)   


if __name__ == "__main__":
    random.seed(1234)
    createdataset()
