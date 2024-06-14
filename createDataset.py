from src.base.base import *
from src.base.DatasetGenerator import DatasetGenerator

def create_dataset():

    CR = 1
    numOBW = 280
    timeGranularity = 6
    freqGranularity = 1
    dynamic = False

    numTX_list = [1] * 3
    numFragments_list = [31] * 3
    runs_list = [384] * 3

    multiTX = [2, 5, 10, 20, 50, 100, 200, 300]

    numFragments_list += ([0] * len(multiTX)) + ([31] * len(multiTX))
    runs_list += [100] * (len(multiTX) * 2)
    numTX_list += multiTX * 2

    datasetGenerator = DatasetGenerator(CR, numOBW, freqGranularity, timeGranularity)

    datasetGenerator.create_boundingbox_dataset(dynamic, "dataset1", runs_list, numTX_list, numFragments_list)
    #datasetGenerator.create_boundingbox_dataset(dynamic, "dataset1", [3]*4, [1, 1, 3, 20], [0]*4)


if __name__ == "__main__":
    random.seed(123)
    create_dataset()
