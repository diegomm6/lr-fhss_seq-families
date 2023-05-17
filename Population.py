from Node import Node
import numpy as np

class Population():

    def __init__(self, family, useGrid, numOCW, numGrids, n, startLimit) -> None:
        self.family = family
        self.useGrid = useGrid
        self.numOCW = numOCW
        self.numGrids = numGrids
        self.nodes = self.init_nodes(n, startLimit)


    def init_nodes(self, n, startLimit):

        nodes = []
        for _ in range(n):

            startTime = np.random.randint(startLimit)
            seq_id = np.random.randint(len(self.family))
            seq = self.family[seq_id]

            ocw = np.random.randint(self.numOCW)
            grid = 0
            if self.useGrid:
                grid = np.random.randint(self.numGrids)

            nodes.append(Node(seq, ocw, grid, startTime))
        
        return nodes

