from Node import Node
import numpy as np

class Population():

    def __init__(self, family, useGrid, numOCW, numGrids, n, startLimit) -> None:
        self.family = family
        self.useGrid = useGrid
        self.numOCW = numOCW
        self.numGrids = numGrids
        self.n = n
        self.startLimit = startLimit
        self.nodes = self.init_nodes()


    def restart(self):

        node : Node
        for node in self.nodes:

            node.startTime = np.random.randint(self.startLimit)
            seq_id = np.random.randint(len(self.family))
            node.seq = self.family[seq_id]

            node.ocw = np.random.randint(self.numOCW)
            node.grid = 0
            if self.useGrid:
                node.grid = np.random.randint(self.numGrids)
        

    def init_nodes(self):

        nodes = []
        for _ in range(self.n):

            startTime = np.random.randint(self.startLimit)
            seq_id = np.random.randint(len(self.family))
            seq = self.family[seq_id]

            ocw = np.random.randint(self.numOCW)
            grid = 0
            if self.useGrid:
                grid = np.random.randint(self.numGrids)

            nodes.append(Node(seq, ocw, grid, startTime))
        
        return nodes

