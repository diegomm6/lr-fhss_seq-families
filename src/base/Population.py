from src.base.Node import Node
import numpy as np

class Population():

    def __init__(self, family, useGrid, numOCW, numGrids, n, startLimit, granularity) -> None:
        self.family = family
        self.useGrid = useGrid
        self.numOCW = numOCW
        self.numGrids = numGrids
        self.n = n
        self.startLimit = startLimit
        self.granularity = granularity
        self.nodes = self.init_nodes()


    def restart(self):

        node : Node
        for node in self.nodes:

            node.ocw = np.random.randint(self.numOCW)
            node.startTime = np.random.randint(self.startLimit)
            seq_id = np.random.randint(len(self.family))
            node.seq = self.family[seq_id]

            node.grid = 0
            if self.useGrid:
                node.grid = np.random.randint(self.numGrids)

            node.gran = 0
            if self.granularity:
                node.gran = np.random.randint(self.granularity)
        

    def init_nodes(self):

        nodes = []
        for _ in range(self.n):
            
            ocw = np.random.randint(self.numOCW)
            startTime = np.random.randint(self.startLimit)
            seq_id = np.random.randint(len(self.family))
            seq = self.family[seq_id]

            grid = 0
            if self.useGrid:
                grid = np.random.randint(self.numGrids)

            gran = 0
            if self.granularity:
                gran = np.random.randint(self.granularity)


            nodes.append(Node(seq, ocw, grid, startTime, gran))
        
        return nodes

