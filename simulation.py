import numpy as np

class Simulation():

    def __init__(self, nodes, family, useGrid, numOCW, numOBW, numGrids, startLimit, seq_length) -> None:
        self.nodes = nodes
        self.family = family
        self.useGrid = useGrid
        self.numOCW = numOCW
        self.numOBW = numOBW
        self.numGrids = numGrids
        self.startLimit = startLimit
        self.seq_length = seq_length

    # obtain a random slot to start transmission
    # and a random sequence from the family
    def get_seq_time(self):
        startTime = np.random.randint(self.startLimit)
        seq_id = np.random.randint(len(self.family))
        return self.family[seq_id], startTime
    

    def run(self):
        transmissions = np.zeros((self.numOCW,
                                  self.numOBW,
                                  self.startLimit + self.seq_length))

        for n in range(self.nodes):
            ocw = np.random.randint(self.numOCW)
            grid = 0
            if self.useGrid:
                grid = np.random.randint(self.numGrids)

            # choose random sequence and starting time
            seq, t0 = self.get_seq_time()

            for t, obw in enumerate(seq):
                transmissions[ocw][obw + grid][t0 + t] += 1

        return transmissions
    