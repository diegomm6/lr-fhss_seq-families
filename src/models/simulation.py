import numpy as np

class Simulation():
    """
    A class that simulates time slotted communication.

    Args:
        nodes (int): The number of nodes in the simulation.
        family (list): A list of sequences that can be transmitted.
        useGrid (bool): Whether to use a grid.
        numOCW (int): The number of orthogonal frequency channels (OCWs).
        numOBW (int): The number of orthogonal bandwidths (OBWs).
        numGrids (int): The number of grids.
        startLimit (int): The start limit.
        seq_length (int): The sequence length.

    Methods:
        get_seq_time(): Obtains a random slot to start transmission and a random sequence from the family.
        run(): Runs the simulation and returns a NumPy array of transmissions.
    """

    def __init__(self, nodes, family, useGrid, numOCW, numOBW, numGrids, startLimit, seq_length) -> None:
        self.nodes = nodes
        self.family = family
        self.useGrid = useGrid
        self.numOCW = numOCW
        self.numOBW = numOBW
        self.numGrids = numGrids
        self.startLimit = startLimit
        self.seq_length = seq_length


    def get_seq_time(self):
        """
        Obtains a random slot to start transmission and a random sequence from the family.

        Returns:
            A tuple of the random sequence and the random start time.
        """

        startTime = np.random.randint(self.startLimit)
        seq_id = np.random.randint(len(self.family))
        return self.family[seq_id], startTime
    

    def run(self):
        """
        Runs the simulation and returns a NumPy array of transmissions.

        Returns:
            A NumPy array of transmissions.
        """

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
    