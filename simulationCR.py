import numpy as np
from Population import Population
from Node import Node

class SimulationCR():
    """
    A class that simulates time slotted communication with coding rate

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

    def __init__(self, nodes, family, useGrid, numOCW, numOBW, numGrids, startLimit, seq_length, CR) -> None:
        self.numOCW = numOCW
        self.numOBW = numOBW
        self.startLimit = startLimit
        self.seq_length = seq_length

        self.threshold = self.get_thershold(CR)
        self.population = Population(family, useGrid, numOCW, numGrids, nodes, startLimit)


    def get_thershold(self, CR):
        # 1/3 of packets needed
        if CR == 1:
            return self.seq_length - np.ceil(self.seq_length / 3)
        
        # 2/3 of packets needed
        return self.seq_length - np.ceil(2 * self.seq_length / 3)


    def get_packet_collision_rate(self, runs):

        avg_collided_rate = 0
        for r in range(runs):

            txData = self.run()

            collided_packets = 0
            for node in self.population.nodes:

                collided_fragments = 0      
                for t, obw in enumerate(node.seq):

                    if txData[node.ocw][obw + node.grid][node.startTime + t] != 1:
                        collided_fragments += 1
                    
                if collided_fragments > self.threshold:
                    collided_packets += 1

            avg_collided_rate += collided_packets / len(self.population.nodes)
        
        return avg_collided_rate / runs


    def run(self):
        """
        Runs the simulation and returns a NumPy array of transmissions.

        Returns:
            A NumPy array of transmissions.
        """

        transmissions = np.zeros((self.numOCW,
                                  self.numOBW,
                                  self.startLimit + self.seq_length))
        
        for node in self.population.nodes:            
            for t, obw in enumerate(node.seq):
                transmissions[node.ocw][obw + node.grid][node.startTime + t] += 1

        return transmissions