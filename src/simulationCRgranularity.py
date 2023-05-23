import numpy as np

from src.base.Node import Node
from src.base.Population import Population


class SimulationCRgranularity():
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
        CR (int): Coding rate
        granularity(int): internal time slot subdivisions  

    Methods:
        get_thershold(): determine the minimum number of fragments required according to the CR
        get_packet_collision_rate(): calculate the packet collision rate 
        run(): Runs the simulation and returns a NumPy array of transmissions.
    """

    def __init__(self, nodes, family, useGrid, numOCW, numOBW, numGrids,
                  startLimit, seq_length, CR, granularity) -> None:
        self.numOCW = numOCW
        self.numOBW = numOBW
        self.startLimit = startLimit
        self.seq_length = seq_length
        self.granularity = granularity

        self.threshold = self.get_thershold(CR)
        self.population = Population(family, useGrid, numOCW, numGrids, nodes, startLimit, granularity)


    def get_thershold(self, CR : int) -> int:
        # 1/3 of packets needed
        if CR == 1:
            return self.seq_length - np.ceil(self.seq_length / 3)
        
        # 2/3 of packets needed
        return self.seq_length - np.ceil(2 * self.seq_length / 3)


    def get_packet_collision_rate(self, runs : int) -> float:

        avg_collided_rate = 0
        for r in range(runs):

            txData = self.run()

            node : Node
            collided_packets = 0
            for node in self.population.nodes:

                collided_fragments = 0
                for t, obw in enumerate(node.seq):
                    
                    collided = False
                    slot = node.gran + self.granularity * (node.startTime + t)
                    for g in range(self.granularity):
                        
                        if txData[node.ocw][obw + node.grid][slot + g] != 1:
                            collided = True

                    if collided:
                        collided_fragments += 1
                    
                if collided_fragments > self.threshold:
                    collided_packets += 1

            avg_collided_rate += collided_packets / len(self.population.nodes)
            self.population.restart()
        
        return avg_collided_rate / runs


    def run(self):
        """
        Runs the simulation and returns a NumPy array of transmissions.

        Returns:
            A NumPy array of transmissions.
        """
        timeSlots = self.granularity * (self.startLimit + self.seq_length + 1)
        transmissions = np.zeros((self.numOCW,
                                  self.numOBW,
                                  timeSlots))
        
        node : Node
        for node in self.population.nodes:

            for t, obw in enumerate(node.seq):

                slot = node.gran + self.granularity * (node.startTime + t)
                for g in range(self.granularity):

                    transmissions[node.ocw][obw + node.grid][slot + g] += 1

        return transmissions