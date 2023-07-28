import numpy as np
from src.base.Population import Node, Population

class LoRaNetworkLite():
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
        get_packet_collision_rate(): calculate the packet collision rate 
        run(): Runs the simulation and returns a NumPy array of transmissions.
    """

    def __init__(self, simTime, familyname, numGrids, numOCW, numOBW, numNodes,
                 numFragments, CR, granularity) -> None:
        
        self.simTime = simTime
        self.numOCW = numOCW
        self.numOBW = numOBW
        self.granularity = granularity
        self.minFragments = np.ceil(CR * numFragments / 3)
        self.header_replicas = 3 if CR==1 else 2
        self.seq_length = (numFragments * granularity) + (self.header_replicas * int(granularity * 7 / 3))

        max_packet_length_in_slots = (31 * granularity) + (3 * int(granularity * 7 / 3))
        startLimit = simTime - max_packet_length_in_slots

        self.population = Population(familyname, numGrids, numOCW, numNodes,
                                     startLimit, numFragments, self.header_replicas)


    def run(self, runs : int) -> list[float]:

        avg_decoded_payloads = 0
        avg_decoded_packets = 0
        for _ in range(runs):

            txData = self.get_collision_matrix()

            node : Node
            decoded_payloads = 0
            decoded_packets = 0
            for node in self.population.nodes:

                seq_status = np.zeros(len(node.sequence))
                for fh, obw in enumerate(node.sequence):
                    
                    # header collision
                    if fh < self.header_replicas:
                        used_timeslots = int(self.granularity * 7 / 3)
                        currentSlot = node.startSlot + fh * used_timeslots

                    # fragment collision
                    else:
                        used_timeslots = self.granularity
                        currentSlot = node.startSlot + \
                                      self.header_replicas * int(self.granularity * 7 / 3) + \
                                      (fh - self.header_replicas) * used_timeslots


                    for g in range(used_timeslots):
                        if txData[node.ocw][obw][currentSlot + g] != 1:
                            seq_status[fh] = 1


                validFragments = (seq_status[self.header_replicas:] == 0).sum()
                if validFragments >= self.minFragments:
                    decoded_payloads += 1 

                    if (seq_status[:self.header_replicas] == 1).sum() < self.header_replicas:
                        decoded_packets += 1 

            avg_decoded_payloads += decoded_payloads
            avg_decoded_packets += decoded_packets

            self.population.restart()
        
        return [avg_decoded_payloads / runs, avg_decoded_packets / runs]
    

    def channel_occupancy(self, runs : int) -> np.ndarray:

        avg_channel_occupancy = np.zeros((self.numOCW, self.numOBW))

        for _ in range(runs):

            txData = self.get_collision_matrix()
            channel_occupancy = np.apply_along_axis(lambda x : (x!=0).sum(), 2, txData)
            avg_channel_occupancy += (channel_occupancy / self.simTime)
            self.population.restart()

        return avg_channel_occupancy / runs
    

    def get_collision_matrix(self) -> np.ndarray:

        collision_matrix = np.zeros((self.numOCW, self.numOBW, self.simTime))
        
        node : Node
        for node in self.population.nodes:

            time = node.startSlot
            for fh, obw in enumerate(node.sequence):

                used_timeslots = self.granularity # write fragment
                if fh < self.header_replicas:     # write header
                    used_timeslots = int(self.granularity * 7 / 3)

                for g in range(used_timeslots):
                    collision_matrix[node.ocw][obw][time + g] += 1

                time += used_timeslots

        return collision_matrix
    