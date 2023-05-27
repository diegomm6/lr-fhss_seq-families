import numpy as np
from src.base.LoRaTransmission import LoRaTransmission

class LoRaNode():

    def __init__(self, id, CR, numOCW, useGrid, numGrids, startLimit) -> None:
        self.id = id
        self.CR = CR
        self.numOCW = numOCW
        self.useGrid = useGrid
        self.numGrids = numGrids
        self.startLimit = startLimit


    def numHops(self, payload_length):
        """
        calculate the number of frequency hops for the payload
        as a function of its size and the coding rate
        """

        length_bits = ( payload_length + 2 ) * 8 + 6
        length_bits *= (3/self.CR)

        nb_hops_out = ( length_bits + 47 ) // 48

        return nb_hops_out


    def get_transmissions(self, family):
        """
        Obtain all transmission during simulation time for this node.
        The current model only support one transmission per node,
        therefore we use the node id as the transmission id
        """

        ocw = np.random.randint(self.numOCW)

        grid = 0
        if self.useGrid:
            grid = np.random.randint(self.numGrids)

        seq_id = np.random.randint(len(family))
        sequence = family[seq_id]

        startTime = np.random.randint(self.startLimit)

        # payload length is set to 58 bytes when CR1 is used
        # and 121 bytes for CR2, this ensures 31 fragments
        payload_length = 58
        if self.CR == 2:
            payload_length = 121

        numFragments = self.numHops(payload_length)

        tx = LoRaTransmission(self.id, self.id, startTime, ocw,
                              grid, numFragments, sequence)

        return [tx]
