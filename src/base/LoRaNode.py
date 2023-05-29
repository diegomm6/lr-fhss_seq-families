import numpy as np
from src.base.LoRaTransmission import LoRaTransmission

class LoRaNode():
    """
    A class representing a LoRa node.

    Args:
        id: The node ID.
        CR: The coding rate.
        numOCW: The number of occupied chirps per symbol.
        useGrid: Whether to use a grid topology.
        numGrids: The number of grids in the network.
        startLimit: The start limit of the simulation time.

    Attributes:
        id (int): The node ID.
        CR (int): The coding rate.
        numOCW (int): The number of occupied chirps per symbol.
        useGrid (bool): Whether to use a grid topology.
        numGrids (int): The number of grids in the network.
        startLimit (int): The start limit of the simulation time.
        sent_packets (int): The number of packets sent by the node.
        sent_payload_bytes (int): The number of payload bytes sent by the node.

    Methods:
        restart(): Reset the node's statistics.
        numHops(payload_length): Calculate the number of frequency hops for the payload.
        get_sequence(family): Get the sequence for the payload.
        get_transmissions(family): Get all transmissions during simulation time for this node.

    """
    def __init__(self, id, CR, numOCW, useGrid, numGrids, startLimit) -> None:
        self.id = id
        self.CR = CR
        self.numOCW = numOCW
        self.useGrid = useGrid
        self.numGrids = numGrids
        self.startLimit = startLimit
        self.sent_packets = 0
        self.sent_payload_bytes = 0


    def restart(self) -> None:
        self.sent_packets = 0
        self.sent_payload_bytes = 0


    def numHops(self, payload_length: int) -> int:
        """
        Calculate the number of frequency hops for the payload
        as a function of its size and the coding rate
        """

        length_bits = ( payload_length + 2 ) * 8 + 6
        length_bits *= (3/self.CR)

        nb_hops_out = ( length_bits + 47 ) // 48

        return nb_hops_out
    

    def get_sequence(self, family):

        # payload size is set to 58 bytes when CR1 is used
        # and 121 bytes for CR2, this ensures 31 fragments
        payload_size = np.random.randint(10, 58+1)
        if self.CR == 2:
            payload_size = np.random.randint(10, 121+1)

        numFragments = self.numHops(payload_size)

        seq_id = np.random.randint(len(family))
        sequence = family[seq_id]

        seq_start_id = np.random.randint(len(sequence))
        seq_end_id = (seq_start_id + numFragments) % len(sequence)

        if seq_start_id < seq_end_id:
            final_sequence = sequence[seq_start_id:seq_end_id]

        else:
            final_sequence = sequence[seq_start_id:] + sequence[:seq_end_id]

        return final_sequence


    def get_transmissions(self, family) -> list[LoRaTransmission]:
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

        # payload size is set to 58 bytes when CR1 is used
        # and 121 bytes for CR2, this ensures 31 fragments
        payload_size = 58
        if self.CR == 2:
            payload_size = 121

        self.sent_packets += 1
        self.sent_payload_bytes += payload_size

        numFragments = self.numHops(payload_size)

        tx = LoRaTransmission(self.id, self.id, startTime, ocw, grid,
                              payload_size, numFragments, sequence)

        return [tx]
