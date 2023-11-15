import random
from src.base.base import get_randomDoppler
from src.base.LoRaTransmission import LoRaTransmission
from src.families.LR_FHSS_DriverMethod import FHSfamily

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
    def __init__(self, id, CR, numOCW, startLimit) -> None:
        self.id = id
        self.CR = CR
        self.numOCW = numOCW
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

        nb_hops_out = ( length_bits + 47 ) // 48  # can be just ceil(bits/48)

        return nb_hops_out


    def get_transmissions(self, family: FHSfamily) -> list[LoRaTransmission]:
        """
        Obtain all transmission during simulation time for this node.
        The current model only support one transmission per node,
        therefore we use the node id as the transmission id
        """

        ocw = random.randrange(0, self.numOCW)
        startSlot = random.randrange(0, self.startLimit)

        if self.CR == 1:
            payload_size = random.randrange(13, 58)  # 8 - 30 fragments
            numHeaders = 3

        elif self.CR == 2:
            payload_size = random.randrange(29, 118) # 8 - 30 fragments
            numHeaders = 2

        else:
            raise Exception(f"Invalid coding rate '{self.CR}'") 

        numFragments = self.numHops(payload_size)
        seq_length = int(numFragments + numHeaders)

        sequence = family.get_random_sequence()
        sequence = sequence[:seq_length]

        self.sent_packets += 1
        self.sent_payload_bytes += payload_size

        dopplerShift = get_randomDoppler()

        tx = LoRaTransmission(self.id, self.id, startSlot, ocw, numHeaders,
                              payload_size, numFragments, sequence, dopplerShift)

        return [tx]
