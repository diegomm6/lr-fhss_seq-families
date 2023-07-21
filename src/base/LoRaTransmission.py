
class LoRaTransmission():
    """
    A class represents a LoRa FHSS transmission

    Args:
        id (int): transmission unique identifier
        node_id (int): sender node unique identifier
        startSlot (int): time slot when transmission stars
        ocw (int): OCW channel where transmission takes place
        grid (int): grid number used by the transmission
        numFragments (int): number of payload fragments
        sequence (list[int]): sequence of frequence hopping channels 
    """

    def __init__(self, id: int, node_id: int, startSlot: int, ocw: int, header_replicas: int,
                 payload_size: int, numFragments: int, sequence: list[int]) -> None:
        self.id = id
        self.node_id = node_id
        self.startSlot = startSlot
        self.ocw = ocw
        self.header_replicas = header_replicas
        self.payload_size = payload_size
        self.numFragments = numFragments
        self.sequence = sequence
    