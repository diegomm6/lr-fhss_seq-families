
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

    def __init__(self, id: int, node_id: int, startSlot: int, ocw: int, numHeaders: int,
                 payload_size: int, numFragments: int, sequence: list[int], dopplerShift: float) -> None:
        self.id = id
        self.node_id = node_id
        self.startSlot = startSlot
        self.ocw = ocw
        self.numHeaders = numHeaders
        self.payload_size = payload_size
        self.numFragments = numFragments
        self.sequence = sequence
        self.dopplerShift = dopplerShift
    
    def __lt__(self, other) -> bool:
        return self.startSlot < other.startSlot
    
    def __str__(self) -> str:
        return f"tx {self.id} at time {self.startSlot}, {self.numFragments} fragments"
    