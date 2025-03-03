from src.base.base import *

class LRFHSSTransmission():
    """
    A class represents a LR-FHSS transmission

    Args:
        id (int): transmission unique identifier
        node_id (int): node unique identifier
        startSlot (int): time slot when transmission stars
        ocw (int): OCW channel where transmission takes place
        numHeaders (int): number of header repetitions
        payload_size (int): information bytes encoded in the payload
        numFragments (int): number of payload fragments
        sequence (list[int]): sequence of frequence hopping channels 
        seqid (int): frequence hopping sequence unique identifier
        distance (float): node-satellite distance at the beggining of the signal
        dopplerShift (list[float]): static doppler shift list for each header/fragment
        power (float): transmission power
    """

    def __init__(self, id: int, node_id: int, startSlot: int, ocw: int, numHeaders: int,
                 payload_size: int, numFragments: int, sequence: list[int], seqid: int,
                 distance: float, dopplerShift: list[float], power: float) -> None:
        self.id = id
        self.node_id = node_id
        self.startSlot = startSlot
        self.ocw = ocw
        self.numHeaders = numHeaders
        self.payload_size = payload_size
        self.numFragments = numFragments
        self.sequence = sequence
        self.seqid = seqid
        self.distance = distance
        self.dopplerShift = dopplerShift
        self.power = power
    
    def __lt__(self, other) -> bool:
        return self.startSlot < other.startSlot
    
    def __str__(self) -> str:
        return f"tx {self.id} at time {self.startSlot}, {self.numFragments} fragments"
    