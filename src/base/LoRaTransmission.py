
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
        sequence (list): sequence of frequence hopping channels 
    """

    def __init__(self, id, node_id, startSlot, ocw, grid, payload_size,
                 numFragments, sequence) -> None:
        self.id = id
        self.node_id = node_id
        self.startSlot = startSlot
        self.ocw = ocw
        self.grid = grid
        self.payload_size = payload_size
        self.numFragments = numFragments
        self.sequence = sequence
    
    