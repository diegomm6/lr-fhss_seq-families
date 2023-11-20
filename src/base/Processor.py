import numpy as np
from src.base.Event import *
from src.base.LoRaTransmission import LoRaTransmission

class Processor():
    """
    A class for decoding LoRa transmissions.

    Args:
        granularity (int): The granularity of the decoding process, in slots.
        CR (int): The coding rate of the LoRa transmissions.

    Attributes:
        granularity (int): The granularity of the decoding process, in slots.
        CR (int): The coding rate of the LoRa transmissions.
        _current_tx (LoRaTransmission): The current transmission being decoded.
        _fragment_status (list[int]): Status of each fragment of the current frame.
        is_busy (bool): Whether the processor is currently decoding a transmission.
        collided_packets (int): The number of transmissions that have collided.
        decoded_packets (int): The number of transmissions that have been decoded successfully.
        decoded_bytes (int): The number of bytes that have been decoded successfully.

    Methods:
        reset(): Resets all counters and states to initial state.
        get_thershold(seq_length: int) -> int:
            Determine the minimum number of fragments needed to successfully
            decode a packet. Two Coding Rates supported:
                if CR==1 then a 1/3 of fragmenst are required
                if CR==2 then a 2/3 of fragmenst are required
        start_decoding(start_event : StartEvent) -> None:
            Lock processor to the given transmission.
        finish_decoding(end_event : EndEvent) -> None:
            Check if the ending trasnmission handed by the gateway corresponds
            to the one being decoded by this processor then determine the outcome
            of the transmission and update counters 
        handle_collision(collision_event : CollisionEvent) -> None:
            Check if the current transmission being decoded matches the given
            collision event and if so updates the counter of collided fragments
    """

    def __init__(self, CR: int, timeGranularity: int, freqGranularity: int, use_earlydrop: bool,
                 use_earlydecode: bool, use_headerdrop: bool) -> None:
        self.CR = CR
        self.timeGranularity = timeGranularity
        self.freqGranularity = freqGranularity
        self.header_slots = round(timeGranularity * 233.472 / 102.4)
        self.use_earlydrop = use_earlydrop
        self.use_earlydecode = use_earlydecode
        self.use_headerdrop = use_headerdrop
        self.header_tolerance = 4

        self.tracked_txs = 0
        self.decoded_bytes = 0
        self.header_drop_packets = 0
        self.decoded_hrd_pld = 0  # case 1
        self.decoded_hdr = 0      # case 3
        self.decodable_pld = 0    # case 2
        self.collided_hdr_pld = 0 # case 4


    def reset(self) -> None:
        """
        Reset all counters and states to initial state
        """
        self.tracked_txs = 0
        self.decoded_bytes = 0
        self.header_drop_packets = 0
        self.decoded_hrd_pld = 0  # case 1
        self.decoded_hdr = 0      # case 3
        self.decodable_pld = 0    # case 2
        self.collided_hdr_pld = 0 # case 4


    def get_minfragments(self, seq_length : int) -> int:
        """
        Determine minimum required number of fragment for 
        successful decoding. Two Coding Rates supported:
            if CR==1 then a 1/3 of fragmenst are required
            if CR==2 then a 2/3 of fragmenst are required
        """
        return np.ceil(self.CR * seq_length / 3)
    

    def isCollided(self, subm: np.ndarray) -> bool:
        return not (subm == 1).all()

    
    def decode(self, tx: LoRaTransmission, collision_matrix: np.ndarray) -> int:
        """
        Determine status of incoming transmissions and return free up time
        """

        self.tracked_txs += 1
        collided_headers = 0
        decoded_fragments = 0
        collided_fragments = 0

        carrierOffset = 0
        maxDopplerShift = (200000 - 137000) / 2# 20000
        maxShift = carrierOffset + maxDopplerShift
        freqPerSlot= 488.28125 / self.freqGranularity

        time = tx.startSlot
        baseFreq = round((maxShift + tx.dopplerShift) / freqPerSlot)

        maxFrgCollisions = tx.numFragments - self.get_minfragments(tx.numFragments)

        for fh, obw in enumerate(tx.sequence):

            startFreq = baseFreq + obw * self.freqGranularity
            endFreq = startFreq + self.freqGranularity

            # header
            if fh < tx.numHeaders:

                endTime = time + self.header_slots
                header = collision_matrix[tx.ocw, startFreq : endFreq, time : endTime]

                if self.isCollided(header):
                    collided_headers += 1
                
                time = endTime

            # fragment
            else:

                # collided header, abort payload reception
                if self.use_headerdrop and collided_headers == tx.numHeaders:
                    self.header_drop_packets += 1
                    return endTime

                endTime = time + self.timeGranularity
                fragment = collision_matrix[tx.ocw, startFreq : endFreq, time : endTime]

                if self.isCollided(fragment):
                    collided_fragments += 1

                else:
                    decoded_fragments += 1

                # early decode - decoded or decodable payload
                if self.use_earlydecode and decoded_fragments >= self.get_minfragments(tx.numFragments):

                    # case 1
                    if collided_headers < tx.numHeaders:
                        self.decoded_hrd_pld += 1
                        self.decoded_bytes += tx.payload_size

                    # case 2
                    else:
                        self.decodable_pld += 1

                    return endTime

                # early drop - collided payload
                if self.use_earlydrop and collided_fragments > maxFrgCollisions:
                    
                    # case 3
                    if collided_headers < tx.numHeaders:
                        self.decoded_hdr += 1

                    # case 4
                    else:
                        self.collided_hdr_pld += 1
                    
                    return endTime

                time = endTime

