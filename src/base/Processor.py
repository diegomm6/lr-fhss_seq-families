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

    def __init__(self, granularity: int, CR: int) -> None:
        self.CR = CR
        self.granularity = granularity
        self._current_tx : LoRaTransmission = None
        self._fragment_status = []
        self.is_busy = False
        self.collided_packets = 0
        self.decoded_packets = 0
        self.decoded_bytes = 0


    def reset(self) -> None:
        """
        Reset all counters and states to initial state
        """

        self.is_busy = False
        self._current_tx = None
        self._fragment_status = []
        self.collided_packets = 0
        self.decoded_packets = 0
        self.decoded_bytes = 0


    def get_thershold(self, seq_length : int) -> int:
        """
        Determine limit of collided fragments to successfully
        decode a packet. Two Coding Rates supported:
            if CR==1 then a 1/3 of fragmenst are required
            if CR==2 then a 2/3 of fragmenst are required
        """

        return seq_length - np.ceil(self.CR * seq_length / 3)


    def start_decoding(self, start_event : StartEvent) -> None:
        """
        Lock processor to the given transmission
        """

        self.is_busy = True
        self._current_tx = start_event._transmission
        self._fragment_status = np.zeros(int(start_event._transmission.numFragments))
        

    def finish_decoding(self, end_event : EndEvent) -> None:
        """
        Check if the ending trasnmission handed by the gateway corresponds
        to the one being decoded by this processor then determine the outcome
        of the transmission and update counters 
        """

        if not self.is_busy:
            return

        if self._current_tx.id == end_event._transmission.id:

            thershold = self.get_thershold(end_event._transmission.numFragments)
            collided_fragments = (self._fragment_status == 1).sum()
            
            if collided_fragments <= thershold:
                self.decoded_packets += 1
                self.decoded_bytes += self._current_tx.payload_size

            else:
                self.collided_packets += 1

            self.is_busy = False
            self._current_tx = None
            self._fragment_status = []
    

    def handle_collision(self, collision_event : CollisionEvent) -> None:
        """
        Check if the current transmission being decoded matches the given
        collision event and if so updates the counter of collided fragments
        """

        # if process is not receiving a signal return
        # else, current_tx is not none
        if not self.is_busy:
            return

        t = collision_event._time
        ocw = collision_event._ocw
        obw = collision_event._obw

        current_fragment_id = (t - self._current_tx.startSlot) // self.granularity
        current_obw = self._current_tx.sequence[current_fragment_id] + self._current_tx.grid

        # is current_tx is involved in collision, fragment is collided
        if self._current_tx.ocw == ocw and current_obw == obw:
            self._fragment_status[current_fragment_id] = 1

        # if collided fragments surpass the threshold
        # then discard the packet and free the processor
        thershold = self.get_thershold(self._current_tx.numFragments)
        collided_fragments = (self._fragment_status == 1).sum()
        if collided_fragments > thershold:
            self.collided_packets += 1
            self.is_busy = False
            self._current_tx = None
            self._fragment_status = []
    

    def early_decode(self, earlydecode_event : EarlyDecodeEvent)-> None:

        if not self.is_busy:
            return
        
        # if collided fragments surpass the threshold
        # then discard the packet and free the processor
        thershold = self.get_thershold(self._current_tx.numFragments)
        minFragments = self._current_tx.numFragments - thershold

        current_fragment_id = (earlydecode_event._time - self._current_tx.startSlot) // self.granularity
        validFragments = (self._fragment_status[:current_fragment_id+1] == 0).sum()

        if validFragments >= minFragments:
            self.decoded_packets += 1
            self.decoded_bytes += self._current_tx.payload_size
            self.is_busy = False
            self._current_tx = None
            self._fragment_status = []
