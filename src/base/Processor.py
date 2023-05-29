import numpy as np
from src.base.Event import *
from src.base.LoRaTransmission import LoRaTransmission

class Processor():

    def __init__(self, CR, granularity) -> None:
        self.CR = CR
        self.granularity = granularity
        self._current_tx : LoRaTransmission = None
        self._current_collided_fragments = 0
        self.is_busy = False
        self.collided_packets = 0
        self.decoded_packets = 0
        self.decoded_bytes = 0


    def reset(self):
        self._current_tx = None
        self._current_collided_fragments = 0
        self.is_busy = False
        self.decoded_packets = 0
        self.decoded_bytes = 0
        self.collided_packets = 0


    def get_thershold(self, seq_length : int) -> int:
        # 1/3 of packets needed
        if self.CR == 1:
            return seq_length - np.ceil(seq_length / 3)
        
        # 2/3 of packets needed
        return seq_length - np.ceil(2 * seq_length / 3)


    def start_decoding(self, start_event : StartEvent):
        self._current_tx = start_event._transmission
        self._current_collided_fragments = 0
        self.is_busy = True

    def finish_decoding(self, end_event : EndEvent):

        if not self.is_busy:
            return

        if self._current_tx.id == end_event._transmission.id:

            thershold = self.get_thershold(end_event._transmission.numFragments)
            
            if self._current_collided_fragments <= thershold:
                self.decoded_packets += 1
                self.decoded_bytes += self._current_tx.payload_size

            else:
                self.collided_packets += 1

            self._current_tx = None
            self.is_busy = False
    

    def handle_collision(self, collision_event : CollisionEvent):

        # if process is not receiving a signal return
        # else, current_tx is not none
        if not self.is_busy:
            return

        t = collision_event._time
        ocw = collision_event._ocw
        obw = collision_event._obw

        current_obw_id = (t - self._current_tx.startSlot) // self.granularity
        current_obw = self._current_tx.sequence[current_obw_id]

        # is current_tx is involved in collision, fragment is collided
        if self._current_tx.ocw == ocw and current_obw == obw:
            self._current_collided_fragments += 1

        
