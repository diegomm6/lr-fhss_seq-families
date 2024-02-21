import numpy as np
from src.base.Processor import Processor
from src.base.LoRaTransmission import LoRaTransmission

class LoRaGateway():

    """
    A class for decoding LoRa transmissions.

    Args:
        CR (int): The coding rate of the LoRa transmissions.
        timeGranularity (int): number of time slots per fragment.
        freqGranularity (int): number of frequency slots per OBW.
        use_earlydrop (bool): early drop mechanism flag.
        use_earlydecode (bool): early decode mechanism flag.
        use_headerdrop (bool): header drop mechanism flag.
        numDecoders (int): number of decoders/processors in the gateway.

    Attributes:
        _processors (list[Processor]): A list of processors that handle decoding of LoRa transmissions.

    Methods:
        restart(): Reset all processors to initial state.
        available_processor(): Return the first available processor or None if no processor is available.
        get_collided_packets(): Return total collided packets.
        get_decoded_packets(): Return total successfully decoded packets.
        get_decoded_bytes(): Return total successfully decoded bytes.
        run(list[AbstractEvent]): Decode given list of transmissions.
    """

    def __init__(self, CR: int, timeGranularity: int, freqGranularity: int, use_earlydrop: bool, 
                 use_earlydecode: bool, use_headerdrop: bool, numDecoders: int, baseFreq: int, collision_method: str) -> None:
        self.numDecoders = numDecoders
        self._processors = [Processor(CR, timeGranularity, freqGranularity, use_earlydrop, use_earlydecode,
                                      use_headerdrop, baseFreq, collision_method) for _ in range(numDecoders)]
    

    def restart(self) -> None:
        """
        Reset all processor to initial state
        """
        processor : Processor
        for processor in self._processors:
            processor.reset()
        
    def get_decoded(self) -> list:
        """
        Return decoded transmissiond, 1 means decoded payload
        """
        decoded = []
        for processor in self._processors:
            for tx_status in processor.decoded:
                decoded.append(tx_status)
        return decoded

    def get_tracked_txs(self) -> int:
        """
        Return total tracked frames by the gateway
        """
        tracked_txs = 0
        processor : Processor
        for processor in self._processors:
            tracked_txs += processor.tracked_txs
        return tracked_txs
    
    def get_decoded_bytes(self) -> int:
        """
        Return total successfully decoded bytes
        """
        decoded_bytes = 0
        processor : Processor
        for processor in self._processors:
            decoded_bytes += processor.decoded_bytes
        return decoded_bytes
    
    def get_header_drop_packets(self) -> int:
        """
        Return packets dropped due to no header decoding
        """
        header_drop_packets = 0
        processor : Processor
        for processor in self._processors:
            header_drop_packets += processor.header_drop_packets
        return header_drop_packets
    
    def get_decoded_hrd_pld(self) -> int:
        """
        Return total fully decoded packets
        """
        decoded_hrd_pld = 0
        processor : Processor
        for processor in self._processors:
            decoded_hrd_pld += processor.decoded_hrd_pld
        return decoded_hrd_pld

    def get_decoded_hdr(self) -> int:
        """
        Return total decoded headers with collided payload
        """
        decoded_hdr = 0
        processor : Processor
        for processor in self._processors:
            decoded_hdr += processor.decoded_hdr

        return decoded_hdr
    
    def get_decodable_pld(self) -> int:
        """
        Return total decodable payloads with collided header
        """
        decodable_pld = 0
        processor : Processor
        for processor in self._processors:
            decodable_pld += processor.decodable_pld
        return decodable_pld
    
    def get_collided_hdr_pld(self) -> int:
        """
        Return total fully collided packets
        """
        collided_hdr_pld = 0
        processor : Processor
        for processor in self._processors:
            collided_hdr_pld += processor.collided_hdr_pld
        return collided_hdr_pld
    

    def run(self, transmissions: list[LoRaTransmission],
            collision_matrix: np.ndarray, dynamic: bool) -> None:

        freeUpTimes = np.zeros(self.numDecoders)
        for tx in transmissions:
            for i, fut in enumerate(freeUpTimes):

                if tx.startSlot >= fut:
                    processor = self._processors[i]
                    freeUpTimes[i] = processor.decode(tx, collision_matrix, dynamic)
                    break
