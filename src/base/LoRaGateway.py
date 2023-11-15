import numpy as np
from src.base.Event import *
from src.base.Processor import Processor

class LoRaGateway():

    """
    A class for decoding LoRa transmissions.

    Args:
        granularity (int): The granularity of the decoding process, in seconds.
        CR (int): The coding rate of the LoRa transmissions.
        numDecoders (int): The number of decoders in the gateway.

    Attributes:
        _processors (list[Processor]): A list of processors that handle decoding of LoRa transmissions.

    Methods:
        restart(): Reset all processors to initial state.
        available_processor(): Return the first available processor or None if no processor is available.
        get_collided_packets(): Return total collided packets.
        get_decoded_packets(): Return total successfully decoded packets.
        get_decoded_bytes(): Return total successfully decoded bytes.
        run(list[AbstractEvent]): Execute all given events.
    """

    def __init__(self, CR: int, timeGranularity: int, freqGranularity: int, use_earlydrop: bool, 
                 use_earlydecode: bool, use_headerdrop: bool, numDecoders: int) -> None:
        self.numDecoders = numDecoders
        self._processors = [Processor(CR, timeGranularity, freqGranularity, use_earlydrop, 
                                      use_earlydecode, use_headerdrop) for _ in range(numDecoders)]
    

    def restart(self) -> None:
        """
        Reset all processor to initial state
        """
        processor : Processor
        for processor in self._processors:
            processor.reset()

    
    def get_tracked_txs(self) -> int:
        """
        Return total tracked frames by the gateway
        """

        tracked_txs = 0
        processor : Processor
        for processor in self._processors:
            tracked_txs += processor.tracked_txs

        return tracked_txs
    

    def get_collided_payloads(self) -> int:
        """
        Return total collided payloads
        """

        collided_payloads = 0
        processor : Processor
        for processor in self._processors:
            collided_payloads += processor.collided_payloads

        return collided_payloads
    

    def get_decoded_packets(self) -> int:
        """
        Return total successfully decoded packets,
        successful payload reception and at least one header
        """

        decoded_packets = 0
        processor : Processor
        for processor in self._processors:
            decoded_packets += processor.decoded_packets

        return decoded_packets
    

    def get_decoded_payloads(self) -> int:
        """
        Return total successfully decoded payloads,
        regardless of header successful reception
        """

        decoded_payloads = 0
        processor : Processor
        for processor in self._processors:
            decoded_payloads += processor.decoded_payloads

        return decoded_payloads
    

    def get_decoded_bytes(self) -> int:
        """
        Return total successfully decoded bytes,
        considering only payload data
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
    

    def run(self, transmissions: list[LoRaTransmission], collision_matrix: np.ndarray) -> None:

        freeUpTimes = np.zeros(self.numDecoders)

        for tx in transmissions:

            for i, fut in enumerate(freeUpTimes):

                if tx.startSlot >= fut:
                    processor = self._processors[i]
                    freeUpTimes[i] = processor.decode(tx, collision_matrix)
                    break


