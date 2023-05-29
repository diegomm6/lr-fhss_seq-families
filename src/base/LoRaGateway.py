from src.base.Event import *
from src.base.Processor import Processor

class LoRaGateway():

    """
    A class for decoding LoRa transmissions.

    Args:
        granularity (int): The granularity of the decoding process, in seconds.
        CR (int): The coding rate of the LoRa transmissions.
        numDecoders (int): The number of decoders in the gateway.
        decodeCapacity (int): The decoding capacity of each decoder.

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

    def __init__(self, granularity: int, CR: int, numDecoders: int, decodeCapacity: int) -> None:
        self._processors = self.init_processors(granularity, CR, numDecoders, decodeCapacity)


    def init_processors(self, granularity: int, CR: int, numDecoders: int, decodeCapacity: int) -> list[Processor]:
        """
        Instanciate processors
        """
        numProcessors = numDecoders * decodeCapacity
        processors = [Processor(granularity, CR) for _ in range(numProcessors)]
        return processors
    

    def restart(self) -> None:
        """
        Reset all processor to initial state
        """
        processor : Processor
        for processor in self._processors:
            processor.reset()


    def available_processor(self) -> Processor | None:
        """
        Return the first availabe processor or None if no processor is available
        """

        processor : Processor
        for processor in self._processors:
            if not processor.is_busy:
                return processor

        return None
    

    def get_collided_packets(self) -> int:
        """
        Return total collided packets
        """

        collided_packets = 0
        processor : Processor
        for processor in self._processors:
            collided_packets += processor.collided_packets

        return collided_packets
    

    def get_decoded_packets(self) -> int:
        """
        Return total successfully decoded packets
        """

        decoded_packets = 0
        processor : Processor
        for processor in self._processors:
            decoded_packets += processor.decoded_packets

        return decoded_packets
    

    def get_decoded_bytes(self) -> int:
        """
        Return total successfully decoded bytes
        """

        decoded_bytes = 0
        processor : Processor
        for processor in self._processors:
            decoded_bytes += processor.decoded_bytes

        return decoded_bytes
    

    def run(self, events: list[AbstractEvent]) -> None:
        """
        Execute all given events
        """

        for event in events:
        
            if event._name == 'start':
                processor = self.available_processor()
                if processor is not None:
                    processor.start_decoding(event)

            elif event._name == 'collision':
                for processor in self._processors:
                    processor.handle_collision(event)

            elif event._name == 'end':
                for processor in self._processors:
                    processor.finish_decoding(event)

            else:
                raise Exception(f"Invalid event name {event._name}") 
