import numpy as np
from src.base.Event import *
from src.base.Processor import Processor
from src.base.LoRaTransmission import LoRaTransmission

class LoRaGateway():
    """
    A class that representes a LoRa Gateway with decoders

    Args:
        numDecoders (int): The number of decoders on the gateway
        decodeCapacity (int): The number of simultaneous demodulations per decoder

    Methods:
        init_processors(): initialize processor
        
    """


    def __init__(self, granularity, simTime, threshold, numDecoders, decodeCapacity) -> None:
        self.granularity = granularity
        self.simTime = simTime
        self.threshold = threshold
        self._processors = self.init_processors(numDecoders, decodeCapacity)


    def init_processors(self, numDecoders, decodeCapacity):
        numProcessors = numDecoders * decodeCapacity
        processors = [Processor(self.threshold, self.granularity) for _ in range(numProcessors)]
        return processors
    

    def available_processor(self):
        """
        Return the first availabe processor or None if no processor is available
        """

        processor : Processor
        for processor in self._processors:
            if not processor.is_busy:
                return processor

        return None
    

    def get_decoded_transmissions(self):
        """
        Return total successfully decoded transmissions
        """

        decoded = 0
        processor : Processor
        for processor in self._processors:
            decoded += processor.decoded_tx

        return decoded

    
    def run(self, events):
        """
        Execute all given events
        """

        event : AbstractEvent
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
