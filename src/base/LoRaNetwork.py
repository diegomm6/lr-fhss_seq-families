import numpy as np
from src.base.Event import *
from src.base.LoRaNode import LoRaNode
from src.base.LoRaGateway import LoRaGateway
from src.base.LoRaTransmission import LoRaTransmission

class LoRaNetwork():

    def __init__(self, numNodes, family, useGrid, numOCW, numOBW, numGrids, CR,
                 granularity, max_seq_length, simTime, numDecoders, use_earlydecode) -> None:
        
        self.family = family
        self.numOCW = numOCW
        self.numOBW = numOBW
        self.granularity = granularity
        self.simTime = simTime
        self.use_earlydecode = use_earlydecode

        assert CR==1 or CR==2, "Only CR 1/3 and CR 2/3 supported"

        startLimit = simTime - (granularity * max_seq_length)
        self.nodes = [LoRaNode(i, CR, numOCW, useGrid, numGrids, startLimit)
                      for i in range(numNodes)]

        # add support for multiple gateways in the future
        self.gateway = LoRaGateway(granularity, CR, numDecoders)


    def get_transmissions(self) -> list[LoRaTransmission]:

        transmissions = []
        node : LoRaNode
        for node in self.nodes:
            transmissions += node.get_transmissions(self.family)

        return transmissions
    

    def get_collision_matrix(self, transmissions: list[LoRaTransmission]) -> np.ndarray:

        collision_matrix = np.zeros((self.numOCW, self.numOBW, self.simTime))
        
        tx : LoRaTransmission
        for tx in transmissions:

            for t, obw in enumerate(tx.sequence):

                timeSlot = tx.startSlot + (t * self.granularity)
                for g in range(self.granularity):

                    collision_matrix[tx.ocw][obw + tx.grid][timeSlot + g] += 1

        return collision_matrix
    

    def get_events(self, collision_matrix: np.ndarray, transmissions: list[LoRaTransmission]) -> list[EndEvent | CollisionEvent | StartEvent]:

        start_events = self.get_start_events(transmissions)
        collision_events = self.get_collision_events(collision_matrix)
        end_events = self.get_end_events(transmissions)

        events = start_events + collision_events + end_events

        if self.use_earlydecode:
            events += self.get_earlydecode_events()

        sorted_events = sorted(events)
        #for ev in sorted_events: print(ev)

        return sorted_events
        

    def get_start_events(self, transmissions: list[LoRaTransmission]) -> list[StartEvent]:

        start_events = []
        tx : LoRaTransmission
        for tx in transmissions:
            new_start_event = StartEvent(tx.startSlot, 'start', tx)
            start_events.append(new_start_event)

        return start_events
    

    def get_end_events(self, transmissions: list[LoRaTransmission]) -> list[EndEvent]:

        end_events = []
        for tx in transmissions:
            tx_end = tx.startSlot + (tx.numFragments * self.granularity) - 1
            new_end_event = EndEvent(tx_end, 'end', tx)
            end_events.append(new_end_event)

        return end_events


    def get_collision_events(self, collision_matrix: np.ndarray) -> list[CollisionEvent]:

        collision_events = []
        for t in range(self.simTime):
            sub_collision_matrix = collision_matrix[:,:,t]
            indeces = np.where(sub_collision_matrix > 1)

            for ocw, obw in zip(indeces[0], indeces[1]):
                new_collision_event = CollisionEvent(t, 'collision', ocw, obw)
                collision_events.append(new_collision_event)

        return collision_events
    

    def get_earlydecode_events(self, period=1) -> list[EarlyDecodeEvent]:

        earlydecode_events = []
        t = period
        while t < self.simTime:
            new_earlydecode_event = EarlyDecodeEvent(t, 'early_decode')
            earlydecode_events.append(new_earlydecode_event)
            t += period

        return earlydecode_events


    def run(self) -> None:

        transmissions = self.get_transmissions()
        collision_matrix = self.get_collision_matrix(transmissions)
        events = self.get_events(collision_matrix, transmissions)

        self.gateway.run(events)


    def get_collided_packets(self) -> int:
        return self.gateway.get_collided_packets()
    

    def get_decoded_packets(self) -> int:
        return self.gateway.get_decoded_packets()
    

    def get_decoded_bytes(self) -> int:
        return self.gateway.get_decoded_bytes()
    

    def restart(self) -> None:

        self.gateway.restart()

        node : LoRaNode
        for node in self.nodes:
            node.restart()
    