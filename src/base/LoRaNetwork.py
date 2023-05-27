import numpy as np
from src.base.Event import *
from src.base.LoRaNode import LoRaNode
from src.base.LoRaGateway import LoRaGateway
from src.base.LoRaTransmission import LoRaTransmission

class LoRaNetwork():

    def __init__(self, numNodes, family, useGrid, numOCW, numOBW, numGrids, CR,
                 granularity, max_seq_length, simTime, numDecoders, decodeCapacity) -> None:
        
        self.family = family
        self.numOCW = numOCW
        self.numOBW = numOBW
        self.granularity = granularity
        self.simTime = simTime

        assert CR==1 or CR==2, "Only CR 1/3 and CR 2/3 supported"

        startLimit = simTime - max_seq_length - granularity -30
        self.nodes = [LoRaNode(i, CR, numOCW, useGrid, numGrids, startLimit)
                      for i in range(numNodes)]

        # add support for multiple gateways in the future
        self.gateway = LoRaGateway(granularity, simTime, CR, numDecoders, decodeCapacity)


    def get_transmissions(self):

        transmissions = []
        node : LoRaNode
        for node in self.nodes:
            transmissions += node.get_transmissions(self.family)

        return transmissions
    

    def get_collision_matrix(self, transmissions):

        collision_matrix = np.zeros((self.numOCW, self.numOBW, self.simTime))
        
        tx : LoRaTransmission
        for tx in transmissions:

            for t, obw in enumerate(tx.sequence):

                timeSlot = tx.startSlot + (t * self.granularity)
                for g in range(self.granularity):

                    collision_matrix[tx.ocw][obw + tx.grid][timeSlot + g] += 1

        return collision_matrix
    

    def get_events(self, collision_matrix, transmissions):

        start_events = self.get_start_events(transmissions)
        collision_events = self.get_collision_events(collision_matrix)
        end_events = self.get_end_events(transmissions)

        start_times = [ev._time for ev in start_events]
        collision_times = [ev._time for ev in collision_events]
        end_times = [ev._time for ev in end_events]

        events = start_events + collision_events + end_events
        event_times = start_times + collision_times + end_times

        sorted_events = [event for _, event in sorted(zip(event_times, events))]

        return sorted_events
        

    def get_start_events(self, transmissions):

        start_events = []
        tx : LoRaTransmission
        for tx in transmissions:
            new_start_event = StartEvent(tx.startSlot, 'start', tx)
            start_events.append(new_start_event)

        return start_events
    

    def get_end_events(self, transmissions):

        end_events = []
        tx : LoRaTransmission
        for tx in transmissions:
            tx_end = tx.startSlot + (tx.numFragments * self.granularity) - 1
            new_end_event = EndEvent(tx_end, 'end', tx)
            end_events.append(new_end_event)

        return end_events


    def get_collision_events(self, collision_matrix):

        collision_events = []
        for t in range(self.simTime):

            sub_collision_matrix = collision_matrix[:][:][t]
            indeces = np.where(sub_collision_matrix > 1)

            for ocw, obw in zip(indeces[0], indeces[1]):
                new_collision_event = CollisionEvent(t, 'collision', ocw, obw)
                collision_events.append(new_collision_event)

        return collision_events


    def run(self):

        transmissions = self.get_transmissions()
        collision_matrix = self.get_collision_matrix(transmissions)
        events = self.get_events(collision_matrix, transmissions)

        self.gateway.run(events)
    

    def get_decoded_transmissions(self):
        return self.gateway.get_decoded_transmissions()
    