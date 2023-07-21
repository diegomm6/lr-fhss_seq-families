import numpy as np
import galois
from src.base.Event import *
from src.base.LoRaNode import LoRaNode
from src.base.LoRaGateway import LoRaGateway
from src.base.LoRaTransmission import LoRaTransmission
from src.families.LiFanMethod import LiFanFamily
from src.families.LR_FHSS_DriverMethod import LR_FHSS_DriverFamily
from src.families.LempelGreenbergMethod import LempelGreenbergFamily


class LoRaNetwork():

    def __init__(self, numNodes, familyname, numOCW, numOBW, numGrids, CR, granularity,
                 simTime, numDecoders, use_earlydecode, use_earlydrop) -> None:
        
        self.numOCW = numOCW
        self.numOBW = numOBW
        self.granularity = granularity
        self.simTime = simTime
        self.use_earlydecode = use_earlydecode
        self.FHSfam = self.set_FHSfamily(familyname, numGrids)

        assert CR==1 or CR==2, "Only CR 1/3 and CR 2/3 supported"

        max_packet_length_in_slots = (31 * granularity) + (3 * int(granularity * 7 / 3))
        startLimit = simTime - max_packet_length_in_slots
        self.nodes = [LoRaNode(i, CR, numOCW, startLimit) for i in range(numNodes)]

        # add support for multiple gateways in the future
        self.gateway = LoRaGateway(granularity, CR, use_earlydrop, numDecoders)


    def get_transmissions(self) -> list[LoRaTransmission]:

        transmissions = []
        node : LoRaNode
        for node in self.nodes:
            transmissions += node.get_transmissions(self.FHSfam)

        return transmissions
    

    def set_FHSfamily(self, familyname, numGrids):

        if familyname == "lemgreen":
            polys = galois.primitive_polys(2, 5)
            poly1 = next(polys)
            lempelGreenbergFHSfam = LempelGreenbergFamily(p=2, n=5, k=5, poly=poly1)
            lempelGreenbergFHSfam.set_family(numGrids)
            return lempelGreenbergFHSfam

        elif familyname == "driver":
            driverFHSfam = LR_FHSS_DriverFamily(q=34)
            driverFHSfam.set_family(numGrids)
            return driverFHSfam

        elif familyname == "lifan":
            liFanFHSfam = LiFanFamily(q=34, maxfreq=280, mingap=8)
            liFanFHSfam.set_family(281, 8, '2l')
            return liFanFHSfam

        else:
            raise Exception(f"Invalid family name '{familyname}'")
    

    def get_collision_matrix(self, transmissions: list[LoRaTransmission]) -> np.ndarray:

        collision_matrix = np.zeros((self.numOCW, self.numOBW, self.simTime))
        
        tx : LoRaTransmission
        for tx in transmissions:

            time = tx.startSlot
            for fh, obw in enumerate(tx.sequence):

                used_timeslots = self.granularity # write fragment
                if fh < tx.header_replicas:       # write header
                    used_timeslots = int(self.granularity * 7 / 3)

                for g in range(used_timeslots):
                    collision_matrix[tx.ocw][obw][time + g] += 1

                time += used_timeslots

        return collision_matrix
    

    def get_events(self, collision_matrix: np.ndarray, transmissions: list[LoRaTransmission]) \
    -> list[EndEvent | CollisionEvent | StartEvent | EarlyDecodeEvent]:

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

            tx_end = tx.startSlot + (tx.header_replicas * int(self.granularity * 7 / 3)) + \
                (tx.numFragments * self.granularity) - 1
            
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


    def get_collided_payloads(self) -> int:
        return self.gateway.get_collided_payloads()

    def get_decoded_packets(self) -> int:
        return self.gateway.get_decoded_packets()

    def get_decoded_payloads(self) -> int:
        return self.gateway.get_decoded_payloads()
    
    def get_decoded_bytes(self) -> int:
        return self.gateway.get_decoded_bytes()
    

    def get_sent_packets(self) -> int:
        sent_packets = 0
        node: LoRaNode
        for node in self.nodes:
            sent_packets += node.sent_packets

        return sent_packets
    

    def get_sent_bytes(self) -> int:
        sent_bytes = 0
        node: LoRaNode
        for node in self.nodes:
            sent_bytes += node.sent_payload_bytes

        return sent_bytes
    

    def run(self) -> None:

        transmissions = self.get_transmissions()
        collision_matrix = self.get_collision_matrix(transmissions)
        events = self.get_events(collision_matrix, transmissions)

        self.gateway.run(events)
    

    def restart(self) -> None:

        self.gateway.restart()

        node : LoRaNode
        for node in self.nodes:
            node.restart()
    