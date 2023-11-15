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

    def __init__(self, numNodes, familyname, numOCW, numOBW, numGrids, CR, timeGranularity, freqGranularity,
                 simTime, numDecoders, use_earlydecode, use_earlydrop, use_headerdrop) -> None:
        
        self.numOCW = numOCW
        self.numOBW = numOBW
        self.timeGranularity = timeGranularity
        self.freqGranularity = freqGranularity
        self.header_slots = round(timeGranularity * 233.472 / 102.4)
        self.simTime = simTime
        self.use_earlydecode = use_earlydecode
        self.FHSfam = self.set_FHSfamily(familyname, numGrids)

        assert CR==1 or CR==2, "Only CR 1/3 and CR 2/3 supported"

        max_packet_length_in_slots = 31 * timeGranularity + 3 * self.header_slots
        startLimit = simTime - max_packet_length_in_slots
        self.nodes = [LoRaNode(i, CR, numOCW, startLimit) for i in range(numNodes)]

        # add support for multiple gateways in the future
        self.gateway = LoRaGateway(CR, timeGranularity, freqGranularity, use_earlydrop, 
                                   use_earlydecode, use_headerdrop, numDecoders)
        
        self.header = np.ones((freqGranularity, self.header_slots))
        self.fragment = np.ones((freqGranularity, timeGranularity))

    
    def set_FHSfamily(self, familyname, numGrids):

        if familyname == "lemgreen":
            polys = galois.primitive_polys(2, 5)
            poly1 = next(polys)
            lempelGreenbergFHSfam = LempelGreenbergFamily(p=2, n=5, k=5, poly=poly1)
            lempelGreenbergFHSfam.set_family(numGrids)
            return lempelGreenbergFHSfam

        elif familyname == "driver":
            driverFHSfam = LR_FHSS_DriverFamily(q=34, regionDR="EU137")
            return driverFHSfam

        elif familyname == "lifan":
            liFanFHSfam = LiFanFamily(q=34, maxfreq=280, mingap=8)
            liFanFHSfam.set_family(281, 8, '2l')
            return liFanFHSfam

        else:
            raise Exception(f"Invalid family name '{familyname}'")
        

    def get_transmissions(self) -> list[LoRaTransmission]:

        transmissions = []
        node : LoRaNode
        for node in self.nodes:
            transmissions += node.get_transmissions(self.FHSfam)

        sorted_txs = sorted(transmissions)
        #for tx in sorted_txs: print(tx)

        return sorted_txs
    

    def get_collision_matrix(self, transmissions: list[LoRaTransmission]) -> np.ndarray:

        carrierOffset = 0
        maxDopplerShift = 20000
        maxShift = carrierOffset + maxDopplerShift

        freqPerSlot = 488.28125 / self.freqGranularity
        frequencySlots = int(self.numOBW * self.freqGranularity + 2 * maxShift / freqPerSlot)

        collision_matrix = np.zeros((self.numOCW, frequencySlots, self.simTime))
        
        tx : LoRaTransmission
        for tx in transmissions:

            baseFreq = round((maxShift + tx.dopplerShift) / freqPerSlot)

            time = tx.startSlot
            for fh, obw in enumerate(tx.sequence):

                startFreq = baseFreq + obw * self.freqGranularity
                endFreq = startFreq + self.freqGranularity

                # write header
                if fh < tx.numHeaders:
                    endTime = time + self.header_slots
                    collision_matrix[tx.ocw, startFreq : endFreq, time : endTime] += self.header

                # write fragment
                else:
                    endTime = time + self.timeGranularity
                    collision_matrix[tx.ocw, startFreq : endFreq, time : endTime] += self.fragment
                
                time = endTime

        return collision_matrix
    

    def get_tracked_txs(self) -> int:
        return self.gateway.get_tracked_txs()

    def get_collided_payloads(self) -> int:
        return self.gateway.get_collided_payloads()

    def get_decoded_packets(self) -> int:
        return self.gateway.get_decoded_packets()

    def get_decoded_payloads(self) -> int:
        return self.gateway.get_decoded_payloads()
    
    def get_decoded_bytes(self) -> int:
        return self.gateway.get_decoded_bytes()
    
    def get_header_drop_packets(self) -> int:
        return self.gateway.get_header_drop_packets()
    

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

        self.gateway.run(transmissions, collision_matrix)

    
    def get_m(self):
        return self.get_collision_matrix(self.get_transmissions())
    

    def restart(self) -> None:

        self.gateway.restart()

        node : LoRaNode
        for node in self.nodes:
            node.restart()
    