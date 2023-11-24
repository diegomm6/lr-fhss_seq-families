import time
import galois
import random
import numpy as np
from src.base.Event import *
from src.base.LoRaNode import LoRaNode
from src.base.LoRaGateway import LoRaGateway
from src.base.LoRaTransmission import LoRaTransmission
from src.families.LiFanMethod import LiFanFamily
from src.families.LR_FHSS_DriverMethod import LR_FHSS_DriverFamily
from src.families.LempelGreenbergMethod import LempelGreenbergFamily
from src.base.milp import MILPsolver


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

        self.numHeaders = 3 # CR == 1
        if CR == 2:
            self.numHeaders = 2

        max_packet_duration = 31 * timeGranularity + 3 * self.header_slots
        startLimit = simTime - max_packet_duration
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
        maxDopplerShift = (200000 - 137000) / 2 # 20000
        maxShift = carrierOffset + maxDopplerShift

        freqPerSlot = 488.28125 / self.freqGranularity
        frequencySlots = int(self.numOBW * self.freqGranularity + 2 * maxShift / freqPerSlot)

        collision_matrix = np.zeros((self.numOCW, frequencySlots, self.simTime))
        
        tx : LoRaTransmission
        for tx in transmissions:

            baseFreq = round(maxShift / freqPerSlot) + round(tx.dopplerShift / freqPerSlot)

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
    

    def get_decoded_matrix(self, binary : bool) -> np.ndarray:

        carrierOffset = 0
        maxDopplerShift = (200000 - 137000) / 2 # 20000
        maxShift = carrierOffset + maxDopplerShift

        freqPerSlot = 488.28125 / self.freqGranularity
        frequencySlots = int(self.numOBW * self.freqGranularity + 2 * maxShift / freqPerSlot)

        decoded_matrix = np.zeros((self.numOCW, frequencySlots, self.simTime))

        decoded = self.gateway.get_decoded()
        tx : LoRaTransmission
        for tx, pld_status in decoded:

            baseFreq = round(maxShift / freqPerSlot) + round(tx.dopplerShift / freqPerSlot)

            time = tx.startSlot
            for fh, obw in enumerate(tx.sequence):

                startFreq = baseFreq + obw * self.freqGranularity
                endFreq = startFreq + self.freqGranularity

                # write header
                if fh < tx.numHeaders:
                    endTime = time + self.header_slots
                    decoded_matrix[tx.ocw, startFreq : endFreq, time : endTime] += self.header

                # write fragment
                else:
                    endTime = time + self.timeGranularity
                    decoded_matrix[tx.ocw, startFreq : endFreq, time : endTime] += self.fragment
                
                time = endTime
        
        if binary:
            decoded_matrix[decoded_matrix > 1] = 1

        return decoded_matrix
    

    def get_tracked_txs(self) -> int:
        return self.gateway.get_tracked_txs()
    
    def get_decoded_bytes(self) -> int:
        return self.gateway.get_decoded_bytes()
    
    def get_header_drop_packets(self) -> int:
        return self.gateway.get_header_drop_packets()

    def get_decoded_hrd_pld(self) -> int:
        return self.gateway.get_decoded_hrd_pld()

    def get_decoded_hdr(self) -> int:
        return self.gateway.get_decoded_hdr()

    def get_decodable_pld(self) -> int:
        return self.gateway.get_decodable_pld()
    
    def get_collided_hdr_pld(self) -> int:
        return self.gateway.get_collided_hdr_pld()
    

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
    

    def generate_extended_m(self):

        seq_length_range = 23  # max length(seqs) - min length(seqs) +1 over "y" axis

        extendedFamily = []
        for fhs in self.FHSfam.FHSfam:
            extendedFamily.append(fhs[:33])

        Tt = []
        transmissions = self.get_transmissions()
        tx : LoRaTransmission
        for tx in transmissions:
            t = tx.startSlot
            s = tx.seqid
            Tt.append((t, s))
            #Tt.append((t, s, len(tx.sequence)))
            #print(f"time = {t}    seq = {s}    shift = {tx.dopplerShift//70}")
        
        collision_matrix = self.get_collision_matrix(transmissions)
        collision_matrix[collision_matrix > 1] = 1

        return extendedFamily, Tt, collision_matrix[0].T
    

    def milp_solve(self):

        seqs, Tt, m = self.generate_extended_m()
        milpsolver = MILPsolver(self.simTime, self.numHeaders, self.timeGranularity, self.freqGranularity)

        start_time = time.process_time()
        #Tp = milpsolver.solve_by_milp(m, seqs)
        #Tp = milpsolver.create_Tp(m, seqs)
        Tp = milpsolver.create_Tp_variable_length(m, seqs)

        solve_time = time.process_time() - start_time
        return milpsolver.print_metrics(Tt, Tp, solve_time)
