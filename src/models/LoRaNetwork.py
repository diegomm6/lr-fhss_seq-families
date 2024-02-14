import time
import galois
import random
import numpy as np
from src.base.base import get_FS_pathloss, dBm2mW, mW2dBm
from src.base.Event import *
from src.base.LoRaNode import LoRaNode
from src.base.LoRaGateway import LoRaGateway
from src.base.LoRaTransmission import LoRaTransmission
from src.families.LiFanMethod import LiFanFamily
from src.families.LR_FHSS_DriverMethod import LR_FHSS_DriverFamily
from src.families.LempelGreenbergMethod import LempelGreenbergFamily
from src.base.FHSLocator import FHSLocator


class LoRaNetwork():

    def __init__(self, numNodes, familyname, numOCW, numOBW, numGrids, CR, timeGranularity, freqGranularity,
                 simTime, numDecoders, use_earlydecode, use_earlydrop, use_headerdrop) -> None:
        
        self.numOCW = numOCW
        self.numOBW = numOBW
        self.simTime = simTime
        self.numNodes = numNodes
        self.timeGranularity = timeGranularity # time slots per fragmet
        self.freqGranularity = freqGranularity # freq slots per OBW
        self.use_earlydecode = use_earlydecode
        self.FHSfam = self.set_FHSfamily(familyname, numGrids)

        ###########################
        # GLOBAL LR-FHSS PARAMETERS
        ###########################

        CFO = 0 # carrier frequency offset
        OBWchannelBW = 488.28125 # OBW bandwidth in Hz
        OCWchannelTXBW = 137000  # OCW transmitter bandwidth in Hz
        OCWchannelRXBW = 200000  # OCW receiver bandwidth in Hz
        headerTime = 0.233472    # seconds
        fragmentTime = 0.1024    # seconds

        self.OCWcarrier = 868100000 # OCW channel carrier freq

        self.headerSlots = round(timeGranularity * headerTime / fragmentTime)
        self.freqPerSlot = OBWchannelBW / self.freqGranularity
        self.frequencySlots = int(round(OCWchannelRXBW / self.freqPerSlot))

        assert CR==1 or CR==2, "Only CR 1/3 and CR 2/3 supported"
        self.numHeaders = 3 # CR == 1
        if CR == 2:
            self.numHeaders = 2
        
        self.header = np.ones((freqGranularity, self.headerSlots))  # header block
        self.fragment = np.ones((freqGranularity, timeGranularity)) # fragment block

        maxDopplerShift = (OCWchannelRXBW - OCWchannelTXBW) / 2
        self.maxFreqShift = CFO + maxDopplerShift
        self.baseFreq = round(self.maxFreqShift / self.freqPerSlot) # freq offset to center TX window over RX window

        self.TXgain_dB = 5
        self.RXgain_dB = 0
        self.AWGNvar_dB = -174 + 6 + 10 * np.log10(OBWchannelBW) # noise figure = 6 dB

        ###########################
        # INTIALIZE NETWORK MODULES
        ###########################

        max_packet_duration = 31 * timeGranularity + 3 * self.headerSlots
        startLimit = simTime - max_packet_duration
        self.nodes = [LoRaNode(i, CR, numOCW, startLimit) for i in range(numNodes)]

        # add support for multiple gateways in the future
        self.gateway = LoRaGateway(CR, timeGranularity, freqGranularity, use_earlydrop, 
                                   use_earlydecode, use_headerdrop, numDecoders)
        
        self.fhsLocator = FHSLocator(self.simTime, self.numHeaders, self.timeGranularity, self.freqGranularity,
                                     self.freqPerSlot, headerTime, fragmentTime, self.headerSlots,
                                     max_packet_duration, self.maxFreqShift)


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
    
    def run(self) -> None:
        transmissions = self.get_transmissions()
        #collision_matrix = self.get_staticdoppler_collision_matrix(transmissions)
        collision_matrix = self.get_power_collision_matrix(transmissions)
        self.gateway.run(transmissions, collision_matrix)

    def restart(self) -> None:

        self.gateway.restart()

        node : LoRaNode
        for node in self.nodes:
            node.restart()


    ####################################
    # TIME-FREQ MATRIX GENERATOR METHODS
    ####################################

    def get_staticdoppler_collision_matrix(self, transmissions: list[LoRaTransmission]) -> np.ndarray:
        """
        Create receiver matrix from given transmissions set
        Fixed Static doppler shift FOR ALL header/fragment is considered
        Received matrix is based on tranmission count
        """
        collision_matrix = np.zeros((self.numOCW, self.frequencySlots, self.simTime))
        
        tx : LoRaTransmission
        for tx in transmissions:

            staticShift = self.baseFreq + round(tx.dopplerShift[0] / self.freqPerSlot)

            time = tx.startSlot
            for fh, obw in enumerate(tx.sequence):

                startFreq = staticShift + obw * self.freqGranularity
                endFreq = startFreq + self.freqGranularity

                # write header
                if fh < tx.numHeaders:
                    endTime = time + self.headerSlots
                    collision_matrix[tx.ocw, startFreq : endFreq, time : endTime] += self.header

                # write fragment
                else:
                    endTime = time + self.timeGranularity
                    collision_matrix[tx.ocw, startFreq : endFreq, time : endTime] += self.fragment
                
                time = endTime

        return collision_matrix
    

    def get_dynamicdoppler_collision_matrix(self, transmissions: list[LoRaTransmission]) -> np.ndarray:
        """
        Create receiver matrix from given transmissions set
        Variable Static doppler shift PER header/fragment is considered
        Received matrix is based on tranmission count
        """

        collision_matrix = np.zeros((self.numOCW, self.frequencySlots, self.simTime))

        # static doppler per header / fragment
        for tx in transmissions:

            time = tx.startSlot
            for fh, obw in enumerate(tx.sequence):

                startFreq = self.baseFreq + obw * self.freqGranularity + round(tx.dopplerShift[fh] / self.freqPerSlot)
                endFreq = startFreq + self.freqGranularity

                # write header
                if fh < tx.numHeaders:
                    endTime = time + self.headerSlots
                    collision_matrix[tx.ocw, startFreq : endFreq, time : endTime] += self.header

                # write fragment
                else:
                    endTime = time + self.timeGranularity
                    collision_matrix[tx.ocw, startFreq : endFreq, time : endTime] += self.fragment
                
                time = endTime

        return collision_matrix
    

    def get_decoded_matrix(self, binary : bool) -> np.ndarray:
        """
        Create decode matrix from decoded transmissions set
        MUST be used after run(), otherwise there will be no decoded txs
        """

        decoded_matrix = np.zeros((self.numOCW, self.frequencySlots, self.simTime))

        decoded = self.gateway.get_decoded()
        tx : LoRaTransmission
        for tx, pld_status in decoded:

            staticShift = self.baseFreq + round(tx.dopplerShift[0] / self.freqPerSlot)

            time = tx.startSlot
            for fh, obw in enumerate(tx.sequence):

                startFreq = staticShift + obw * self.freqGranularity
                endFreq = startFreq + self.freqGranularity

                # write header
                if fh < tx.numHeaders:
                    endTime = time + self.headerSlots
                    decoded_matrix[tx.ocw, startFreq : endFreq, time : endTime] += self.header

                # write fragment
                else:
                    endTime = time + self.timeGranularity
                    decoded_matrix[tx.ocw, startFreq : endFreq, time : endTime] += self.fragment
                
                time = endTime
        
        if binary:
            decoded_matrix[decoded_matrix > 1] = 1

        return decoded_matrix
    

    def get_power_collision_matrix(self, transmissions: list[LoRaTransmission]) -> np.ndarray:
        """
        Create receiver matrix from given transmissions set
        Variable Static doppler shift PER header/fragment is considered
        Received matrix is based on received power
        """

        collision_matrix = np.random.rayleigh(1, (self.numOCW, self.frequencySlots, self.simTime)) * dBm2mW(self.AWGNvar_dB)

        # static doppler per header / fragment
        for tx in transmissions:

            time = tx.startSlot
            for fh, obw in enumerate(tx.sequence):

                startFreq = self.baseFreq + obw * self.freqGranularity + round(tx.dopplerShift[fh] / self.freqPerSlot)
                endFreq = startFreq + self.freqGranularity

                carrier = self.OCWcarrier + startFreq * self.freqPerSlot
                RXpower = dBm2mW(self.TXgain_dB) * dBm2mW(self.RXgain_dB) * dBm2mW(tx.power) * get_FS_pathloss(tx.distance, carrier)

                # write header
                if fh < tx.numHeaders:
                    endTime = time + self.headerSlots
                    collision_matrix[tx.ocw, startFreq : endFreq, time : endTime] += (self.header * RXpower)

                # write fragment
                else:
                    endTime = time + self.timeGranularity
                    collision_matrix[tx.ocw, startFreq : endFreq, time : endTime] += (self.fragment * RXpower)
                
                time = endTime

        #return collision_matrix
        return mW2dBm(collision_matrix)
    

    ##################################
    # FHS SEARCH FOR HEADERLESS DECODE
    ##################################
            
    def generate_Tt_M(self):

        # create tx list in the form (time, seqid, seqlength)
        transmissions = self.get_transmissions()
        Tt = []
        tx : LoRaTransmission
        for tx in transmissions:
            ds = round(tx.dopplerShift[0] / self.freqPerSlot)
            Tt.append((tx.startSlot, tx.seqid)) # , len(tx.sequence), ds
        
        # create binary received matrix
        collision_matrix = self.get_dynamicdoppler_collision_matrix(transmissions)
        #collision_matrix = self.get_staticdoppler_collision_matrix(transmissions)
        collision_matrix[collision_matrix > 1] = 1 # binary recv matrix

        return Tt, collision_matrix[0]
    

    def exhaustive_search(self):

        Tt, RXbinary_matrix = self.generate_Tt_M()
        self.fhsLocator.set_RXmatrix(RXbinary_matrix)

        start = time.time()
        Tp = self.fhsLocator.create_Tp_parallel(self.FHSfam.FHSfam)
        solve_time = time.time()-start

        # self.printknapSack(self.numNodes, Tp, RXbinary_matrix)
        tp, fp, fn = self.fhsLocator.get_metrics(Tt, Tp)

        return tp, fp, fn, solve_time
    

    ######################################
    # KNAPSACK TO FILTER FHS SEARCH RESULT
    ######################################

    #def estimateDynamicDoppler(self, ds):
    #    dsHz = ds * self.freqPerSlot
    
    def _add(self, tx, Mp):

        time = tx[0]
        seq = self.FHSfam.FHSfam[tx[1]][:tx[2]]
        ds = tx[3] #dopplerShift = self.estimateDynamicDoppler(tx[3])

        auxM = np.zeros(Mp.shape)
        for fh, obw in enumerate(seq):

            startFreq = self.baseFreq + obw * self.freqGranularity + ds
            endFreq = startFreq + self.freqGranularity

            # write header
            if fh < self.numHeaders:
                endTime = time + self.headerSlots
                auxM[startFreq : endFreq, time : endTime] += self.header

            # write fragment
            else:
                endTime = time + self.timeGranularity
                auxM[startFreq : endFreq, time : endTime] += self.fragment
            
            time = endTime

        boolauxM = np.array(auxM, dtype=bool)
        return np.logical_or(Mp, boolauxM)


    def get_ToverM_fitness(self, M, tx, Mp):
        newMp = self._add(tx, Mp)
        return (M == newMp).sum(), newMp


    def printknapSack(self, numTX, Tp, M):

        F,T = M.shape
        boolM = np.array(M, dtype=bool)
        numTp = len(Tp)

        K = [[0 for w in range(numTX + 1)] for i in range(numTp + 1)]

        matricesOld = np.zeros((numTX+1, F, T), dtype=bool) # i-1 matrices
        matricesNew = np.zeros((numTX+1, F, T), dtype=bool) # i matrices

        selected = [['-' for w in range(numTX + 1)] for i in range(numTp + 1)]
                
        # Build table K[][] in bottom up manner
        for i in range(numTp + 1):
            for w in range(numTX + 1):

                if i == 0 or w == 0:
                    K[i][w] = 0

                else:
                    fitnessWitem, newMp = self.get_ToverM_fitness(boolM, Tp[i-1], matricesOld[w-1])
                    
                    if fitnessWitem > K[i - 1][w]:
                        K[i][w] = fitnessWitem
                        matricesNew[w] = newMp
                        selected[i][w] = selected[i-1][w] + f"{i}-"

                    else:
                        K[i][w] = K[i - 1][w]
                        matricesNew[w] = matricesOld[w]
                        selected[i][w] = selected[i-1][w]
                        
            print(f'{i}:\tbestfit = {K[i][w]}\tselected = {selected[i][w]}\t\tTi = {Tp[i-1]}')
            matricesOld = matricesNew
            

        filteredTp_stringlist = selected[numTp][numTX][:-1].split("-")

        print(f'total = {F*T}\tfitness = {K[i][w]}')
        print(len(filteredTp_stringlist))
        #for Tp_id in filteredTp_stringlist:
            #print(Tp[int(Tp_id)])


    ########################
    # DATA GATHERING METHODS
    ########################
 
    def get_decoded_txs(self):
        """
        Returns start time and seq id for each decoded header, regardless of payload status
        MUST be used after run(), otherwise there will be no decoded txs
        """
        decoded = self.gateway.get_decoded()

        Tt_decoded = []
        tx : LoRaTransmission
        for tx, pld_status in decoded:
            Tt_decoded.append((tx.startSlot, tx.seqid))

        return Tt_decoded

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
