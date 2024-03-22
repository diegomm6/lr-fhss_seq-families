import time
import galois
import random
import numpy as np
from src.base.base import *
from src.base.LoRaNode import LoRaNode
from src.base.LoRaGateway import LoRaGateway
from src.base.LoRaTransmission import LoRaTransmission
from src.families.LiFanMethod import LiFanFamily
from src.families.LR_FHSS_DriverMethod import LR_FHSS_DriverFamily
from src.families.LempelGreenbergMethod import LempelGreenbergFamily
from src.base.FHSLocator import FHSLocator


class LoRaNetwork():

    def __init__(self, numNodes, familyname, numOCW, numOBW, numGrids, CR, timeGranularity, freqGranularity,
                 simTime, numDecoders, use_earlydecode, use_earlydrop, use_headerdrop, collision_method) -> None:
        
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

        self.headerSlots = round(timeGranularity * HDR_TIME / FRG_TIME)
        self.freqPerSlot = OBW_BW / self.freqGranularity
        self.frequencySlots = int(round(OCW_RX_BW / self.freqPerSlot))

        OCWchannelTXBW = numOBW * OBW_BW  # OCW transmitter bandwidth in Hz
        self.maxFreqShift = (OCW_RX_BW - OCWchannelTXBW) / 2
        self.baseFreq = round(self.maxFreqShift / self.freqPerSlot) # freq offset to center TX window over RX window

        assert CR==1 or CR==2, "Only CR 1/3 and CR 2/3 supported"
        self.numHeaders = 3 # CR == 1
        if CR == 2:
            self.numHeaders = 2
        
        self.header = np.ones((freqGranularity, self.headerSlots))  # header block
        self.fragment = np.ones((freqGranularity, timeGranularity)) # fragment block


        ###########################
        # INTIALIZE NETWORK MODULES
        ###########################

        max_packet_duration = MAX_HDRS * self.headerSlots + MAX_FRGS * timeGranularity
        startLimit = simTime - max_packet_duration
        self.nodes = [LoRaNode(i, CR, numOCW, startLimit) for i in range(numNodes)]

        self.TXset = self.set_transmissions()

        # add support for multiple gateways in the future
        self.gateway = LoRaGateway(CR, timeGranularity, freqGranularity, use_earlydrop, use_earlydecode,
                                   use_headerdrop, numDecoders, self.baseFreq, collision_method)
        
        self.fhsLocator = FHSLocator(self.simTime, self.numHeaders, self.timeGranularity, self.freqGranularity,
                                     self.freqPerSlot, self.headerSlots, max_packet_duration, self.baseFreq)


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
        

    def set_transmissions(self) -> list[LoRaTransmission]:

        transmissions = []
        node : LoRaNode
        for node in self.nodes:
            transmissions += node.get_transmissions(self.FHSfam)

        sorted_txs = sorted(transmissions)
        #for tx in sorted_txs: print(tx)

        return sorted_txs
    

    def run(self, power: bool, dynamic: bool) -> None:
        collision_matrix = self.get_rcvM(self.TXset, power, dynamic)
        self.gateway.run(self.TXset, collision_matrix, dynamic)


    def restart(self) -> None:

        self.gateway.restart()
        self.TXset = self.set_transmissions()

        node : LoRaNode
        for node in self.nodes:
            node.restart()


    ####################################
    # TIME-FREQ MATRIX GENERATOR METHODS
    ####################################

    def get_rcvM(self, transmissions: list[LoRaTransmission], power: bool, dynamic: bool) -> np.ndarray:
        """
        Create received matrix from given transmissions set in 4 ways.

        The dynamic flag indicates to use dynamic doppler, variable doppler shift PER header/fragment.
        If dynamic flag is off (static doppler), same doppler shift FOR ALL header/fragment.

        The power flag on will output a matrix based on received power.
        When the power flag is off, the output matrix is based on counts.
        """

        # count based received matrix
        RXpower = 1
        rcvM = np.zeros((self.numOCW, self.frequencySlots, self.simTime))

        # power based received matrix
        if power: 
            rcvM = np.random.rayleigh(1, (self.numOCW, self.frequencySlots, self.simTime))
            rcvM = (rcvM / np.linalg.norm(rcvM)) * np.sqrt(dBm2mW(AWGN_VAR_DB))

        # add transmissions to received matrix
        for tx in transmissions:

            dopplershift = round(tx.dopplerShift[0] / self.freqPerSlot)

            time = tx.startSlot
            for fh, obw in enumerate(tx.sequence):

                # variable doppler shift per header / fragment
                if dynamic:
                    dopplershift = round(tx.dopplerShift[fh] / self.freqPerSlot)

                startFreq = self.baseFreq + obw * self.freqGranularity + dopplershift
                endFreq = startFreq + self.freqGranularity

                # power based received matrix
                if power:
                    carrier = OCW_FC + startFreq * self.freqPerSlot
                    RXpower = dBm2mW(GAIN_TX) * dBm2mW(GAIN_RX) * dBm2mW(tx.power) \
                            * get_FS_pathloss(tx.distance, carrier)

                # write header
                if fh < tx.numHeaders:
                    endTime = time + self.headerSlots
                    rcvM[tx.ocw, startFreq : endFreq, time : endTime] += (self.header * RXpower)

                # write fragment
                else:
                    endTime = time + self.timeGranularity
                    rcvM[tx.ocw, startFreq : endFreq, time : endTime] += (self.fragment * RXpower)
                
                time = endTime

        return rcvM
    

    def get_OCWchannel_occupancy(self) -> float:

        transmissions = self.TXset
        count_dynamic_rcvM = self.get_rcvM(transmissions, power=False, dynamic=True)[0]
        count_dynamic_rcvM[count_dynamic_rcvM > 1] = 1

        fslots, tslots = count_dynamic_rcvM.shape

        filered_recvM = count_dynamic_rcvM[:, int(tslots/2 - tslots/4) : int(tslots/2 + tslots/4)]

        perFreqSlot_occ = np.mean(filered_recvM, axis=1)

        # ignore unused channels
        unused_ch = len(np.where(perFreqSlot_occ == 0)[0])

        return np.sum(perFreqSlot_occ) / (fslots - unused_ch)


    ##################################
    # FHS SEARCH FOR HEADERLESS DECODE
    ##################################

    def get_predecoded_data(self):

        # get TXset and rcvd matrix
        transmissions = self.TXset
        count_dynamic_rcvM = self.get_rcvM(transmissions, power=False, dynamic=True)

        # predecode headers
        self.gateway.predecode(transmissions, count_dynamic_rcvM, dynamic=True)
        decoded_headers = self.gateway.get_decoded_headers()
        decoded_m = self.get_rcvM(decoded_headers, power=False, dynamic=True)

        # 3-value - noise (0), signal (1), interference (2)
        value3_matrix = count_dynamic_rcvM.copy()[0]
        value3_matrix[value3_matrix > 2] = 2

        # interference, noise/signal (0), interference (2)
        interference = count_dynamic_rcvM.copy()[0]
        interference[interference > 2] = 2
        interference[interference < 2] = 0

        # detected/received difference matrix
        diff = np.subtract(value3_matrix, decoded_m[0])
        diff = np.add(diff, interference)
        diff[diff > 1] = 1

        collided_TXset = self.get_collided_TXset()

        return collided_TXset, diff
    

    def get_collided_TXset(self) -> list[LoRaTransmission]:

        TXsetids = [tx.id for tx in self.TXset]
        decoded_headers = self.gateway.get_decoded_headers()
        decoded_headers_ids = [tx.id for tx in decoded_headers]

        collidedTXset = []
        for i, txid in enumerate(TXsetids):
            if txid not in decoded_headers_ids:
                collidedTXset.append(self.TXset[i])
        
        return collidedTXset

            
    def exhaustive_search(self, transmissions: list[LoRaTransmission], rcvM: np.ndarray):

        # create tx list in the form (time, seqid, seqlength)
        trueTXs = []
        for tx in transmissions:
            ds = round(tx.dopplerShift[0] / self.freqPerSlot)
            trueTXs.append((tx.startSlot, tx.seqid, len(tx.sequence))) # , len(tx.sequence), ds
        
        self.fhsLocator.set_RXmatrix(rcvM)

        start = time.time()
        estTXs = self.fhsLocator.get_estTXs_parallel(self.FHSfam.FHSfam)
        solve_time = time.time()-start

        # self.printknapSack(self.numNodes, Tp, RXbinary_matrix)
        tp, fp, fn, lenmatch, minlenerr = self.fhsLocator.get_metrics2(trueTXs, estTXs)

        return tp, fp, fn, solve_time, lenmatch, minlenerr
    

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
