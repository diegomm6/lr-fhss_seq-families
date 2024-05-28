import random
import numpy as np
from src.base.base import *
from src.base.LoRaTransmission import LoRaTransmission
from src.families.LR_FHSS_DriverMethod import LR_FHSS_DriverFamily


class DatasetGenerator():

    def __init__(self, CR, numOCW, numOBW, simTime, freqGranularity, timeGranularity) -> None:

        self.id = 0
        self.OCW = 0
        self.CR = CR
        self.numOCW = numOCW
        self.simTime = simTime
        self.TXpower_dB = TX_PWR_DB
        self.maxFrameT = MAX_FRM_TM
        self.freqGranularity = freqGranularity # freq slots per OBW
        self.timeGranularity = timeGranularity # time slots per fragmet

        driverFHSfam = LR_FHSS_DriverFamily(q=34, regionDR="EU137")
        self.FHSfam = driverFHSfam.FHSfam
        self.FHSfamsize = len(self.FHSfam)

        self.frequencySlots = numOBW
        self.headerSlots = round(timeGranularity * HDR_TIME / FRG_TIME)
        max_packet_duration = MAX_HDRS * self.headerSlots + MAX_FRGS * timeGranularity
        self.startLimit = simTime - max_packet_duration

        assert CR==1 or CR==2, "Only CR 1/3 and CR 2/3 supported"
        self.numHeaders = 3 # CR == 1
        if CR == 2:
            self.numHeaders = 2
        
        self.header = np.ones((freqGranularity, self.headerSlots))  # header block
        self.fragment = np.ones((freqGranularity, timeGranularity)) # fragment block
     

    def get_transmission(self, seq_id, numFragments) -> LoRaTransmission:

        if numFragments==0:
            numFragments = random.randrange(8, 32)

        payload_size = numFragments
        seq_length = int(numFragments + self.numHeaders)

        startSlot = random.randrange(0, self.startLimit)

        sequence = self.FHSfam[seq_id]
        sequence = sequence[:seq_length]

        dis2sat = random.uniform(SAT_H, SAT_RANGE)
        dynamicDoppler = [0] * (self.numHeaders + numFragments)

        tx = LoRaTransmission(self.id, self.id, startSlot, self.OCW, self.numHeaders,
                              payload_size, numFragments, sequence, seq_id, dis2sat,
                              dynamicDoppler, self.TXpower_dB)

        return tx
    

    def get_TXlist(self, numTX, numFragments) -> list[LoRaTransmission]:

        TXlist = []
        for i in range(numTX):
            seq_id = random.randrange(0, self.FHSfamsize)
            TXlist.append(self.get_transmission(seq_id, numFragments))

        return TXlist
    

    def get_rcvM(self, transmissions: list[LoRaTransmission]) -> np.ndarray:

        # count based received matrix
        RXpower = 1
        rcvM = np.zeros((self.numOCW, self.frequencySlots, self.simTime))

        # add transmissions to received matrix
        for tx in transmissions:

            time = tx.startSlot
            for fh, obw in enumerate(tx.sequence):

                startFreq = obw * self.freqGranularity 
                endFreq = startFreq + self.freqGranularity

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

