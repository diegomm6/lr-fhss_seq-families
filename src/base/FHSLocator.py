import numpy as np
from multiprocessing import Pool
from src.base.base import *


class FHSLocator():

    def __init__(self, simTime: int, numHeaders: int, timeGranularity: int, freqGranularity: int,
                 freqPerSlot: float, headerSlots: int, max_packet_duration: int, baseFreq: int) -> None:
        
        self.simTime = simTime
        self.freqPerSlot = freqPerSlot
        self.numHeaders = numHeaders
        self.headerSlots = headerSlots
        self.timeGranularity = timeGranularity
        self.freqGranularity = freqGranularity
        self.max_packet_duration = max_packet_duration
        self.baseFreq = baseFreq

        self.headerSize = headerSlots * freqGranularity       # time-freq hdr size
        self.fragmentSize = timeGranularity * freqGranularity # time-freq frg size

        self.receivedMatrix = np.zeros(1)
        self.min_seqlength = 11 # CHANGE HERE FOR DIFFERENT CR cr1=11

        maxtau = get_visibility_time(SAT_RANGE)
        self.T = np.linspace(-maxtau, maxtau, 100*simTime)
        self.DS = np.asarray([dopplerShift(t) for t in self.T])

        self.maxDopplerSlots = round(self.DS[-1] / freqPerSlot)
        self.DSperHdr = round(HDR_TIME / (self.T[1] - self.T[0]))
        self.DSperFrg = round(FRG_TIME / (self.T[1] - self.T[0]))

    
    def set_RXmatrix(self, RXMatrix: np.ndarray):
        self.receivedMatrix = RXMatrix


    def fits(self, subm: np.ndarray, isHeader: bool) -> bool:
        """
        Determines is a full header/fragment is present in the given search window
        """

        if isHeader: return (subm == 1).sum() >= self.headerSize
        else:        return (subm == 1).sum() >= self.fragmentSize
    

    def get_estTXs_parallel(self, seqs):

        _input = []
        subseq = int(len(seqs) / 16)
        i = 0
        k = 0
        while i < len(seqs):
            _input.append([seqs[i:i+subseq], k*subseq])
            i += subseq
            k += 1

        pool = Pool(processes = 16)
        result = pool.map(self.get_estTXs, _input)
        pool.close()
        pool.join()

        Tp = []
        for tp in result:
            Tp += tp

        return Tp
    

    def get_estTXs(self, input):
        """
        Exhaustive FHS locator method
        """

        seqs, shift = input
        
        estTXs = []
        for t in range(self.simTime - self.max_packet_duration):
            for s, seq in enumerate(seqs):

                possibleShift = []
                for DS in range(-self.maxDopplerSlots, self.maxDopplerSlots, 1):

                    fits, estLen = self.isPossibeShift(DS, t, seq)
                    if fits:
                        possibleShift.append(DS)
                        break # select first possible Doppler Shift that fits seq s at time t

                if len(possibleShift) > 0:
                    estTXs.append((t, s+shift, estLen)) #estLen, possibleShift[0]
        
        return estTXs


    def isPossibeShift(self, staticShift, startTime, seq):
        """
        Determines if a given sequence can fit in the received OCW channel
        for a given static doppler shift at the beginnig of the transmission 
        """

        estDSidx = bisection(self.DS, staticShift * self.freqPerSlot)

        fitness = 0
        time = startTime
        for fh, obw in enumerate(seq):

            startFreq = self.baseFreq + obw * self.freqGranularity + round(self.DS[estDSidx] / self.freqPerSlot) -1
            endFreq = startFreq + self.freqGranularity +1

            # header
            if fh < self.numHeaders:

                endTime = time + self.headerSlots
                header = self.receivedMatrix[startFreq : endFreq, time : endTime]

                if self.fits(header, True):
                    fitness += 1
                    time = endTime
                    estDSidx -= self.DSperHdr
                else:
                    return False, 0
        
            # fragment
            else:

                endTime = time + self.timeGranularity
                fragment = self.receivedMatrix[startFreq : endFreq, time : endTime]

                if self.fits(fragment, False):
                    fitness += 1
                    time = endTime
                    estDSidx -= self.DSperFrg
                else:
                    if fitness >= self.min_seqlength:
                        return True, fitness

                    return False, 0
                
        return True, fitness
    

    def get_metrics2(self, trueTXs, estTXs):

        tp = 0 # tx  in trueTXs &  in estTXs
        fp = 0 # tx !in trueTXs &  in estTXs
        fn = 0 # tx  in trueTXs & !in estTXs

        trueTXs_len = [l for t,s,l in trueTXs]
        estTXs_len = [l for t,s,l in estTXs]

        _trueTXs = [[t,s] for t,s,l in trueTXs]
        _estTXs = [[t,s] for t,s,l in estTXs]

        diff = []
        for i, tx in enumerate(_trueTXs):
            if tx in _estTXs:
                tp += 1
                diff.append(estTXs_len[_estTXs.index(tx)] - trueTXs_len[i])
                #print('TP:', t)
            else:
                fn += 1
                #print('FN:', t)

        fplist = []
        for tx in _estTXs:
            if tx not in _trueTXs:
                fp += 1
                fplist.append(list(tx))

        if len(fplist): # print fp txs in chronological order
            fplist = np.array(fplist)
            fplist = fplist[fplist[:, 0].argsort()]
            #[print('FP:', tuple(t)) for t in fplist]

        diff = np.array(diff)
        lendiff0 = (diff == 0).sum()
        lendiff1 = (diff == 1).sum()
        lendiff2 = (diff == 2).sum()
        lendiff3 = (diff > 2).sum()

        return tp, fp, fn, lendiff0, lendiff1, lendiff2, lendiff3
    
    def get_metrics(self, Tt, Tp):

        tp = 0 # (t,s,l) in T     and  in T'
        fp = 0 # (t,s,l) not in T but  in T'
        fn = 0 # (t,s,l) in T     but  not in T'

        for tx in Tt:
            if tx in Tp:
                tp += 1
                #print('TP:', t)
            else:
                fn += 1
                #print('FN:', t)

        fplist = []
        for tx in Tp:
            if tx not in Tt:
                fp += 1
                fplist.append(list(tx))

        if len(fplist): # print fp txs in chronological order
            fplist = np.array(fplist)
            fplist = fplist[fplist[:, 0].argsort()]
            #[print('FP:', tuple(t)) for t in fplist]

        return tp, fp, fn