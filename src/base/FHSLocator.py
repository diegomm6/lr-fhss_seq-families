import numpy as np
from multiprocessing import Pool

class FHSLocator():

    def __init__(self, simTime: int, numHeaders: int, timeGranularity: int, freqGranularity: int,
                 freqPerSlot: float, hdrTime: float, frgTime: float, headerSlots: int,
                 max_packet_duration: int, maxFreqShift: float) -> None:
        
        self.simTime = simTime
        self.numHeaders = numHeaders
        self.headerSlots = headerSlots
        self.timeGranularity = timeGranularity
        self.freqGranularity = freqGranularity
        self.max_packet_duration = max_packet_duration

        self.headerSize = headerSlots * freqGranularity       # time-freq hdr size
        self.fragmentSize = timeGranularity * freqGranularity # time-freq frg size

        self.min_seqlength = 11 # CHANGE HERE FOR DIFFERENT CR
        maxDopplerRate = 300    # Hz/s

        self.maxDopplerSlots = round(22000 / freqPerSlot)
        self.baseFreq = round(maxFreqShift / freqPerSlot)

        self.maxHdrShift = int(np.ceil(maxDopplerRate * hdrTime / freqPerSlot))
        self.maxFrgShift = int(np.ceil(maxDopplerRate * frgTime / freqPerSlot))

        self.receivedMatrix = np.zeros(1)

    
    def set_RXmatrix(self, RXMatrix: np.ndarray):
        self.receivedMatrix = RXMatrix


    def fits(self, subm: np.ndarray, isHeader: bool) -> bool:
        """
        Determines is a full header/fragment is present in the given search window
        """

        if isHeader:
            return (subm == 1).sum() >= self.headerSize

        else:
            return (subm == 1).sum() >= self.fragmentSize
    

    def create_Tp_parallel(self, seqs):

        _input = []
        subseq = int(len(seqs) / 16)
        i = 0
        k = 0
        while i < len(seqs):
            _input.append([seqs[i:i+subseq], k*subseq])
            i += subseq
            k += 1

        pool = Pool(processes = 16)
        result = pool.map(self.create_Tp, _input)
        pool.close()
        pool.join()

        Tp = []
        for tp in result:
            Tp += tp

        return Tp
    

    def create_Tp(self, input):
        """
        Exhaustive FHS locator method
        """

        seqs, shift = input
        
        Tp = []
        for t in range(self.simTime - self.max_packet_duration):
            for s, seq in enumerate(seqs):

                possibleShift = []
                for fs in range(-self.maxDopplerSlots, self.maxDopplerSlots, 1):

                    fits, fitness = self.isPossibeShift(fs, t, seq)
                    if fits:
                        possibleShift.append(fs)
                        break # select first possible shift fs that fits seq s at time t

                if len(possibleShift) > 0:
                    Tp.append((t, s+shift)) #fitness, possibleShift[0]
        
        return Tp
    

    def isPossibeShift(self, staticShift, startTime, seq):
        """
        Determines if a given sequence can fit in the received OCW channel
        for a given static doppler shift at the beginnig of the transmission 
        """

        dynamicShift = 0
        fitness = 0
        time = startTime
        for fh, obw in enumerate(seq):

            startFreq = self.baseFreq + obw * self.freqGranularity + staticShift
            endFreq = startFreq + self.freqGranularity 

            # header
            if fh < self.numHeaders:

                endTime = time + self.headerSlots
                header = self.receivedMatrix[(startFreq - dynamicShift) : endFreq, time : endTime]

                if self.fits(header, True):
                    fitness += 1
                    time = endTime
                    dynamicShift += self.maxHdrShift
                else:
                    return False, 0
        
            # fragment
            else:

                endTime = time + self.timeGranularity
                fragment = self.receivedMatrix[(startFreq - dynamicShift) : endFreq, time : endTime]

                if self.fits(fragment, False):
                    fitness += 1
                    time = endTime
                    dynamicShift += self.maxFrgShift
                else:
                    if fitness >= self.min_seqlength:
                        return True, fitness

                    return False, 0
                
        return True, fitness
    

    def print_metrics(self, Tt, Tp, solve_time):

        tp = 0 # (t,s,l) in T     and  in T'
        fp = 0 # (t,s,l) not in T but  in T'
        fn = 0 # (t,s,l) in T     but  not in T'

        fplist = []
        #for t in Tt:
        #    print('True seq:', t)
        
        for t in Tt:
            if t in Tp:
                tp += 1
                #print('TP:', t)
            else:
                fn += 1
                #print('FN:', t)
        for t in Tp:
            time,s = t # time,s,l,ds = t
            if t not in Tt:
                fp += 1
                fplist.append(list(t))
                #print('FP:', t)

        if len(fplist):
            fplist = np.array(fplist)
            fplist = fplist[fplist[:, 0].argsort()]
            #[print('FP:', tuple(t)) for t in fplist]


        header = 'TP,FP,FN,len(T),len(T\'),time[s]\n'
        string = f'{tp},{fp},{fn},{len(Tt)},{len(Tp)},{solve_time:.2f}'
        #print(header+string)

        return tp, fp, fn, 0 #self.metric_processing2(Tt, Tp)
    

    def metric_processing(self, Tt, Tp):

        Tt_nolength = [[t,s] for t,s,l in Tt]
        Tp_nolength = [[t,s] for t,s,l in Tp]
        
        Tt_lengths = [l for t,s,l in Tt]
        Tp_lengths = [l for t,s,l in Tp]

        diff1 = 0
        i=0
        for tt in Tt_nolength:
            if tt in Tp_nolength:
                _id = Tp_nolength.index(tt)
                if Tt_lengths[i] == Tp_lengths[_id]-1:
                    diff1 += 1
            i+=1

        return diff1
    
    def metric_processing2(self, Tt, Tp):

        Tt_nolength = [[t,s] for t,s,l in Tt]
        Tp_nolength = [[t,s] for t,s,l in Tp]
        
        lengthmismatch = 0
        for tt in Tt_nolength:
            if tt in Tp_nolength:
                lengthmismatch += 1

        return lengthmismatch
