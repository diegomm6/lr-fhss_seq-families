import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


class MILPsolver():

    def __init__(self, simTime: int, numHeaders: int, timeGranularity: int, freqGranularity: int) -> None:
        self.simTime = simTime
        self.numHeaders = numHeaders
        self.timeGranularity = timeGranularity
        self.freqGranularity = freqGranularity
        self.header_slots = round(timeGranularity * 233.472 / 102.4)
        self.max_packet_duration = 31 * timeGranularity + 3 * self.header_slots
        
        maxShift = (200000 - 137000) / 2
        freqPerSlot= 488.28125 / freqGranularity
        self.maxDopplerSlots = round(20000 / freqPerSlot)
        self.baseFreq = round(maxShift / freqPerSlot)

        self.min_seqlength = 11


    def fits(self, subm: np.ndarray) -> bool:
        return (subm == 1).all()
        

    def isPossibeShift(self, m, fs, t, seq):

        count = 0
        time = t
        for fh, obw in enumerate(seq):

            startFreq = self.baseFreq + fs + obw * self.freqGranularity
            endFreq = startFreq + self.freqGranularity

            # header
            if fh < self.numHeaders:
                endTime = time + self.header_slots
                header = m[time : endTime, startFreq : endFreq]

                if self.fits(header):
                    count += np.sum(header)
                    time = endTime
                else:
                    return False, 0
        
            # fragment
            else:
                endTime = time + self.timeGranularity
                fragment = m[time : endTime, startFreq : endFreq]

                if self.fits(fragment):
                    count += np.sum(fragment)
                    time = endTime
                else:
                    return False, 0
                
        return True, count
    


    def isPossibeShift_variable_length(self, m, fs, t, seq):

        fitness = 0
        time = t
        for fh, obw in enumerate(seq):

            startFreq = self.baseFreq + fs + obw * self.freqGranularity
            endFreq = startFreq + self.freqGranularity

            # header
            if fh < self.numHeaders:
                endTime = time + self.header_slots
                header = m[time : endTime, startFreq : endFreq]

                if self.fits(header):
                    fitness += 1
                    time = endTime
                else:
                    return False, 0
        
            # fragment
            else:
                endTime = time + self.timeGranularity
                fragment = m[time : endTime, startFreq : endFreq]

                if self.fits(fragment):
                    fitness += 1
                    time = endTime
                else:
                    if fitness >= self.min_seqlength:
                        return True, fitness

                    return False, 0
                
        return True, fitness


    def create_Tp_variable_length(self, m, seqs):
        
        Tp = []
        for t in range(self.simTime - self.max_packet_duration):
            for s, seq in enumerate(seqs):

                possibleShift = []
                for fs in range(-self.maxDopplerSlots, self.maxDopplerSlots, 1):

                    fits, fitness = self.isPossibeShift_variable_length(m, fs, t, seq)
                    if fits:
                        possibleShift.append(fs)
                        break # select first possible shift fs that fits seq s at time t

                if len(possibleShift) > 0:
                    Tp.append((t, s))
                    #Tp.append((t, s, fitness))
        
        return Tp
    

    def create_T(self, m, seqs):
        
        # create T_[t,s] matrix
        Tvars_m = np.zeros((self.simTime - self.max_packet_duration, len(seqs)))
        slots = np.zeros((self.simTime - self.max_packet_duration, len(seqs)))
        shifts = np.zeros((self.simTime - self.max_packet_duration, len(seqs)))

        for t in range(self.simTime - self.max_packet_duration):
            for s, seq in enumerate(seqs):

                possibleShift = []
                for fs in range(-self.maxDopplerSlots, self.maxDopplerSlots, 1):

                    fits, count = self.isPossibeShift(m, fs, t, seq)
                    if fits:
                        slots[t][s] = count
                        shifts[t][s] = fs
                        possibleShift.append(fs)
                        break # select first possible shift fs that fits seq s at time t

                if len(possibleShift) > 0:
                    #print(f"time = {t}    seq = {s}    shift = {possibleShift[0]}")
                    Tvars_m[t][s] = 1
        
        return Tvars_m, slots, shifts


    def create_Tp(self, m, seqs):
        
        Tp = []
        for t in range(self.simTime - self.max_packet_duration):
            for s, seq in enumerate(seqs):

                possibleShift = []
                for fs in range(-self.maxDopplerSlots, self.maxDopplerSlots, 1):

                    fits, count = self.isPossibeShift(m, fs, t, seq)
                    if fits:
                        possibleShift.append(fs)
                        break # select first possible shift fs that fits seq s at time t

                if len(possibleShift) > 0:
                    Tp.append((t, s, len(seq)))
        
        return Tp
    

    # filter sequence length by assuming only the longest sequence that fits is possible
    def filter_T(self, Tvars_m, num_seq):
        # filter longest sequence that fits T
        seq_length_range = 23  # max length(seqs) - min length(seqs) +1 over "y" axis
        for t in range(self.simTime - self.max_packet_duration):

            i = 0
            while i < num_seq:
                sub_Tvars_m = Tvars_m[t][i : i + seq_length_range]

                j = 0
                while j < seq_length_range and sub_Tvars_m[j] == 1:
                    Tvars_m[t][i + j] = 0
                    j += 1

                if j:
                    Tvars_m[t][i + j - 1] = 1
                    #Tvars_m[t][i + j - 2] = 1 # if j>1, include also maxlength -1

                i += seq_length_range


    def solve_by_milp(self, m, seqs):

        timeSlots, frequencySlots = m.shape

        # output
        T = []

        # prevent gurobi output
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()

        # solve milp model
        try:
            # create model
            model = gp.Model("milp1", env=env)

            # variable: create M_{t,c}
            Mvars = {}
            for t in range(self.simTime):
                for c in range(frequencySlots):
                    if m[t][c] > 0:
                        Mvars[t, c] = model.addVar(vtype=GRB.BINARY, name="M.{}.{}".format(t, c))

            # create T_[t,s] matrix
            Tvars_m, slots, shifts = self.create_T(m, seqs)

            # filter by sequence length
            #Tvars_m = self.filter_T(Tvars_m, len(seqs))

            # variable: create T_{t,s}
            Tvars = {}
            indeces = np.argwhere(Tvars_m > 0)
            print(len(indeces))
            for i, id in enumerate(indeces):
                t = id[0]
                s = id[1]
                #print(f"{i} t= {t}, s= {s//23}")
                #print(f"time = {t}    seq = {s}    shift = {shifts[t][s]}")
                Tvars[t, s] = model.addVar(vtype=GRB.BINARY, name="T.{}.{}".format(t, s))

            # constraint: FRG * T_{t,s} = Sum M_{t,c} corresponding to t,s
            for key in Tvars:
                t = key[0]
                s = key[1]

                s_len = self.header_slots * self.numHeaders + self.timeGranularity * (len(seqs[s]) - self.numHeaders)
                expr1 = s_len * Tvars[t, s]
                expr2 = slots[t][s]

                model.addConstr(expr1 == expr2, "cT.{}.{}".format(t, s))

            # set objective
            model.setObjective(quicksum(list(Tvars.values())), GRB.MINIMIZE)

            # optimize model
            model.optimize()

            # print results
            for v in model.getVars():
                #print("{} {}".format(v.VarName, v.X))
                var_list = v.VarName.split('.')
                if var_list[0] == 'T':
                    t = int(var_list[1])
                    s = int(var_list[2])
                    if v.X == 1:
                        T.append((t, s, len(seqs[s])))
                        # print('added {}'.format(T[-1]))

        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))

        #except AttributeError:
        #    print('Encountered an attribute error')

        return T


    def print_metrics(self, Tt, Tp, solve_time):

        tp = 0 # (s, t, p) in T     and  in T'
        fp = 0 # (s, t, p) not in T but  in T'
        fn = 0 # (s, t, p) in T     but  not in T'
        
        for t in Tt:
            if t in Tp:
                tp += 1
                # print('TP:', t)
            else:
                fn += 1
                # print('FN:', t)
        for t in Tp:
            if t not in Tt:
                fp += 1
                # print('FP:', t)

        Tt_set = set(Tt)
        Tp_set = set(Tp)

        header = 'TP,FP,FN,len(T),len(T\'),dup(T),dup(T\'),time[s]\n'
        string = '{},{},{},{},{},{},{},{:.2f}'.format(tp, fp, fn, len(Tt), len(Tp), len(Tt) - len(Tt_set), len(Tp) - len(Tp_set), solve_time)
        print(header+string)

        return tp, fp, fn
    