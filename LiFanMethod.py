import numpy as np
from base import gcd, get_min_gap, filter_freq, split_seq

"""
Generates a wide gap sequence with minimum gap e and optimal maximal hamming autocorrelation
let l, d s.t. 1 < d < l/2
& gcd(l, d) = gcd(l, d+1) = 1

WGFHS with parameters (2*l, l, 2, d-1) = (length, alphabet_size, autoHC, min_gap)
"""
def get_2l_sequence(l, d):
    s = []
    t = []

    for i in range(l):
        s.append( (i*d) % l )
        t.append( (i*(d+1) + 1) % l )

    return np.array(s + t)


"""
Generates a wide gap sequence with minimum gap e and optimal maximal hamming autocorrelation
let l, d s.t. 1 < d < (l-1)/2
& gcd(l, d) = gcd(l, d+1) = gcd(l, d+2) = 1

WGFHS with parameters (3*l, l, 3, d-1) = (length, alphabet_size, autoHC, min_gap)
"""
def get_3l_sequence(l, d):
    s = []
    t = []
    u = []

    for i in range(l):
        s.append( (i*d) % l )
        t.append( (i*(d+1) + 1) % l )
        u.append( (i*(d+2) + 2) % l )

    return np.array(s + t + u)


"""
The method proposed in [2] produces a long wide gap sequence
which is split into smaller sequences to generate an optimal family
of sequences of size q

[2] Li, P., Fan, C., Mesnager, S., Yang, Y., & Zhou, Z. (2021). Constructions of optimal
uniform wide-gap frequency-hopping sequences. IEEE Transactions on Information Theory, 68(1), 692-700.
"""
class LiFanFamily():

    def __init__(self, q, maxfreq, mingap) -> None:
        self.q = q
        self.mingap = mingap
        self.maxfreq = maxfreq


    def get_2l_family(self, l, d):

        crit1 = 1 < d and d < l/2
        crit2 = gcd(l, d) == 1 and gcd(l, d+1) == 1
        assert crit1 and crit2, "criteria for 2l sequence not met"

        seq_2l = get_2l_sequence(l, d)

        if self.maxfreq < l:
            seq_2l = filter_freq(seq_2l, self.maxfreq, self.mingap)

        return split_seq(seq_2l, self.q)
    

    def get_3l_family(self, l, d):

        crit1 = 1 < d < (l-1)/2
        crit2 = gcd(l, d) == 1 and gcd(l, d+1) == 1 and gcd(l, d+2) == 1
        assert crit1 and crit2, "criteria for 3l sequence not met"

        seq_3l = get_3l_sequence(l, d)

        if self.maxfreq < l:
            seq_3l = filter_freq(seq_3l, self.maxfreq, self.mingap)

        return split_seq(seq_3l, self.q)



