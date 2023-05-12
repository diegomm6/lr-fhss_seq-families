import numpy as np
import hashlib

# obw to transmit fragment k for node x following the hash method explained in [1]
# [1] Boquet, G., Tuset-Peiró, P., Adelantado, F., Watteyne, T., & Vilajosana, X. (2021).
# LR-FHSS: Overview and performance analysis. IEEE Communications Magazine, 59(3), 30-36.
def get_obw(x, k):

    i = x + k * 2**16
    i = int.to_bytes(i, 4, 'little')

    h = int.from_bytes(hashlib.sha256(i).digest()[:4], 'little')

    return ( h % 35 ) * 8


# q length sequence of owb's for node x
def get_hashFHS(x, q):
    fhs = []
    for k in range(q):
        fhs.append(get_obw(x, k))
        
    return fhs

class HashFamily():

    def __init__(self, q) -> None:
        self.q = q

    # obatin a family of m sequences of length q
    def get_family(self, m):
        fam = []
        for x in range(m):
            fam.append(get_hashFHS(x, self.q))

        return np.array(fam)
