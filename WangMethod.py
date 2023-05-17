import numpy as np
import galois
from base import get_min_gap

"""
The method proposed in [1] produces afamily of OCWGFHS by mergining all the OCWGFHS
sets obtained for each primitive element in the filed GF(p), given p, w, and d

[1] Wang, T., Niu, X., Wang, J., & Shao, M. (2022, August). A New Class of Optimal 
Wide-Gap One-Coincidence Frequency-Hopping Sequence. In 2022 10th International Workshop
on Signal Design and Its Applications in Communications (IWSDA) (pp. 1-5). IEEE.
"""
class WangFamily():

    def __init__(self, p, q, w, d) -> None:

        assert p>4, "p must be prime a greater than 4"
        assert d>1 and d<p//4, "d must satisfy 2<=d<=p//4"
        assert q<p, "the length q of the sequences must be less than p"

        self.p = p
        self.q = q
        self.w = w
        self.d = d
        self.p_elements = galois.GF(p).primitive_elements


    def get_FHSsequence(self, a, k):
        fhs = []
        for i in range(self.q):
            fhs.append( sum( [(a**i + j) % self.p for j in range(k, k+self.w)] ) +k)

        return fhs


    def get_OCWGFHSfamily(self):

        WGFHSfamily = []
        for a in self.p_elements:
            for k in range(self.p):

                seq = self.get_FHSsequence(int(a), k)

                if get_min_gap(seq) >= self.d:
                    WGFHSfamily.append(seq)

        WGFHSfamily = np.array(WGFHSfamily) - np.min(WGFHSfamily)
        return WGFHSfamily


        