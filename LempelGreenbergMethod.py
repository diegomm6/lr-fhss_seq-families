import numpy as np
from galois import GLFSR, primitive_poly
from base import numberToBase

"""
maps a sequence X in P^k to a value in P_k
where P^k is the set of all words of length k over P
P is the finite field (GF) of order p, i.e. P = {0, 1, ..., p-1}
P_k is the finite field (GF) of order p^k, i.e. P_k = {0, 1, ..., p^k-1}
"""
def sigma_transform(X, p, k):

    Y = []
    q = len(X)
    for j in range(q):

        y_j = 0
        for i in range(k):
            y_j += X[(j+i) % q] * p**i

        Y.append(y_j)

    return Y


"""
Following the construction presented in [1]
given an m-sequence X of length q = p^n - 1 over GF(p)
where p is a prime number
then for each k s.t. 1 <= k <= n, the set F of p^k
sequences is an optimal family

a variation of the sigma transform is used

[1] Lempel, A., & Greenberger, H. (1974). Families of sequences with optimal
Hamming-correlation properties. IEEE Transactions on Information Theory, 20(1), 90-94.
"""
def optimal_family(X, p, k):

    family = []
    q = len(X)
    for v in range(p**k):

        z = numberToBase(v, p)
        while len(z) < k:
            z.insert(0,0)
        z.reverse()

        Y = []
        for j in range(q):

            y_j = 0
            for i in range(k):
                y_j += ( ( X[ (j+i) % q] + z[i]) % p ) * p**i

            Y.append(y_j)

        family.append(Y)
        #print(f"Y{v} = ", Y)

    return np.array(family)


"""
Generates a family of sequences with optimal maximal hammig correlation from a single m-sequence

Generated sequences have length q = p^n - 1
over an alphabet A of size |A| = p^k
for any given prime number p
and integers k,n s.t. 1 <= k <= n
"""
class LempelGreenbergFamily():

    def __init__(self, p, n, k) -> None:
        assert k <= n, "condition k <= n is not met"
        self.p = p
        self.n = n
        self.k = k
        self.q = self.p**self.n - 1
        self.lfsr = GLFSR(primitive_poly(self.p, self.n).reverse())

    # return current lfsr state
    def get_lfsr_state(self):
        return self.lfsr.state
    
    # obtain m sequence of length q by advancing the lfsr q steps
    def get_msequence(self):
        return self.lfsr.step(self.q)

    def get_optimal_family(self):
        msequence = np.array(self.get_msequence())
        return optimal_family(msequence, self.p, self.k)
