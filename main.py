from multiprocessing import Pool
import numpy as np
import galois
import hashlib


# Obtain minimal gap between adyacent values for sequence X
def get_min_gap(X):
    gap = np.inf
    q = len(X)

    for i in range(q):

        d = abs(X[(i+1) % q] - X[i])

        if d < gap:
            gap = d

    return gap


# Transform number n to base b
def numberToBase(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]


# hamming correlation with shift 0 for sequences
# u and v with the same length (assumed)
def hamming_correlation(u,v):
    u_eq_v = u == v
    return u_eq_v.sum()

# maximal hamming correlation for sequences
# u and v with the same length (assumed)
def maxHC(u,v):
    current_maxHC = 0

    for shift in range(1, len(u)):

        hc = hamming_correlation(u, np.roll(v, shift))
        
        if hc > current_maxHC:
            current_maxHC = hc
    
    return current_maxHC


# average maximal hamming correlation for all paris of
# sequences from a single family
def avg_maxHC(fam):
    mean = 0
    s = len(fam)
    for i in range(s):
        for j in range(i):
            mean += maxHC(fam[i], fam[j])

    n = s * (s+1) / 2
    return mean / n


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

c = galois.primitive_poly(2, 5)
lfsr = galois.GLFSR(c.reverse())

p = 2
k = 5
n = 5
q = p**n - 1

x1 = lfsr.step(q)
method1_fam = optimal_family(np.array(x1), p, k)
method1_fam = method1_fam *8

"""
let l, d s.t. 1 < d < l/2
& gcd(l, d) = gcd(l, d+1) = 1

generate an optimal WGFHS with parameters (2*l, l, 2, d-1)
"""
def get_2l_sequence(l, d):
    s = []
    t = []

    for i in range(l):
        s.append( (i*d) % l )
        t.append( (i*(d+1) + 1) % l )

    return np.array(s + t)


"""
let l, d s.t. 1 < d < (l-1)/2
& gcd(l, d) = gcd(l, d+1) = gcd(l, d+2) = 1

generate an optimal WGFHS with parameters (3*l, l, 3, d-1)
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


# greatest common division
def gcd(a, b):
    if(b == 0): return abs(a)
    else: return gcd(b, a % b)


l = 277
d = 8
x = get_3l_sequence(l, d)

method2_family = []
i=0
j=31
while j < len(x):
    method2_family.append(x[i:j])
    i+=31
    j+=31

method2_family = np.array(method2_family)


# obw to transmit fragment k for node x 
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


# all possible fhs of length q
def get_hashFamily(q):
    fam = []
    for x in range(2**9):
        fam.append(get_hashFHS(x, q))

    return np.array(fam)


q = 31
hash_fam = get_hashFamily(q)


# array representoing the number of transmissions
# on each channed OCW and sub-channel OBW
numOCW = 7
numOBW = 280
numGrids = 8
seq_length = 31
startLimit = 500


def get_seq_time(family, startLimit):
    startTime = np.random.randint(startLimit)
    seq_id = np.random.randint(len(family))
    return family[seq_id], startTime


# default FHS generator with hash function
def transmissions_simulation(nodes, family, useGrid):
    transmissions = np.zeros((numOCW, numOBW, startLimit + seq_length))

    for n in range(nodes):
        ocw = np.random.randint(numOCW)
        grid = 0
        if useGrid:
            grid = np.random.randint(numGrids)

        # choose random sequence and starting time
        seq, t0 = get_seq_time(family, startLimit)

        for t, obw in enumerate(seq):
            transmissions[ocw][obw + grid][t0 + t] += 1

    return transmissions


def get_avg_collisions(v):
    nodes, runs, family, useGrid = v
    avg = 0
    for r in range(runs):
        tx = transmissions_simulation(nodes, family, useGrid)
        avg += (tx > 1).sum()
    return avg / runs


def main():
    n = 1500
    runs = 30
    pool = Pool(processes=3)  # set the processes max number 3
    result = pool.map(get_avg_collisions,
                      [(n, runs, hash_fam, True),
                       (n, runs, method1_fam, True),
                       (n, runs, method2_family, False)])
    pool.close()
    pool.join()

    hash_avg, method1_avg, method2_avg = result

    print(f"n = {n} nodes; runs = {runs}")
    print(f"hash collisions = {hash_avg}")
    print(f"method1 collisions = {method1_avg}")
    print(f"method2 collisions = {method2_avg}")


if __name__ == "__main__":
    main()
