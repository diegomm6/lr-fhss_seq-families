import random
import numpy as np

EARTRH_G    = 9.80665      # earth gravitational constant in m/s2
EARTRH_R    = 6371000      # earth radius in meters
SAT_H       = 600000       # satellite altitude in meters
SAT_RANGE   = 1500000      # satellite comm max range in meters
HDR_TIME    = 0.233472     # header time in seconds
FRG_TIME    = 0.1024       # fragment time in seconds
OBW_BW      = 488.28125    # OBW bandwidth in Hz
OCW_RX_BW   = 200000       # OCW receiver bandwidth in Hz
OCW_FC      = 868100000    # OCW channel carrier freq
GAIN_TX     = 2.5          # transmitter antenna gain in db
GAIN_RX     = 22.6         # receiver antenna gain in db
TX_PWR_DB   = 30           # transmission power in dbm

TH2         = 0            # SINR collision determination threshold in dB
SYM_THRESH  = 0.2          # symbol collision threshold per hdr/frg in %

MIN_FRGS    = 8            # minimum number of payload fragments
MAX_FRGS    = 31           # maximum number of payload fragments
MAX_HDRS    = 3            # maximum number of headers

AWGN_VAR_DB = -174 + 6 + 10*np.log10(OBW_BW)          # AWGN variance in db (-174 + 6)
MAX_FRM_TM  =  MAX_HDRS*HDR_TIME + MAX_FRGS*FRG_TIME  # maximum lr-fhss frame time in sec


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


# greatest common division
def gcd(a, b):
    if(b == 0):
        return abs(a)
    else:
        return gcd(b, a % b)


# hamming correlation with shift 0 for sequences
# u and v with the same length (assumed)
def hamming_correlation(u,v):
    u_eq_v = u == v
    return u_eq_v.sum()


# maximal hamming correlation for sequences
# u and v with the same length (assumed)
#
# as implemented in equation (3) and (4) from [1]
# [1] Lempel, A., & Greenberger, H. (1974). Families of sequences with optimal
# Hamming-correlation properties. IEEE Transactions on Information Theory, 20(1), 90-94.
def maxHC(u,v):

    # for crosscorrelation shift stars in 0
    # for autocorrelation shift stars in 1
    start = 0
    if np.array_equal(u,v): start = 1

    current_maxHC = 0
    for shift in range(start, len(u)):

        hc = hamming_correlation(u, np.roll(v, shift))
        
        if hc > current_maxHC:
            current_maxHC = hc
    
    return current_maxHC


# average hamming auto correlation for a family of
# sequences with the same length (assumed)
#
# as implemented in equation (4) from [2]
# [2] Peng, D. Y., Niu, X. H., & Tang, X. H. (2010). Average Hamming correlation 
# for the cubic polynomial hopping sequences. IET communications, 4(15), 1775-1786.
def avg_autoHC(fam):
    _avgHC = 0
    M, L = fam.shape
    for i in range(M):
        for shift in range(1, L):
            _avgHC += hamming_correlation(fam[i], np.roll(fam[i], shift))

    return _avgHC / (M * (L-1))


# average hamming cross correlation for a family of
# sequences with the same length (assumed)
#
# as implemented in equation (5) from [2]
# [2] Peng, D. Y., Niu, X. H., & Tang, X. H. (2010). Average Hamming correlation 
# for the cubic polynomial hopping sequences. IET communications, 4(15), 1775-1786.
def avg_crossHC(fam):
    _avgHC = 0
    M, L = fam.shape
    for i in range(M):
        for j in range(i):
            for shift in range(L):
                _avgHC += hamming_correlation(fam[i], np.roll(fam[j], shift))

    n = M * (M-1) / 2
    return _avgHC / (L*n)


# average maximal hamming correlation for all paris of
# sequences from a single family
def avg_maxHC(fam):
    mean = 0
    s = len(fam)
    for i in range(s):
        for j in range(i+1):
            mean += maxHC(fam[i], fam[j])

    n = s * (s+1) / 2
    return mean / n


# average maximal hamming correlation for all paris of
# sequences from two families
def avg_maxHC_2fam(fam1, fam2):
    mean = 0
    s = len(fam1)
    for i in range(s):
        for j in range(s):
            mean += maxHC(fam1[i], fam2[j])
        
    n = s**2
    return mean / n


# delete frequences above the maximum, raises error if it
# disrupts the minimum gap property
def filter_freq(seq, maxfreq, mingap):
    newseq = np.delete(seq, np.where(seq >= maxfreq)[0])
    assert get_min_gap(newseq) == mingap, "couldn't filter sequences while preserving minimum gap"
    return newseq


# split the given sequence into a family of sequences with size q
def split_seq(seq, q):
    family = []
    i=0
    j=q
    while j < len(seq):
        family.append(seq[i:j])
        i+=q
        j+=q

    return np.array(family)
    

# calculate the number of frequency hops for the payload
# as a function of its size and the coding rate
def numHops(payload_length, CR):

    assert CR==1 or CR==2, "Only CR 1/3 and CR 2/3 supported"

    length_bits = ( payload_length + 2 ) * 8 + 6
    length_bits *= (3/CR)

    nb_hops_out = ( length_bits + 47 ) // 48

    return nb_hops_out


#############################################
################## Doppler ##################
#############################################

def dBm2mW(dBm):
    return np.power(10, (dBm/10))

def mW2dBm(mW):
    return 10 * np.log10(mW)

# m, lora range
def get_coverageTime(r):

	R = 6371000 # m
	H = 600000  # m, satellite altitude
	v = 7562    # m/s

	x = (R**2 + (R+H)**2 - r**2) / (2*R*(R+H))
	theta = np.arccos(x)

	return theta * (R+H) / v


def get_FS_pathloss(d, f):
    c = 299792458  # m/s
    return (c / (4*np.pi*d*f))**2


def get_distance(sensitivity_dBm):

	c = 299792458  # m/s
	fc = 868000000 # hz, carrier frequency

	sensitivity_mW = dBm2mW(sensitivity_dBm)
	TXpower = dBm2mW(14)
	Txgain = dBm2mW(0)
	RXgain = dBm2mW(5)
	
	a = np.sqrt(sensitivity_mW/(TXpower*Txgain*RXgain))

	return c/(4*np.pi*a*fc)


def get_coverageRadius(maxRange):

	R = 6371000 # m
	H = 600000  # m, satellite altitude

	x = 2*R*R + 2*H*R

	z = (x + H**2 - maxRange**2) / x
	beta = np.arccos(z)

	return beta*R


def dopplerShift(t):

	c = 299792458  # m/s
	g = 9.80665    # m/s2
	R = 6371000    # m
	H = 600000     # m,  satellite altitude
	fc = 868000000 # hz, carrier frequency

	x = 1 + H/R

	a = fc / c

	b = np.sqrt(g*R/x)

	psi = t * np.sqrt(g/R) / np.sqrt(np.power(x, 3))

	c = np.sin(psi) / np.sqrt(np.power(x,2) - 2*x*np.cos(psi) + 1)

	return a*b*c


def get_randomDoppler() -> float:

    sensitivity = -137
    maxRange = get_distance(sensitivity)
    Rcov = get_coverageRadius(maxRange)
    Tcov = get_coverageTime(Rcov)

    r0 =  np.sqrt(random.uniform(0,1))
    theta0 = 2 * np.pi * random.uniform(0,1)

    t0 = r0 * np.cos(theta0) * Tcov

    return dopplerShift(t0)


def get_visibility_time(d):
    
    E = np.arcsin( (SAT_H**2 + 2*SAT_H*EARTRH_R - d**2) / (2*d*EARTRH_R) ) # elevation angle
    dg = EARTRH_R * np.arcsin( d*np.cos(E) / (EARTRH_R+SAT_H) )            # ground range
    v = np.sqrt( EARTRH_G*EARTRH_R / (1 + SAT_H/EARTRH_R) )                # satellite velocity
    tau = dg / v                                                           # half satellite visibility time
    
    return tau


def edgedetect(a):

    copya = np.copy(a)
    for i in range(1,len(a)):
        if not( a[i-1]==0 and a[i]==1 ):
            copya[i] = 0

    return copya


def cornerdetect(m):

    xedges = np.apply_along_axis(edgedetect, 0, m)
    yedges = np.apply_along_axis(edgedetect, 1, m)
    corners = np.logical_and(xedges, yedges)

    return corners

def bisection(array, value):
    '''
    Given an ``array`` , and given a ``value`` , returns an index j such that
    ``value`` is between array[j] and array[j+1].
    ``array`` must be monotonic increasing.
    j=-1 or j=len(array) is returned to indicate that ``value`` is out of range
    below and above respectively.
    '''

    n = len(array)
    if (value < array[0]):
        return -1
    elif (value > array[n-1]):
        return n
    
    jl = 0   # Initialize lower
    ju = n-1 # and upper limits.

    while (ju-jl > 1):    # If we are not yet done,
        jm = (ju+jl) >> 1 # compute a midpoint with a bitshift
        if (value >= array[jm]):
            jl = jm       # and replace either the lower limit
        else:
            ju = jm       # or the upper limit, as appropriate.
        # Repeat until the test condition is satisfied.
            
    if (value == array[0]):# edge cases at bottom
        return 0
    elif (value == array[n-1]):# and top
        return n-1
    else:
        return jl
