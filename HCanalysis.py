import numpy as np
from src.base.base import *
import matplotlib.pyplot as plt
from src.families.LiFanMethod import LiFanFamily
from src.families.LR_FHSS_DriverMethod import LR_FHSS_DriverFamily
from src.families.HashMethod import HashFamily



def c():

    driverFHSfam = LR_FHSS_DriverFamily(35, "EU137")        # EU137 ; EU336 ; (US)

    for i, fh in enumerate(driverFHSfam.FHSfam):
        print(f"seq {i} : {fh}\n")


    driverFHSfam = LR_FHSS_DriverFamily(31, "EU137")        # EU137 ; EU336 ; US1523
    print(f"driver: {avg_crossHC(driverFHSfam.FHSfam)}")

    hashFHSfam = HashFamily(31)
    print(f"hash: {avg_crossHC(hashFHSfam.FHSfam)}")



def max_coincidence():

    driverFHSfam = LR_FHSS_DriverFamily(35, "EU137")        # EU137 ; EU336 ; (US)
    fhsfam = driverFHSfam.FHSfam

    max_matches = 0
    for i in range(len(fhsfam)):

        for j in range(i+1, len(fhsfam)):

            current_max_match = 0

            k=0
            while fhsfam[i][k] == fhsfam[j][k]:
                current_max_match += 1
                k+=1

            if current_max_match > max_matches:
                max_matches = current_max_match

        return max_matches


if __name__ == "__main__":

    print(max_coincidence())

    """
    lifan_HCdata = []
    driver_HCdata = []
    seq_lengths = [2*q for q in range(1,31)]                   # 2,36  ; 1,44  ; 1,31

    for q in seq_lengths:

        driverFHSfam = LR_FHSS_DriverFamily(q, "US1523")        # EU137 ; EU336 ; US1523
        driver_HCdata.append(avg_maxHC(driverFHSfam.FHSfam))   # avg_crossHC ; avg_maxHC

        liFanFHSfam = LiFanFamily(q, maxfreq=3120, mingap=52)    # 280 ; 688 ; 3120
        liFanFHSfam.set_family(3121, 52, '2l')                   # 281-8 ; 689-8 ; 3121-52
        lifan_HCdata.append(avg_maxHC(liFanFHSfam.FHSfam))

        if q%10==0:
            print(f"done for {q}")

    
    print(f"driver : {driver_HCdata}")
    print(f"\nlifan : {lifan_HCdata}")

    plt.figure(figsize=(8,8))
    plt.plot(seq_lengths, driver_HCdata, label='driver')
    plt.plot(seq_lengths, lifan_HCdata, label='lifan')
    plt.title('max Hamming Correlation')
    plt.xlabel('seq length')
    plt.ylabel('maxHC')
    plt.grid()
    plt.legend()
    plt.show()
    """


