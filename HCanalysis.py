import numpy as np
from src.base.base import *
import matplotlib.pyplot as plt
from src.families.LiFanMethod import LiFanFamily
from src.families.LR_FHSS_DriverMethod import LR_FHSS_DriverFamily


if __name__ == "__main__":

    lifan_HCdata = []
    driver_HCdata = []
    seq_lengths = [q for q in range(2,35)]

    for q in seq_lengths:

        lr_fhssGenerator = LR_FHSS_DriverFamily(q)
        driver_family = lr_fhssGenerator.get_family()*8
        for i in range(len(driver_family)):
            driver_family[i] = driver_family[i] + np.random.randint(8)

        driver_HCdata.append(avg_crossHC(driver_family)) # avg_maxHC

        liFanGenerator = LiFanFamily(q, maxfreq=280, mingap=8)
        lifan_family = liFanGenerator.get_family(281, 8, '2l')
        #print(f'q = {q}; famsize = {len(lifan_family)}')
        lifan_HCdata.append(avg_maxHC(lifan_family))

    
    plt.figure(figsize=(8,8))
    plt.plot(seq_lengths, driver_HCdata, label='driver')
    plt.plot(seq_lengths, lifan_HCdata, label='lifan')
    plt.title('max Hamming Correlation')
    plt.xlabel('seq length')
    plt.ylabel('maxHC')
    plt.grid()
    plt.legend()
    plt.show()


