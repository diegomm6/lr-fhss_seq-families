import galois
from src.families.LiFanMethod import LiFanFamily
from src.families.LR_FHSS_DriverMethod import LR_FHSS_DriverFamily
from src.families.LempelGreenbergMethod import LempelGreenbergFamily


# generate 3 families/sets of Frequency Hopping Sequences
# "lemgreen" for Lempel-Greenberger FHS family
# "driver" for LR-FHSS Driver FHS family
# "lifan" for Li-Fan FHS family
def get_FHSfamily(familyname, numGrids):

    if familyname == "lemgreen":
        polys = galois.primitive_polys(2, 5)
        poly1 = next(polys)
        lempelGreenbergFHSfam = LempelGreenbergFamily(p=2, n=5, k=5, poly=poly1)
        lempelGreenbergFHSfam.set_family(numGrids)
        return lempelGreenbergFHSfam

    elif familyname == "driver":
        driverFHSfam = LR_FHSS_DriverFamily(q=34, regionDR="EU137")
        return driverFHSfam

    elif familyname == "lifan":
        liFanFHSfam = LiFanFamily(q=34, maxfreq=280, mingap=8)
        liFanFHSfam.set_family(281, 8, '2l')
        return liFanFHSfam

    else:
        raise Exception(f"Invalid family name '{familyname}'")


if __name__ == "__main__":

    numGrids = 8            # number of LR-FHSS hopping grids
    familyname = "driver"   # FHS family name
    FHSfamily = get_FHSfamily(familyname, numGrids)

    for fhs in FHSfamily:
        print(fhs)



