import numpy as np
import galois
from src.families.LR_FHSS_DriverMethod import FHSfamily
from src.families.LiFanMethod import LiFanFamily
from src.families.LR_FHSS_DriverMethod import LR_FHSS_DriverFamily
from src.families.LempelGreenbergMethod import LempelGreenbergFamily


class Node():

    def __init__(self, startSlot, ocw, sequence) -> None:
        self.startSlot = startSlot
        self.ocw = ocw
        self.sequence = sequence
    

class Population():

    def __init__(self, familyname, numGrids, numOCW, numNodes, startLimit,
                 numFragments, header_replicas) -> None:
        self.FHSfam: FHSfamily = self.set_FHSfamily(familyname, numGrids)
        self.numOCW = numOCW
        self.numNodes = numNodes
        self.startLimit = startLimit
        self.numFragments = numFragments
        self.header_replicas = header_replicas
        self.nodes = self.init_nodes()


    def set_FHSfamily(self, familyname, numGrids):

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
        

    def restart(self):

        node : Node
        for node in self.nodes:

            node.ocw = np.random.randint(self.numOCW)
            node.startSlot = np.random.randint(self.startLimit)

            seq_length = int(self.numFragments + self.header_replicas)
            sequence = self.FHSfam.get_random_sequence()
            node.sequence = sequence[:seq_length]
        

    def init_nodes(self):

        nodes = []
        for _ in range(self.numNodes):
            
            ocw = np.random.randint(self.numOCW)
            startSlot = np.random.randint(self.startLimit)

            seq_length = int(self.numFragments + self.header_replicas)
            sequence = self.FHSfam.get_random_sequence()
            sequence = sequence[:seq_length]

            nodes.append(Node(startSlot, ocw, sequence))
        
        return nodes

