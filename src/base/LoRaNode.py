import random
import numpy as np
from src.base.base import get_randomDoppler, dopplerShift
from src.base.LoRaTransmission import LoRaTransmission
from src.families.LR_FHSS_DriverMethod import FHSfamily

class LoRaNode():
    """
    A class representing a LoRa node.

    Args:
        id: The node ID.
        CR: The coding rate.
        numOCW: The number of occupied chirps per symbol.
        useGrid: Whether to use a grid topology.
        numGrids: The number of grids in the network.
        startLimit: The start limit of the simulation time.

    Attributes:
        id (int): The node ID.
        CR (int): The coding rate.
        numOCW (int): The number of occupied chirps per symbol.
        useGrid (bool): Whether to use a grid topology.
        numGrids (int): The number of grids in the network.
        startLimit (int): The start limit of the simulation time.
        sent_packets (int): The number of packets sent by the node.
        sent_payload_bytes (int): The number of payload bytes sent by the node.

    Methods:
        restart(): Reset the node's statistics.
        numHops(payload_length): Calculate the number of frequency hops for the payload.
        get_sequence(family): Get the sequence for the payload.
        get_transmissions(family): Get all transmissions during simulation time for this node.

    """
    def __init__(self, id, CR, numOCW, startLimit) -> None:
        self.id = id
        self.CR = CR
        self.numOCW = numOCW
        self.startLimit = startLimit
        self.sent_packets = 0
        self.sent_payload_bytes = 0

        self.headertime = 0.233472
        self.fragmenttime = 0.1024
        self.TXpower = 14 # dBm, approx 25 mW


    def restart(self) -> None:
        self.sent_packets = 0
        self.sent_payload_bytes = 0
          

    def numHops(self, payload_length: int) -> int:
        """
        Calculate the number of frequency hops for the payload
        as a function of its size and the coding rate
        """

        length_bits = ( payload_length + 2 ) * 8 + 6
        length_bits *= (3/self.CR)

        nb_hops_out = ( length_bits + 47 ) // 48  # can be just ceil(bits/48)

        return nb_hops_out
    

    def calculate_hdr_frg_times(self, time, numHeaders, numFragments) -> list[float]:

        hdr_frg_times = []

        # doppler shift decreases as the satellites moves as seeen from the nodes

        for hdr in range(numHeaders):
            hdr_frg_times.append(time)
            time -= self.headertime

        for frg in range(numFragments):
            hdr_frg_times.append(time)
            time -= self.fragmenttime

        return hdr_frg_times
    

    def get_transmissions(self, family: FHSfamily) -> list[LoRaTransmission]:
        """
        Obtain all transmission during simulation time for this node.
        The current model only support one transmission per node,
        therefore we use the node id as the transmission id
        """

        ocw = random.randrange(0, self.numOCW)
        startSlot = random.randrange(0, self.startLimit)

        if self.CR == 1:
            payload_size = random.randrange(13, 58)  # [8-31[ fragments
            numHeaders = 3

        elif self.CR == 2:
            payload_size = random.randrange(29, 118) # [8-31[ fragments
            numHeaders = 2

        else:
            raise Exception(f"Invalid coding rate '{self.CR}'") 

        numFragments = int(self.numHops(payload_size))
        seq_length = int(numFragments + numHeaders)

        seqid, sequence = family.get_random_sequence()
        sequence = sequence[:seq_length]

        self.sent_packets += 1
        self.sent_payload_bytes += payload_size


        g = 9.80665    # m/s2
        R = 6371000    # m
        H = 600000     # m,  satellite altitude
        minRange = H
        maxRange = 1500000 # max range

        d = random.uniform(minRange, maxRange)           # slant distance
        E = np.arcsin( (H**2 + 2*H*R - d**2) / (2*d*R) ) # elevation angle
        dg = R * np.arcsin( d*np.cos(E) / (R+H) )        # ground range
        v = np.sqrt( g*R / (1 + H/R) )                   # satellite velocity
        tau = dg / v                                     # half satellite visibility time
        
        maxFrameTime = 3.8
        time = random.uniform(-tau+maxFrameTime, tau-maxFrameTime)

        hdr_frg_times = self.calculate_hdr_frg_times(time, numHeaders, numFragments)
        dynamicDoppler = [dopplerShift(t) for t in hdr_frg_times]
        #staticDoppler = [0 for t in hdr_frg_times]

        tx = LoRaTransmission(self.id, self.id, startSlot, ocw, numHeaders, payload_size,
                              numFragments, sequence, seqid, d, dynamicDoppler, self.TXpower)

        return [tx]
