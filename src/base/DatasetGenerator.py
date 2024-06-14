import random, json, os
import numpy as np
from src.base.base import *
from PIL import Image
from src.base.LoRaTransmission import LoRaTransmission
from src.families.LR_FHSS_DriverMethod import LR_FHSS_DriverFamily


class DatasetGenerator():

    def __init__(self, CR, numOBW, freqGranularity, timeGranularity) -> None:

        self.id = 0
        self.OCW = 0
        self.CR = CR
        self.TXpower_dB = TX_PWR_DB
        self.maxFrameT = MAX_FRM_TM
        self.freqGranularity = freqGranularity # freq slots per OBW
        self.timeGranularity = timeGranularity # time slots per fragmet

        driverFHSfam = LR_FHSS_DriverFamily(q=34, regionDR="EU137")
        self.FHSfam = driverFHSfam.FHSfam
        self.FHSfamsize = len(self.FHSfam)

        self.headerSlots = round(timeGranularity * HDR_TIME / FRG_TIME)
        self.freqPerSlot = OBW_BW / self.freqGranularity
        self.frequencySlots = int(round(OCW_RX_BW / self.freqPerSlot))

        self.simTime = self.frequencySlots # force square images

        max_packet_duration = MAX_HDRS * self.headerSlots + MAX_FRGS * timeGranularity
        self.startLimit = self.simTime - max_packet_duration

        OCWchannelTXBW = numOBW * OBW_BW  # OCW transmitter bandwidth in Hz
        self.maxFreqShift = (OCW_RX_BW - OCWchannelTXBW) / 2
        self.baseFreq = round(self.maxFreqShift / self.freqPerSlot) # freq offset to center TX window over RX window

        assert CR==1 or CR==2, "Only CR 1/3 and CR 2/3 supported"
        self.numHeaders = 3 # CR == 1
        if CR == 2:
            self.numHeaders = 2
        
        self.header = np.ones((freqGranularity, self.headerSlots))  # header block
        self.fragment = np.ones((freqGranularity, timeGranularity)) # fragment block

        self.saveboxedimg = False

    def calculate_hdr_frg_times(self, time, numHeaders, numFragments) -> list[float]:

        # doppler shift decreases as the satellites moves as seeen from the nodes
        hdr_frg_times = []

        for hdr in range(numHeaders):
            hdr_frg_times.append(time)
            time -= HDR_TIME

        for frg in range(numFragments):
            hdr_frg_times.append(time)
            time -= FRG_TIME

        return hdr_frg_times
    

    def get_transmission(self, seq_id, numFragments) -> LoRaTransmission:

        if numFragments==0:
            numFragments = random.randrange(8, 32)

        payload_size = numFragments
        seq_length = int(numFragments + self.numHeaders)

        startSlot = random.randrange(0, self.startLimit)

        sequence = self.FHSfam[seq_id]
        sequence = sequence[:seq_length]

        dis2sat = random.uniform(SAT_H, SAT_RANGE)
        tau = get_visibility_time(dis2sat)
        time = random.uniform(-tau+self.maxFrameT, tau-self.maxFrameT)
        hdr_frg_times = self.calculate_hdr_frg_times(time, self.numHeaders, numFragments)
        dynamicDoppler = [dopplerShift(t) for t in hdr_frg_times]

        tx = LoRaTransmission(self.id, self.id, startSlot, self.OCW, self.numHeaders,
                              payload_size, numFragments, sequence, seq_id, dis2sat,
                              dynamicDoppler, self.TXpower_dB)

        return tx
    

    def get_TXlist(self, numTX, numFragments) -> list[LoRaTransmission]:

        TXlist = []
        for i in range(numTX):
            seq_id = random.randrange(0, self.FHSfamsize)
            TXlist.append(self.get_transmission(seq_id, numFragments))

        return TXlist
    

    def get_rcvM(self, transmissions: list[LoRaTransmission], dynamic: bool) -> np.ndarray:

        rcvM = np.zeros((self.frequencySlots, self.simTime))

        for tx in transmissions:

            dopplershift = round(tx.dopplerShift[0] / self.freqPerSlot)

            time = tx.startSlot
            for fh, obw in enumerate(tx.sequence):

                # variable doppler shift per header / fragment
                if dynamic:
                    dopplershift = round(tx.dopplerShift[fh] / self.freqPerSlot)

                startFreq = self.baseFreq + obw * self.freqGranularity + dopplershift
                endFreq = startFreq + self.freqGranularity

                # write header
                if fh < tx.numHeaders:
                    endTime = time + self.headerSlots
                    rcvM[startFreq : endFreq, time : endTime] += (self.header)

                # write fragment
                else:
                    endTime = time + self.timeGranularity
                    rcvM[startFreq : endFreq, time : endTime] += (self.fragment)
                
                time = endTime

        return rcvM


    # returns an RGB image from collision matrix rcvM
    def get_RGBimg(self, rcvM: np.ndarray) -> np.ndarray:

        m,n = rcvM.shape
        rcvM_RGB = np.zeros((m,n,3), dtype=np.uint8)
  
        rcvM_RGB[rcvM == 0] = [0, 0, 0]   # idle, black
        rcvM_RGB[rcvM == 1] = [0, 255, 0] # success, green
        rcvM_RGB[rcvM > 1] = [255, 0, 0]  # collision, red
        
        return rcvM_RGB
    

    # calcualte LR-FHSS signal coordiantes in spectogram as lower and upper frquency slot
    # and start and end time slot
    def get_boundingbox(self, tx: LoRaTransmission, dynamic:bool) -> list[int]:

        totalSlots = self.headerSlots * tx.numHeaders + self.timeGranularity * tx.numFragments
        endSlot = tx.startSlot + totalSlots - 1

        dopplershift_start = round(tx.dopplerShift[0] / self.freqPerSlot)
        dopplershift_end = dopplershift_start
        if dynamic:
            dopplershift_end = round(tx.dopplerShift[tx.sequence.index(min(tx.sequence))] / self.freqPerSlot)

        lowerFreq = self.baseFreq + min(tx.sequence) * self.freqGranularity + dopplershift_start
        upperFreq = self.baseFreq + max(tx.sequence) * self.freqGranularity + dopplershift_end

        return tx.startSlot, lowerFreq, endSlot, upperFreq
    

    # Draw rectangle in image given the coordinates as a tuple xmin, ymin, xmax, ymax
    def draw_rectangle(self, image, box_coords):

        margin = 1 # add margin in pixels
        RGBcolor = [0, 0, 255] # define blue color in RGB
        xmin, ymin, xmax, ymax = box_coords

        image[ymin - margin, xmin - margin : xmax + margin +1] = RGBcolor # bottom side
        image[ymax + margin, xmin - margin : xmax + margin +1] = RGBcolor #  upper side
        image[ymin - margin : ymax + margin +1, xmin - margin] = RGBcolor # left side
        image[ymin - margin : ymax + margin +1, xmax + margin] = RGBcolor # righ side

        return image
    

    def create_boundingbox_dataset(self, dynamic:bool, dataset_name:str, runs_list: list[int],
                                   numTX_list: list[int], numFragments_list: list[int]) -> None:
        
        if not os.path.exists(dataset_name):
            os.makedirs(dataset_name)

        coco_json = {
            "images": [],
            "annotations": [],
            "categories": [{"id": i, "name": f"seq{i:03d}"} for i in range(1, 385)]
        }

        counter_1tx = 0
        image_id = 0
        for runs, numTX, numFragments in zip(runs_list, numTX_list, numFragments_list):

            for id in range(runs):

                #img_name = f"{numTX:03d}tx_{numFragments:02d}frg_{id:03d}_{counter_1tx}"
                img_name = f"{image_id:05d}"

                if numTX==1:
                    transmissions = [self.get_transmission(id, numFragments)]
                else:
                    transmissions = self.get_TXlist(numTX, numFragments)

                rcvM = self.get_rcvM(transmissions, dynamic)
                rcvM_RGB = self.get_RGBimg(rcvM)
                image = Image.fromarray(rcvM_RGB, mode='RGB')
                image.save(os.path.join(os.getcwd(), f"{dataset_name}\{img_name}.png"))

                boxed_image = np.copy(image) # BOXED IMAGE

                coco_json["images"].append({
                    "id": image_id,
                    "file_name": f"{img_name}.png"
                })

                annotation_id = 0
                for tx in transmissions:
                    x1, y1, x2, y2 = self.get_boundingbox(tx, dynamic)
                    boxed_image = self.draw_rectangle(boxed_image, (x1, y1, x2, y2))

                    coco_json["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": int(tx.seqid) +1,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "area": int((x2-x1) * (y2-y1)),
                        "iscrowd": 0
                    })

                    annotation_id += 1

                image_id += 1

                if self.saveboxedimg:
                    boxed_image = Image.fromarray(boxed_image, mode='RGB')
                    boxed_image.save(os.path.join(os.getcwd(), f"{dataset_name}\{'b'}oxed_{img_name}.png"))

            if numTX==1: counter_1tx += 1

        # Save to a JSON file
        json_filename = f'{dataset_name}.json'
        with open(json_filename, 'w') as f:
            json.dump(coco_json, f, indent=4)


    # returns a string containing the labels for a list of transmissions
    def get_label(self, transmissions: list[LoRaTransmission]) -> str:

        TXSeqIds = [tx.seqid for tx in transmissions]

        label = ['0'] * 384
        for id in TXSeqIds:
            label[id] = '1'

        return ''.join(label)
    

    def create_classification_dataset(self, dynamic:bool, dataset_name:str, numFragments_list: list[int],
                                      numTX_list: list[int], runs_list: list[int]) -> None:
        
        if not os.path.exists(dataset_name):
            os.makedirs(dataset_name)
        
        csv_filename = f'{dataset_name}.csv'
        if not os.path.exists(csv_filename):
            with open(csv_filename, 'w') as file: pass

        counter_1tx = 0
        for numTX, runs, numFragments in zip(numTX_list, runs_list, numFragments_list):

            labels = ""
            for id in range(runs):
                imgid = f"{numTX:03d}tx_{numFragments:02d}frg_{id:03d}_{counter_1tx}"

                if numTX==1:
                    counter_1tx += 1
                    transmissions = [self.get_transmission(id, numFragments)]
                else:
                    transmissions = self.get_TXlist(numTX, numFragments)

                rcvM = self.get_rcvM(transmissions, dynamic)
                rcvM_RGB = self.get_RGBimg(rcvM)
                image = Image.fromarray(rcvM_RGB, mode='RGB')
                image.save(os.path.join(os.getcwd(), f"{dataset_name}\{imgid}.png"))

                label = self.get_label(transmissions)
                labels += f"{imgid},{label}\n"

            with open(csv_filename, 'a') as file:
                file.write(labels)
