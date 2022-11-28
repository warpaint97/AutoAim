import numpy as np
import cv2 as cv
from mss import mss
import pydirectinput as pdi

class ScreenCapture:
    def __init__(self):
        self.bounding_box = {'top': 0, 'left': 0, 'width': pdi.size()[0], 'height': pdi.size()[1]}
        self.sct = mss()
        
    def getScreenImg(self):
        sct_img = self.sct.grab(self.bounding_box)
        img = np.array(sct_img)[:,:,:3].astype(np.uint8).copy()
        del sct_img
        return img 