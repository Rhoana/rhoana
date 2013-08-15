#---------------------
#Contour Size Tester
#Daniel Miron
#
#runs contour_size_extractor for an entire data set
#7/30/13
#---------------------


import sys
sys.path.append(r'c:\Python27\Lib\site-packages')
import h5py
import numpy as np
import glob
import os

import math
import re
import time
import threading
from Queue import Queue
import contour_size_extractor as extractor


class Tester:
    def __init__(self, directory, max_x, max_y, location):
        self.directory = directory
        self.max_x = max_x
        self.max_y = max_y
        self.z = location[2]
        self.pick_location = location
        
        self.rows = 0
        self.columns = 0
        self.layers = 0
        
        self.extractor_dict = dict() #keys are indices, values are extractor threads
        
        self.num_labels = 0
        self.label_dict = dict() #keys are float indices, values are labels
        
    def set_dimensions(self, rows, columns, layers, w):
        '''sets the dimensions of the viewing box'''
        self.rows = rows
        self.columns = columns
        self.layers = layers
        self.pick_location = (self.pick_location[0]/pow(2, w) - 1, self.pick_location[1]/pow(2, w) - 1, self.pick_location[2])
        
    def run(self, start_z, end_z):
        extractors = []
        for z in range(start_z, end_z):
            extr = extractor.Extractor(directory, z, max_x, max_y)
            extr.find_contours()
                           
    def parse_ids(self, args):
        '''reformat ids into a list with the primary id the first element'''
        primary_id = []
        secondary_ids = []
        split_str = re.split(":", args[0])
        primary_id = [int(split_str[0])]
        if split_str[1] != "":
            secondary_ids = [int(label) for label in re.split(',', split_str[1])]
        ids = [primary_id + secondary_ids]
        return ids
        
        
if __name__ == '__main__':
    sys.argv.pop(0)
    
    #extract command line arguments
    directory = sys.argv[0]
    
    location = (int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])) #x,y,z
    max_x =int(sys.argv[4])
    max_y = int(sys.argv[5])
    
    start_z = int(sys.argv[6])
    end_z = int(sys.argv[7])
        
    extr = extractor.Extractor(directory, location[2], max_x, max_y)
    tester = Tester(directory, max_x, max_y, location)
    
    tester.set_dimensions(extr.rows, extr.columns, extr.layers, extr.w)
    
    tester.run(start_z, end_z)
    
    