#------------------
#Contour Size extractor
#Daniel Miron
#
#writes contours to pickle files given a mojo folder
#7/30/13
#------------------

import sys
sys.path.append(r'c:\Python27\Lib\site-packages')
import h5py
import numpy as np
import glob
import os

import pickle
import math
import time
import cv2
import threading
from Queue import Queue

import Polygon

import scipy.ndimage as sp

from pysqlite2 import dbapi2 as sqlite

def contours_to_poly(contours, hierarchy):
    def is_hole(idx):
        if hierarchy[idx][3] < 0:
            return False
        return not is_hole(hierarchy[idx][3])
    p = Polygon.Polygon()
    hierarchy = hierarchy[0]
    for idx, c in enumerate(contours):
        p.addContour(c.reshape((c.shape[0], c.shape[2])), is_hole(idx))
    return p

class Extractor:
    def __init__(self, directory, slice_num, max_x, max_y):
        self.alive = True
        self.directory = directory #mojo directory
        tiledir = os.path.join(self.directory, 'ids', 'tiles')
        self.w = len(glob.glob(os.path.join(tiledir, '*'))) - 1 #default to lowest resolution
        self.w_str = "w={0:08}".format(self.w)
        self.label_folder = os.path.join(tiledir, self.w_str)
        
        self.segment_file = os.path.join(self.directory, "ids", "segmentInfo.db")
        self.z_folders = sorted(glob.glob(os.path.join(self.label_folder, "*")))
        self.tile_name = glob.glob(os.path.join(self.z_folders[slice_num], "*"))[0]
        h5_file = h5py.File(self.tile_name, "r")
        self.label_key = h5_file.keys()[0]
        self.shape = np.shape(h5_file[self.label_key][...])
        self.label_ids = self.get_ids(h5_file)
        h5_file.close()
        
        self.tile_rows = self.shape[0]
        self.tile_columns = self.shape[1]
        
        self.tiles_per_layer = len(glob.glob(os.path.join(self.z_folders[0], "*")))
        
        self.rows = max_x/pow(2, self.w) - 1
        self.columns = max_y/pow(2, self.w) - 1
        self.layers = int(os.path.basename(str(self.z_folders[-1])).split("0")[-1])+1
        
        self.z = slice_num
        
        color_file = h5py.File(os.path.join(self.directory, "ids", "colorMap.hdf5"))
        self.color_map = color_file["idColorMap"][...]
        color_file.close()
        
        self.contour_name = directory + "\\label_contour_slice" + str(slice_num).zfill8 + ".p" #pad with 0s to ease sorting
        self.normal_name = directory + "\\label_normal_slice" + str(slice_num).zfill8 + ".p"
        
    def get_ids(self, h5_file):
        slice_array = h5_file[self.label_key][...]
        ids = np.unique(slice_array)
        return ids
        
    def find_contours(self):
        st = time.time()
        tot_contours = dict()
        tot_normals = dict()
        tot_mask = np.zeros((self.tile_rows, self.tile_columns))
        for label in self.label_ids:
            if label%50 == 0:
                print time.time()-st, self.z
            t_file = h5py.File(self.tile_name, "r")
            labels = t_file[self.label_key][...]
            t_file.close()
            buffer_array = np.zeros((np.shape(labels)[0]+2, np.shape(labels)[1]+2), np.uint8) #buffer by one pixel on each sid
            buffer_array[1:-1, 1:-1] |= labels == label

            if np.any(buffer_array):
                blur_mask = sp.filters.gaussian_filter(buffer_array.astype(float),11)
                dy, dx = np.gradient(blur_mask)

                contours, hierarchy  = cv2.findContours(buffer_array, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                p = contours_to_poly(contours, hierarchy)
                tristrips = [np.hstack((np.array(s), np.zeros((len(s), 1)))).astype(int) for s in p.triStrip()]
                normals = [np.zeros((s.shape[0], 3), np.float) for s in tristrips]
                for strip, norms in zip(tristrips, normals):
                    norms[:, 0] = -dx[strip[:, 1], strip[:, 0]]
                    norms[:, 1] = -dy[strip[:, 1], strip[:, 0]]
                    strip[:, 2] = self.z

                for strip in tristrips:
                    tmp_n = np.zeros((len(strip), 3), np.float)

                tot_contours[label] = tristrips
                tot_normals[label] = normals
        
        pickle.dump(tot_contours, open(self.contour_name, "wb"))
        pickle.dump(tot_normals, open(self.normal_name, "wb"))
        return
        
    def run(self):
        self.find_contours()
            
        
        