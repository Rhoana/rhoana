#--------------------
#Contour Extractor
#Daniel Miron
#7/17/2013
#
#--------------------

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

import scipy.ndimage as sp

from pysqlite2 import dbapi2 as sqlite

class Extractor:
    def __init__(self, out_q, directory, label_ids, location, max_x, max_y):
        
        self.test_file = open(r'C:\Users\DanielMiron\rhoana\Rendering\test.txt', 'w')
        self.out_q = out_q
        
        self.directory = directory
        self.w = len(glob.glob(self.directory + "\\ids\\tiles\\*"))-1
        self.w_str = "w={0:08}".format(self.w)
        self.label_folder = self.directory +"\\ids\\tiles\\" + self.w_str
        
        self.segment_file = self.directory + "\\ids\\segmentInfo.db"
        self.z_folders = glob.glob(self.label_folder + "\\*")
        h5_file = h5py.File(glob.glob(self.z_folders[0] + "\\*")[0], "r")
        self.label_key = h5_file.keys()[0]
        self.shape = np.shape(h5_file[self.label_key][...])
        h5_file.close()
        
        self.tile_rows = self.shape[0]
        self.tile_columns = self.shape[1]
        
        self.tiles_per_layer = len(glob.glob(self.z_folders[0] + "\\*"))
        
        self.rows = max_x/pow(2, self.w) - 1
        self.columns = max_y/pow(2, self.w) - 1
        self.layers = len(self.z_folders)
        
        self.label_ids = label_ids
        
        self.z_order = self.make_z_order(location[2])
        
        color_file = h5py.File(self.directory + "\\ids\\colorMap.hdf5")
        self.color_map = color_file["idColorMap"][...]
        color_file.close()
        
    def make_z_order(self, start_z):
        z_list = []
        z_list.append(start_z)
        offset= 1
        #continue adding on each side of location until both side filled
        while (start_z>= offset or start_z + offset < self.layers):
            if (start_z>=offset): #don't add z<0
                z_list.append(start_z-offset)
            if (start_z +offset < self.layers): #don't add z>self.layers
                z_list.append(start_z+offset)
            offset +=1
        return z_list
        
    def run(self):
        for label_set in self.label_ids:
            color = self.color_map[label_set[0] % len(self.color_map)]
            #color = self.color_map[1]
            for z in self.z_order:
                contours, normals = self.find_contours(label_set, [z])
                if contours != []:
                    self.out_q.put([contours, color, label_set[0], normals])
                    time.sleep(0.001)
        self.test_file.close()
        
    def find_contours(self, label_ids, z_list):
        tot_contours = []
        tot_normals = []
        tile_list = []
        for label in label_ids:
            tile_list += self.get_tile_list(label, z_list)
        tile_list = set(tile_list)
        for w,x, y, z in tile_list:
            tile_files = glob.glob(self.z_folders[z] + "\\*")
            for tile_name in tile_files:
                if os.path.basename(tile_name) == "y={0:08},x={1:08}.hdf5".format(y, x):
                    t_file = h5py.File(tile_name, "r")
                    labels = t_file[self.label_key][...]
                    t_file.close()
                    buffer_array = np.zeros((np.shape(labels)[0]+2, np.shape(labels)[1]+2), np.uint8) #buffer by one pixel on each side
                    
                    for label in label_ids:
                        buffer_array[1:-1, 1:-1] |= labels == label
                    
                    contours, hierarchy  = cv2.findContours(buffer_array, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours == []:
                        blur_mask = sp.filters.gaussian_filter(buffer_array.astype(float),11)
                        dy, dx = np.gradient(blur_mask)
                        
                        contours = [np.array(cnt) for cnt in contours]
                        normals = [np.zeros(cnt.shape) for cnt in contours]
                        for idx, cnt in enumerate(contours):
                            new_cnt = np.zeros((cnt.shape[0], 3))
                            
                            new_cnt[:, 0] = cnt[:, 0, 0] - 1 + x * self.tile_columns
                            new_cnt[:, 1] = cnt[:, 0, 1] - 1 + y*self.tile_rows
                            new_cnt[:, 2] = z
                            
                            new_normal = np.zeros((cnt.shape[0], 3))
                            new_normal[:,0] = -dx[cnt[:,0,1], cnt[:,0,0]]
                            new_normal[:,1] = -dy[cnt[:,0,1], cnt[:,0,0]]
                            #print new_normal
                            #if np.any(new_normal!=0):
                            #    print new_normal
                            #leave z as 0 for now
                            
                            normals[idx] = new_normal
                            contours[idx] = new_cnt
                            
                        tot_normals += normals
                        tot_contours+=contours
        return tot_contours, tot_normals
        
    def get_tile_list(self, label, z_list):
        con = sqlite.connect(self.segment_file)
        cur = con.cursor()
        #w = 0 requirement specifies highest resolution
        cur.execute('select w,x,y,z from idTileIndex where w =' +str(self.w) + ' AND id =' + str(label))
        tile_list = cur.fetchall()
        end_tile_list = []
        for tile in tile_list:
            if tile[3] in z_list:
                end_tile_list += [tile]
        con.close()
        return end_tile_list  