#--------------------
#Contour Extractor
#Daniel Miron
#7/17/2013
#
#Version Date: 7/26 10:30
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
    def __init__(self, out_q, directory, label_ids, location, max_x, max_y, extractor_idx):
        
        self.out_q = out_q #queue read by viewer
        
        self.directory = directory #mojo directory
        tiledir = os.path.join(self.directory, 'ids', 'tiles')
        self.w = len(glob.glob(os.path.join(tiledir, '*'))) - 1 #default to lowest resolution
        self.w_str = "w={0:08}".format(self.w)
        self.label_folder = os.path.join(tiledir, self.w_str)
        
        self.segment_file = os.path.join(self.directory, "ids", "segmentInfo.db")
        self.z_folders = sorted(glob.glob(os.path.join(self.label_folder, "*")))
        h5_file = h5py.File(glob.glob(os.path.join(self.z_folders[0], "*"))[0], "r")
        self.label_key = h5_file.keys()[0]
        self.shape = np.shape(h5_file[self.label_key][...])
        h5_file.close()
        
        self.tile_rows = self.shape[0]
        self.tile_columns = self.shape[1]
        
        self.tiles_per_layer = len(glob.glob(os.path.join(self.z_folders[0], "*")))
        
        self.rows = max_x/pow(2, self.w) - 1
        self.columns = max_y/pow(2, self.w) - 1
        self.layers = int(os.path.basename(str(self.z_folders[-1])).split("0")[-1])+1
        
        self.label_ids = label_ids
        
        self.z_order = self.make_z_order(location[2])
        self.start_z = location[2]
        
        color_file = h5py.File(os.path.join(self.directory, "ids", "colorMap.hdf5"))
        self.color_map = color_file["idColorMap"][...]
        color_file.close()
        
        self.pos_masks = []
        self.neg_masks = []
        
        self.is_dead = False
        
        self.idx = extractor_idx
        
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
        
    def stop(self):
        self.is_dead = True
        
    def run(self):
        for label_set in self.label_ids:
            color = self.color_map[label_set[0] % len(self.color_map)]
            #color = self.color_map[1]
            for z in self.z_order:
                if self.is_dead:
                    return
                new_contours, new_normals, new_mask = self.find_contours(label_set, [z])
                if new_contours != []:
                    self.out_q.put(["contours", new_contours, color, label_set[0], new_normals, self.idx])
                    '''if ((len(self.pos_masks)<=2) and z >= self.start_z):
                        self.pos_masks += [[new_contours, new_normals, new_mask]]
                    elif (z>=self.start_z):
                        contours, normals = self.get_z_normals(1)
                        self.pos_masks = self.pos_masks[1:] + [[new_contours, new_normals, new_mask]] #ascending z order
                        self.out_q.put([contours, color, label_set[0], normals])
                    if ((len(self.neg_masks)<=2) and z<= self.start_z):
                        self.neg_masks += [[new_contours, new_normals, new_mask]]
                    elif (z<=self.start_z):
                        contours, normals = self.get_z_normals(-1)
                        self.neg_masks = self.neg_masks[1:] + [[new_contours, new_normals, new_mask]] #descending z order
                        self.out_q.put([contours, color, label_set[0], normals])'''
        
    def get_z_normals(self, sign):
        if sign >0:
            z_normals = self.pos_masks[0][2]-self.pos_masks[2][2]
            self.test_file.write(str(np.where(z_normals != 0)) + "\n")
            self.test_file.write(str(np.where(self.pos_masks[2][2]!=0)) + "\n\n\n")
            normals = self.pos_masks[1][1]
            contours = self.pos_masks[1][0]
        elif sign < 0:
            z_normals = self.neg_masks[2][2]-self.neg_masks[0][2] #order reversed because of order of cache lists
            #print z_normals
            normals = self.neg_masks[1][1]
            contours = self.neg_masks[1][0]
        for idx, cnt in enumerate(contours[0]):
            normals[0][idx][2] = z_normals[cnt[1], cnt[0]]
        return contours, normals
        
    def find_contours(self, label_ids, z_list):
        tot_contours = []
        tot_normals = []
        tile_list = []
        tot_mask = np.zeros((self.tile_rows, self.tile_columns))
        for label in label_ids:
            tile_list += self.get_tile_list(label, z_list)
        tile_list = set(tile_list)
        for w,x, y, z in tile_list:
            tile_files = glob.glob(os.path.join(self.z_folders[z], "*"))
            for tile_name in tile_files:
                if os.path.basename(tile_name) == "y={0:08},x={1:08}.hdf5".format(y, x):
                    t_file = h5py.File(tile_name, "r")
                    labels = t_file[self.label_key][...]
                    t_file.close()
                    buffer_array = np.zeros((np.shape(labels)[0]+2, np.shape(labels)[1]+2), np.uint8) #buffer by one pixel on each side

                    for label in label_ids:
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
                            #norms/=np.sqrt(np.sum(norms**2, axis = 1))
                            strip[:, 0] += x * self.tile_columns - 1
                            strip[:, 1] += y * self.tile_rows - 1
                            strip[:, 2] = z

                        for strip in tristrips:
                            tmp_n = np.zeros((len(strip), 3), np.float)

                        tot_contours += tristrips
                        tot_normals += normals
        return tot_contours, tot_normals, tot_mask
        
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
