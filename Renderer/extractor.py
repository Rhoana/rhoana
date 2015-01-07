#--------------------
#Contour Extractor
#Daniel Miron
#7/17/2013
#
#Version Date: 7/26 10:30
#--------------------

import sys
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
        color_file = h5py.File(os.path.join(self.directory, "ids", "colorMap.hdf5"), 'r')
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
            for z in self.z_order:
                if self.is_dead:
                    return
                new_contours = self.find_contours(label_set, [z])
                if new_contours != []:
                    self.out_q.put(["contours", new_contours, color, label_set[0], self.idx, z])

    def find_contours(self, label_ids, z_list):
        def get_mask(buffer, labels):
            S = np.sort(list(label_ids) + [np.inf])
            buffer[1:-1, 1:-1] |= (S[S.searchsorted(labels)] == labels)

        def get_contours(buffer):
            contours, hierarchy  = cv2.findContours(buffer, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            return contours

        def get_polys():
            tiles = self.get_tile_list(label_ids[0], z_list)

            for w, x, y, z in tiles:
                for tile_name in glob.glob(os.path.join(self.z_folders[z], "*")):
                    if os.path.basename(tile_name) == "y={0:08},x={1:08}.hdf5".format(y, x):
                        t_file = h5py.File(tile_name, "r")
                        labels = t_file[self.label_key][...]
                        t_file.close()
                        # buffer by one pixel on each side
                        buffer = np.zeros((np.shape(labels)[0]+2, np.shape(labels)[1]+2), np.uint8)
                        get_mask(buffer, labels)

                        if np.any(buffer):
                            for s in get_contours(buffer):
                                s = s.reshape((-1, 2))
                                s[:, 0] += x * self.tile_columns - 1
                                s[:, 1] += y * self.tile_rows - 1
                                yield s
        return list(get_polys())

    def get_tile_list(self, label, z_list):
	sys.stdout.flush()
        con = sqlite.connect(self.segment_file)
        cur = con.cursor()
        tile_list = []
        for z in z_list:
            cur.execute('select w,x,y,z from idTileIndex WHERE w = %d AND id = %d AND z = %d' % (self.w, label, z))
            tile_list += cur.fetchall()
        con.close()
        return tile_list  
