#---------------------
#ChunkMap Maker
#Daniel Miron
#7/9/2013
#
#--------------------

import h5py
import numpy as np
import time
import math
import sys
import pickle


class Chunk_Map:
    def __init__(self, label_file):
        self.label_file = h5py.File(label_file, 'r')

        self.ds = self.label_file["main"]

        self.rows = self.ds.shape[0]
        self.columns = self.ds.shape[1]
        self.layers = self.ds.shape[2]
        
        '''self.chunk_rows = self.ds.chunks[0]
        self.chunk_columns = self.ds.chunks[1]
        self.chunk_layers = self.ds.chunks[2]'''
        
        self.chunk_rows = 64
        self.chunk_columns = 64
        self.chunk_layers = 16
        self.chunk_map = self.make_map()
        
    def make_map(self):
        st = time.time()
        
        #format is {label:[(row1, column1, layer1),(row2, column2, layer2)]}
        #indices are for start of chunk 
        chunk_map = dict()
        #for r_chunk_n in range(20):
        for r_chunk_n in range(int(math.ceil(self.rows/self.chunk_rows))):
            print time.time()-st
            start_r = r_chunk_n*self.chunk_rows
            end_r = start_r + self.chunk_rows
            #for c_chunk_n in range(20):
            for  c_chunk_n in range(int(math.ceil(self.columns/self.chunk_columns))):
                start_c = c_chunk_n*self.chunk_columns
                end_c = start_c + self.chunk_columns
                for l_chunk_n in range(int(math.ceil(self.layers/self.chunk_layers))):
                    start_l = l_chunk_n*self.chunk_layers
                    end_l = start_l +self.chunk_layers
                    chunk = self.ds[start_r:end_r, start_c:end_c, start_l:end_l][...]
                    unique_labels = np.unique(chunk)
                    for label in unique_labels:
                        if not label ==0:
                            if chunk_map.has_key(label):
                                chunk_map[label] = chunk_map[label] + \
                                                    [(start_r, start_c, start_l)]
                            else:
                                chunk_map[label] = [(start_r, start_c, start_l)]
        return chunk_map
        
    def save_pickle(self, chunk_file):
        pickle.dump(self.chunk_map, open(chunk_file, "wb"))
        
        
chunk_map = Chunk_Map(r'C:\Users\DanielMiron\Documents\3d_rendering\labels_full.h5')
chunk_map.save_pickle(r'C:\Users\DanielMiron\Documents\3d_rendering\label_full_chunk_map.p')
