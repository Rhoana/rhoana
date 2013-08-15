#---------------------
#Formats labels in pickled files to text files to be read by webgl viewer
#Daniel Miron
#8/1/2013
#
#--------------------

import pickle
import json
import glob
import numpy as np
import h5py
import os

directory = r'C:\MojoData\RDExtendLeft\RDExtendLeft\mojo'

cnt_files = glob.glob(directory + r"\contours\*")
cnt_files.sort(key = len, reverse=False) #handle unpadded file names
normal_files = glob.glob(directory+ r"\normals\*")
normal_files.sort(key = len, reverse=False) #handle unpadded file names


color_file = h5py.File(os.path.join(directory, "ids", "colorMap.hdf5"))
color_map = color_file["idColorMap"][...]
color_file.close()
label = 140314

out_file = open(r'C:\MojoData\RDExtendLeft\RDExtendLeft\mojo\web_contours\label_' + str(label).zfill(8) + '.txt', 'w')
out_file.write(str(color_map[label%len(color_map)]) + "\n")

for i, (cnt, normal) in enumerate(zip(cnt_files, normal_files)):
    #if i in range(299, 300): #select specific layers (when commented out all layers are included)
        cnt_f = open(cnt, "rb")
        contours = pickle.load(cnt_f)
        cnt_f.close()
        if label in contours.keys():
            for array in contours[label]:
                for vtx in array:
                    #hardcoded numbers only for testing-should be read from files as dimensions of dataset
                    x = 2*(vtx[0]/511.0)-1
                    y = -2*(vtx[1]/511.0)+1
                    z = -2*(vtx[2]/300.0)+1
                    out_file.write("{: f}".format(x) + " " + "{: f}".format(y) + " " + "{: f}".format(z) + " ")
                out_file.write("\t")
        out_file.write('\n')
        normal_f = open(normal, "rb")
        normals = pickle.load(normal_f)
        normal_f.close()
        if label in normals.keys():
            for array in normals[label]:
                for vtx in array:
                    vtx /= np.linalg.norm(vtx)
                    for value in vtx:
                        out_file.write("{: f}".format(value) + " ")
                out_file.write("\t")
        out_file.write('\n')
        out_file.flush()
    