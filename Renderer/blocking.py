#-------------------------
#3d Renderer
#Daniel Miron
#7/5/2013
#
#-------------------------

import h5py
import numpy as np
import math
import time
import sys
sys.path.append(r'c:\Python27\Lib\site-packages')
import pickle
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
import arcball as arc

import cv2

class Viewer:
    def __init__(self, label_file, chunk_file):
        self.arcball = self.create_arcball()
        self.label_file = h5py.File(label_file, 'r')

        self.ds = self.label_file["labels"]

        self.rows = self.ds.shape[0]
        self.columns = self.ds.shape[1]
        self.layers = self.ds.shape[2]
        
        self.chunk_rows = self.ds.chunks[0]
        self.chunk_columns = self.ds.chunks[1]
        self.chunk_layers = self.ds.chunks[2]
        
        self.chunk_map = self.read_chunk_map(chunk_file)
        
        self.keys = 0
        self.rotation_x = 0
        self.rotation_y = 0
        
        self.contours = []
        
        self.rotate = False
        self.rotation_mat = None
        self.rotation_list = []
        
    def create_arcball(self):
        arcball = arc.Arcball()
        arcball.place([500,500], 500)
        #arcball.setaxes([1,1,0], [-1,1,0])
        return arcball
        
    def scale(self):
        for cnt in self.contours:
            for vtx in cnt:
                #scale by 1.8 to go from -.9 to .9 with .1 buffer room on each side
                vtx[0] = 1.8*(float(vtx[0])/float(self.rows)-0.5)
                vtx[1] = -1.8*(float(vtx[1])/float(self.columns)-0.5) #-1.8 flips about x to correspond to hdf5 orientation
                vtx[2] = 1.8*(float(vtx[2])/float(self.layers)-0.5)
        return
        
    def make_display_list(self):
        self.display_list = glGenLists(1)
        glNewList(self.display_list, GL_COMPILE) 
        for cnt in self.contours:
            glBegin(GL_LINE_LOOP)
            for vtx in cnt:
                glVertex3f(vtx[0], vtx[1], vtx[2])
            glEnd()
        glEndList()
        
        
    def draw(self):
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        #if self.rotate:
        #    glMultMatrixd(self.rotation_mat)


        '''storing the matrices and multiplying them in the order from most recent first, to oldest last
        seems to fix the composition problem. rotation about z-axis is still backwards'''

        if self.rotate:
            glMultMatrixd(self.rotation_mat) 
            #for mat in self.rotation_list:
            for mat in reversed(self.rotation_list):
                glMultMatrixd(mat)
            #glMultMatrixd(self.rotation_mat)

        glCallList(self.display_list)
        glutSwapBuffers()
        return
        
    def on_click(self, button, state, x, y):
        if (button == GLUT_LEFT_BUTTON and state == GLUT_DOWN):
            self.arcball = self.create_arcball()
            self.arcball.down((x,y))
        elif(button==GLUT_LEFT_BUTTON and state == GLUT_UP):
            self.rotation_list.append(self.rotation_mat)
    
    def on_drag(self, x, y):
        self.arcball.drag((x,y))
        self.rotation_mat = self.arcball.matrix()
        self.rotate = True
        self.draw()
    
    def main(self, keys):
        self.keys = keys 
        self.st = time.time()  
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(1000, 1000) #width, height
        glutCreateWindow("Nerve Cord")
        self.make_display_list()
        glutDisplayFunc(self.draw)
        glutMouseFunc(self.on_click)
        glutMotionFunc(self.on_drag)
        glutMainLoop()
        return
        
        
    def get_contours(self, keys):
        chunk_list = self.organize_chunks(keys)
        for chunk in chunk_list:
            for layer in reversed(range(self.chunk_layers)):
                for key in keys:
                    if not layer+chunk[2]>= self.layers: #make sure we stay within bounds
                        labels = self.ds[chunk[0]:chunk[0]+self.chunk_rows, chunk[1]:chunk[1]+self.chunk_columns, chunk[2]+layer][...]
                        labels[labels!=key] = 0
                        labels[labels==key] = 255
                        labels = labels.astype(np.uint8)
                        buffer_array = np.zeros((np.shape(labels)[0] +2, np.shape(labels)[1]+2), np.uint8) #buffer by one pixel on each side
                        buffer_array[1:-1, 1:-1] = labels 
                        contours, hierarchy = cv2.findContours(buffer_array, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        if not contours==[]:
                            contours_3d = []
                            for cnt in contours:
                                cnt_3d = []
                                for vtx in cnt:
                                    cnt_3d += [[vtx[0][1]-1+chunk[0],vtx[0][0]-1+chunk[1], layer+chunk[2]]] #subtract 1 to adjust back after buffer
                                contours_3d += [cnt_3d]
                            self.contours +=contours_3d
        self.scale()
        
        
    def organize_chunks(self, keys):
        chunk_list = []
        for key in keys:
            chunk_list += self.chunk_map[key]
        chunk_list.sort(key=lambda x: x[2]) #sort w/respect to z
        chunk_list.reverse() #make back to front
        return chunk_list
        
    def read_chunk_map(self, chunk_file):
        return pickle.load(open(chunk_file, "rb"))
        

viewer = Viewer(r'C:\Users\DanielMiron\Documents\3d_rendering\labels.hdf5',
                r'C:\Users\DanielMiron\Documents\3d_rendering\label_chunk_map.p')