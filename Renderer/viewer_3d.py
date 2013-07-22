#-------------------------
#3d Renderer
#Daniel Miron
#7/5/2013
#
#Allows 3d viewing of nerve cord or neuron stacks.
#Includes ability to fully rotate image in 3 dimensions and to mark locations in 3-space
#-------------------------

import sys
sys.path.append(r'c:\Python27\Lib\site-packages')
sys.path.append('.')
import h5py
import numpy as np
import glob
import os

import pickle
import math
import re
import time
import threading
from Queue import Queue
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
import arcball as arc

from pysqlite2 import dbapi2 as sqlite

import cv2
import select

import extractor
import input_handler as handler

from ctypes import util
try:
    from OpenGL.platform import win32
except AttributeError:
    pass

class Viewer:
    def __init__(self, location, in_q, directory, max_x, max_y):
        self.st = time.time()
        self.arcball = None
        
        self.directory = directory
        self.in_q =in_q
        self.max_x = max_x
        self.max_y = max_y
        
        self.rows = 0
        self.columns = 0
        self.layers = 0
        
        self.win_h = 0
        self.win_w = 0
        
        self.left = None
        self.slice = None
        self.pick_location = location
        
        self.display_list_idx = 1
        self.marker_color = [1., 1., 1.] #initial marker is white
        
        
        
    def set_dimensions(self, rows, columns, layers, w):
        self.rows = rows
        self.columns = columns
        self.layers = layers
        self.pick_location = (self.pick_location[0]/pow(2, w) - 1, self.pick_location[1]/pow(2, w) - 1, self.pick_location[2])
        
    def main(self):
        #self.contours = self.load_contours(contour_file)
        #set window height and width
        self.win_h = 1000
        self.win_w = 1000
        self.arcball = self.create_arcball()
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(self.win_w, self.win_h) #width, height
        glutCreateWindow("Nerve Cord")
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(65, 1, 1, 10)
        self.fov = 65
        glMatrixMode(GL_MODELVIEW)
        
        self.back_tesselator = gluNewTess()
        gluTessCallback(self.back_tesselator, GLU_TESS_BEGIN, glBegin)
        gluTessCallback(self.back_tesselator, GLU_TESS_END, glEnd)
        gluTessCallback(self.back_tesselator, GLU_TESS_VERTEX, self.back_vertex)
        
        self.front_tesselator = gluNewTess()
        gluTessCallback(self.front_tesselator, GLU_TESS_BEGIN, glBegin)
        gluTessCallback(self.front_tesselator, GLU_TESS_END, glEnd)
        gluTessCallback(self.front_tesselator, GLU_TESS_VERTEX, self.front_vertex) 
        
        glEnable(GL_DEPTH_TEST)
        
        #self.make_display_lists()
        glutDisplayFunc(self.draw)
        glutKeyboardFunc(self.keyboard)
        glutMouseFunc(self.on_click)
        glutMotionFunc(self.on_drag)
        glutMouseWheelFunc(self.on_scroll)
        glutIdleFunc(self.on_idle)
        glutReshapeFunc(self.on_resize)
        
        glutMainLoop()
        return          
        
    def on_resize(self, w, h):
        self.win_h = h
        self.win_w = w
        glViewport(0,0, w,h)
        self.arcball = self.create_arcball()
            
    def on_idle(self):
            while(not self.in_q.empty()):
                temp = self.in_q.get()
                if type(temp) is list:
                    #draw contours
                    contours =  temp[0]
                    color = temp[1]
                    self.make_display_lists(contours, color/255.0)
                    self.draw()
                else:
                    #add new extractor
                    args = temp.split()
                    location = (int(args[0]), int(args[1]), int(args[2]))
                    
                    primary_id = []
                    secondary_ids = []
                    split_str = re.split(":", args[3])
                    primary_id = [int(split_str[0])]
                    if split_str[1] != "":
                        secondary_ids = [int(label) for label in re.split(',', split_str[1])]
                    ids = [primary_id + secondary_ids]

                    extr = extractor.Extractor(self.in_q, self.directory, ids, location, max_x, max_y)
                    extracting_worker = threading.Thread(target = extr.run, name = "extr")
                    extracting_worker.daemon = True
                    extracting_worker.start()
        
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
        return end_tile_list
        
    def create_arcball(self):
        arcball = arc.Arcball()
        #locate the arcball center at center of window with radius half the width
        arcball.place([self.win_w/2, self.win_h/2], self.win_w/2)
        return arcball
        
    def make_display_lists(self, contours, color):
        self.display_lists = glGenLists(2) #first list for front, second for back
        
        self.make_front_list(contours, color)
        self.make_back_list(contours)
        self.display_list_idx +=2
        
    def make_back_list(self, contours):
        '''Creates a display list to encode color for image. Not seen by user'''
        glNewList(self.display_list_idx+1, GL_COMPILE)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glTranslatef(-.9, .9, .9)
        glScalef(1.8/self.columns, -1.8/self.rows, -1.8/self.layers)
        #draw the layers
        for cnt in contours:
            gluTessBeginPolygon(self.back_tesselator, None)
            gluTessBeginContour(self.back_tesselator)
            for vtx in cnt:
                gluTessVertex(self.back_tesselator, vtx, vtx)
            gluTessEndContour(self.back_tesselator)
            gluTessEndPolygon(self.back_tesselator)
        
        glColor3f(.5, .5, .5)
        glBegin(GL_POLYGON)
        glVertex3f(*self.x_axis[2][0])
        glVertex3f(*self.x_axis[2][1])
        glVertex3f(*self.x_axis[3][1])
        glVertex3f(*self.x_axis[3][0])
        glEnd()
        
        glPopMatrix()
        
        glEndList()
        
    def make_front_list(self, contours, color):
        '''Creates a display list to draw a box and the data scaled to .9*the size of the window.
        This list deals with the display seen by the user'''
        
        glNewList(self.display_list_idx, GL_COMPILE)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glTranslatef(-.9, .9, .9)
        glScalef(1.8/self.columns, -1.8/self.rows, -1.8/self.layers)
        
        #draw the layers
        glColor3f(*color)
        for cnt in contours:
            gluTessBeginPolygon(self.front_tesselator, None)
            gluTessBeginContour(self.front_tesselator)
            for vtx in cnt:
                gluTessVertex(self.front_tesselator, vtx, vtx)
            gluTessEndContour(self.front_tesselator)
            gluTessEndPolygon(self.front_tesselator)
            
        #make a box around the image
        self.axes()
        self.make_box()
        
        glPopMatrix()
        
        glEndList()
        
    def make_box(self):
        glBegin(GL_LINES)
        glColor3f(1.0, 0, 0) #x in red
        for line in self.x_axis:
            glVertex3f(line[0][0], line[0][1], line[0][2])
            glVertex3f(line[1][0], line[1][1], line[1][2])
        glColor3f(0,1.0, 0) #y in green
        for line in self.y_axis:
            glVertex3f(line[0][0], line[0][1], line[0][2])
            glVertex3f(line[1][0], line[1][1], line[1][2])
        glColor3f(0,0,1.0) #z in blue
        for line in self.z_axis:
            glVertex3f(line[0][0], line[0][1], line[0][2])
            glVertex3f(line[1][0], line[1][1], line[1][2])
        glEnd() 
                
        glColor3f(0.5, 0.5, 0.5)
        
        #10.0 gives reasonable font size
        glScalef(10.0/self.columns, -10.0/self.rows, -10.0/self.layers)
        glutStrokeString(GLUT_STROKE_ROMAN, "(0,0,0)")
        glTranslatef(self.columns*self.columns/10.8, 0,0)
        glutStrokeString(GLUT_STROKE_ROMAN, "x=" + str(self.columns))
        glTranslatef(-self.columns*self.columns/10.0, -self.rows*self.rows/10.0, 0)
        glutStrokeString(GLUT_STROKE_ROMAN, "y=" + str(self.rows))
        glTranslatef(0, self.rows*self.rows/10.0, -self.layers*self.layers/10.0)
        glutStrokeString(GLUT_STROKE_ROMAN, "z=" + str(self.layers))
        
    def front_vertex(self, vertex):
        glVertex3f(*vertex)
        
    def back_vertex(self, vertex):
        '''sets the color of a single vertex and draws it'''
        #scale by dim-1 to include black 
        #multiply by -1 and add 1 to invert color axis
        glColor3f(1.0*vertex[0]/(self.columns-1), -1.0*vertex[1]/(self.rows-1)+1.0, -1.0*vertex[2]/(self.layers-1)+1.0)
        glVertex3f(*vertex)
        
    def draw(self, pick=False):
        '''draws an image'''
        
        glMatrixMode(GL_MODELVIEW)
        glDrawBuffer(GL_BACK)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluLookAt(0, 0, 3, 0, 0, 2, 0,1,0)
        glMultMatrixd(self.arcball.matrix().T)
        
        if not pick:
            #odd numbers for display
            for idx in range(1, self.display_list_idx+1, 2):
                glCallList(idx)
            self.draw_marker()
            glutSwapBuffers()
        else:
            #even numbers for picking
            for idx in range(2, self.display_list_idx+1, 2):
                glCallList(idx)
            glFlush()
                
      
        return
        
    def draw_marker(self):
        '''Draws a sphere around the chosen point. Color is inverse of chosen pixel'''
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        location = self.pick_location
        glTranslatef(float(1.8*location[0])/self.columns-.9,
                    -(float(1.8*location[1])/self.rows-.9),
                    -(float(1.8*location[2])/self.layers-.9))
        glColor3f(*self.marker_color)
        glutSolidSphere(.01, 50, 50)
        
        glPopMatrix()
        
    def keyboard(self, key, x, y):
        if key == chr(27): #escape to quit
            sys.exit()
        return
        
    def on_scroll(self, wheel, direction, x, y):
        '''zooms in and out on mouse scroll wheel'''
        if direction == 1:
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            self.fov = self.fov - 1
            gluPerspective(self.fov, 1, 1, 10)
            self.draw()
        else:
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            self.fov = self.fov + 1
            gluPerspective(self.fov, 1, 1, 10)
            self.draw()
            return
        
        
    def on_click(self, button, state, x, y):
        #Left click for arcball rotation
        if (button == GLUT_LEFT_BUTTON and state == GLUT_DOWN):
            self.left = True #turn on dragging rotation
            self.arcball.down((x,y))
        #right click to select a pixel location
        elif (button == GLUT_RIGHT_BUTTON and state == GLUT_DOWN):
            self.left = False #turn off dragging rotation
            self.draw(pick=True)
            self.pick_location, self.marker_color = self.pick(x,y)
            print self.pick_location
            sys.stdout.flush()
            self.has_marker = True
    
    def pick(self, x,y):
        '''gets the (x,y,z) location in the full volume of a chosen pixel'''
        click_color = None
        glReadBuffer(GL_BACK)
        click_color = glReadPixels(x,self.win_h-y, 1,1, GL_RGB, GL_FLOAT)[0][0]
        location  = [int(click_color[0]*(self.columns-1)), 
                    int(-(click_color[1]-1)*(self.rows-1)), int(-(click_color[2]-1)*((self.layers-1)))]
        glReadBuffer(GL_FRONT)
        marker_color_neg = glReadPixels(x,self.win_h-y, 1,1, GL_RGB, GL_FLOAT)[0][0]
        marker_color = 1-marker_color_neg
        return location, marker_color
        
    def on_drag(self, x, y):
        '''rotates image on dragging with left mouse down'''
        if self.left:
            self.arcball.drag((x,y))
            self.draw()
        
    def read_chunk_map(self, chunk_file):
        return pickle.load(open(chunk_file, "rb"))
        
    def save_contours(self, contour_file):
        pickle.dump(self.contours, open(contour_file, "wb"))
        return
        
    def load_contours(self, contour_file):
        return pickle.load(open(contour_file, "rb"))
        
    def axes(self):
        '''generates vertices for a box'''
        self.x_axis = [[[0,0,0], [self.columns, 0,0]], [[0,self.rows,0], [self.columns, self.rows, 0]],
                        [[0,0,self.layers], [self.columns,0,self.layers]], [[0, self.rows, self.layers], [self.columns, self.rows, self.layers]]]
        self.y_axis = [[[0,0,0], [0, self.rows,0]], [[self.columns,0,0],[self.columns, self.rows, 0]], [[0,0,self.layers], [0,self.rows, self.layers]],
                        [[self.columns, 0, self.layers],[self.columns, self.rows, self.layers]]]
        self.z_axis = [[[0,0,0], [0,0,self.layers]], [[self.columns,0,0],[self.columns, 0, self.layers]],
                        [[0, self.rows,0], [0, self.rows, self.layers]],[[self.columns, self.rows, 0],[self.columns, self.rows, self.layers]]]
          

if __name__ == '__main__':
    display_queue = Queue()
    sys.argv.pop(0)
    
    #extract command line arguments
    directory = sys.argv[0]
    
    location = (int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])) #x,y,z
    #resolution_level = int(sys.argv[4]) #w
    max_x =int(sys.argv[4])
    max_y = int(sys.argv[5])
    
    ids = []
    for label_set in (sys.argv[6:len(sys.argv)]):
        primary_id = []
        secondary_ids = []
        split_str = re.split(":", label_set)
        primary_id = [int(split_str[0])]
        if split_str[1] != "":
            secondary_ids = [int(label) for label in re.split(',', split_str[1])]
        ids += [primary_id + secondary_ids]
        
    extr = extractor.Extractor(display_queue, directory, ids, location, max_x, max_y)
    viewer  = Viewer(location, display_queue, directory, max_x, max_y)
    handler = handler.Input_Handler(display_queue)
    
    viewer.set_dimensions(extr.rows, extr.columns, extr.layers, extr.w)
    
    extracting_worker = threading.Thread(target = extr.run, name = "extr")
    input_worker = threading.Thread(target = handler.run, name = "input_worker")
    
    input_worker.daemon =True
    extracting_worker.daemon = True
    extracting_worker.start()
    input_worker.start()
    
    viewer.main()