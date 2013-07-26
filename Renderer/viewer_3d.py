#-------------------------
#3d Renderer
#Daniel Miron
#7/5/2013
#
#Allows 3d viewing of nerve cord or neuron stacks.
#Includes ability to fully rotate image in 3 dimensions and to mark locations in 3-space
#
#Version Date: 7/25 5:00
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
try:
    from OpenGL.GLUT.freeglut import *
except Exception:
    pass
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
        self.win_h = 1000
        self.win_w = 1000
        self.arcball = self.create_arcball()
        
        self.directory = directory
        self.in_q =in_q
        self.max_x = max_x #highest resolution
        self.max_y = max_y #highest resolution
        
        self.rows = 0
        self.columns = 0
        self.layers = 0
        
        self.fov = 60
        self.aspect = float(self.win_w)/self.win_h
        
        self.left = None #keep track of left button status
        self.pick_location = location
        
        self.display_list_idx = 2 #count from 1 and use first index for box
        self.display_list_dict = dict() #COLOR as key, display_list indices as value
        self.marker_color = [1., 1., 1.] #initial marker is white
        
        self.first = True #used to control display list flow
        self.icon_color = np.array((0.0, 1.0, 0.0))
        self.st = time.time()
        
        self.center_x = 0
        self.center_y = 0
        
        self.extractor_dict = dict() #keys are indices, values are extractor threads
        self.make_lists = True
        
        self.num_labels = 0
        self.label_dict = dict() #keys are float indices, values are labels
        
    def set_dimensions(self, rows, columns, layers, w):
        '''sets the dimensions of the viewing box'''
        self.rows = rows
        self.columns = columns
        self.layers = layers
        self.pick_location = (self.pick_location[0]/pow(2, w) - 1, self.pick_location[1]/pow(2, w) - 1, self.pick_location[2])
        
    def main(self):
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_ALPHA)
        glutInitWindowSize(self.win_w, self.win_h) #width, height
        glutCreateWindow("3D View")
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fov, self.aspect, 1, 10)
        glMatrixMode(GL_MODELVIEW)
        
        #tesselator for back color encoding
        self.back_tesselator = gluNewTess()
        gluTessCallback(self.back_tesselator, GLU_TESS_BEGIN, glBegin)
        gluTessCallback(self.back_tesselator, GLU_TESS_END, glEnd)
        gluTessCallback(self.back_tesselator, GLU_TESS_VERTEX, self.back_vertex)
        
        #tesselator for actual drawing
        self.front_tesselator = gluNewTess()
        gluTessCallback(self.front_tesselator, GLU_TESS_BEGIN, glBegin)
        gluTessCallback(self.front_tesselator, GLU_TESS_END, glEnd)
        gluTessCallback(self.front_tesselator, GLU_TESS_VERTEX, self.front_vertex) 
        
        glShadeModel(GL_SMOOTH)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_NORMALIZE)
        glColorMaterial(GL_FRONT_AND_BACK, GL_EMISSION)
        glEnable(GL_COLOR_MATERIAL)
        
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
        '''resize the viewing window'''
        self.win_h = h
        self.win_w = w
        glViewport(0,0, w,h)
        self.arcball.place([self.win_w/2, self.win_h/2], self.win_w/2)
    
    def translate(self, x, y):
        '''translate the viewing box based on mouse location'''
        self.center_x = self.center_x+((float(x)/self.win_w)-.5)*2
        self.center_y = self.center_y-((float(y)/self.win_h)-.5)*2
    
    def shift(self, key):
        '''translate the viewing box based on keyboard input'''
        if key == chr(105):
            self.center_y +=float(self.fov)**2/10000
        elif key == chr(106):
            self.center_x +=float(self.fov)**2/10000
        elif key == chr(107):
            self.center_y -= float(self.fov)**2/10000
        elif key == chr(108):
            self.center_x -=float(self.fov)**2/10000
            
    def reset_translation(self):
        '''reset the viewing box to 0 translation'''
        self.center_x = 0
        self.center_y = 0
        glutPostRedisplay()
        
    def reset_zoom(self):
        '''reset the zoom level'''
        self.fov = 60
        glutPostRedisplay()
    
    def reset(self):
        self.reset_translation()
        self.reset_zoom()
            
    def on_idle(self):
        while(not self.in_q.empty()):
            self.icon_color = (self.icon_color + .01)%1 #resets to black when icon is green since 1.0 and 0.0 %1 are equal
            temp = self.in_q.get()
            if temp[0] == "marker":
                self.pick_location = temp[1:][0]
                self.pick_location[0] = int(float(self.pick_location[0]*self.columns)/self.max_x)
                self.pick_location[1] = int(float(self.pick_location[1]*self.rows)/self.max_y)
            elif temp[0] == "ids":
                self.num_labels += 1
                label_idx = self.num_labels
                self.label_dict[label_idx] = temp[1:][0][0]
                extr = extractor.Extractor(self.in_q, self.directory, temp[1:][0], self.pick_location, self.max_x, self.max_y)
                self.extractor_dict[temp[1][0][0]] = extr
                extracting_worker = threading.Thread(target = extr.run, name = "extr")
                extracting_worker.daemon = True
                extracting_worker.start()
            elif temp[0] == "contours":
                contours =  temp[1]
                color = temp[2]
                primary_label = temp[3]
                normals = temp[4]
                if self.make_lists:
                    self.make_display_lists(contours, color/255.0, primary_label, normals)
            elif temp[0] == "limits":
                self.max_x= temp[1]
                self.max_y = temp[2]
                self.layers = temp[3]
            elif temp[0] == "refresh":
                self.refresh()
            elif temp[0] == "remove":
                self.remove_label(temp[1:][0])
            self.st = time.time()
            glutPostRedisplay()
        #set icon to green if processes are done
        if time.time()-self.st > 0.25:
            self.icon_color = np.array((0.0, 1.0, 0.0))
            self.make_lists = True
            glutPostRedisplay()

                
    def loading_icon(self):
        glBegin(GL_QUADS)
        glVertex3f(-1.0, -1.0, 1.0)
        glVertex3f(-.95, -1.0, 1.0)
        glVertex3f(-.95, -.95, 1.0)
        glVertex3f(-1.0, -.95, 1.0)
        glEnd()
    
    def refresh(self):
        '''Clears all contours and deletes working extractors'''
        #first display list is for the box
        self.num_labels = 0
        glDeleteLists(2, self.display_list_idx)
        self.in_q.queue.clear()
        self.make_lists = False
        self.display_list_idx = 2
        for key in self.extractor_dict.keys():
            self.extractor_dict[key].stop()
        glutPostRedisplay()
    
    def undo(self):
        label = self.display_list_dict.keys()[0]
        for display_list in self.display_list_dict[label]:
            glDeleteLists(display_list, 1) #delete back and front lists
            glutPostRedisplay()
            
    def remove_label(self, ids):
        '''remove a single contour'''
        for display_list in self.display_list_dict[ids[0]]:
            glDeleteLists(display_list, 1)
            self.extractor_dict[ids[0]].stop()
            glutPostRedisplay()
        
    def create_arcball(self):
        arcball = arc.Arcball()
        #locate the arcball center at center of window with radius half the width
        arcball.place([self.win_w/2, self.win_h/2], self.win_w/2)
        return arcball
        
    def make_display_lists(self, contours, color, label, normals):
        '''Generates display lists to draw both the front and back buffered images'''
        if self.first: #make the box 
            display_list = glGenLists(1)
            self.axes()
            self.make_box_list()
            self.first = False
        self.display_lists = glGenLists(2) #first list for front, second for back
        if label in self.display_list_dict:
            self.display_list_dict[label] = self.display_list_dict[label] + [self.display_list_idx, self.display_list_idx+1] 
        else:
            self.display_list_dict[label] = [self.display_list_idx, self.display_list_idx+1]
        self.make_front_list(contours, color, normals)
        label_idx = self.num_labels
        sys.stdout.flush()
        self.make_back_list(contours, label_idx)
        self.display_list_idx +=2
        
    def make_back_list(self, contours, label_idx):
        '''Creates a display list to encode color for image. Not seen by user'''
        glNewList(self.display_list_idx+1, GL_COMPILE)
        glDisable(GL_LIGHTING) #don't use lighting for color encoding
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glTranslatef(-.9, .9, .9)
        glScalef(1.8/self.columns, -1.8/self.rows, -1.8/self.layers)
        #draw the layers

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        for cnt in contours:
            colors = np.zeros((cnt.shape[0], 4), np.float)
            colors[:, :3] = cnt
            colors[:, 0] /= self.columns - 1
            colors[:, 1] /= - (self.rows - 1)
            colors[:, 2] /= - (self.layers - 1)
            colors[:, 1:3] += 1
            colors[:, 3] = label_idx
            glVertexPointer(3, GL_INT, 0, cnt)
            glColorPointer(4, GL_FLOAT, 0, colors)
            glDrawArrays(GL_TRIANGLE_STRIP, 0, cnt.shape[0])
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

        glPopMatrix()

        glEnable(GL_LIGHTING)

        glEndList()

    def make_front_list(self, contours, color, normals):
        '''Creates a display list to draw a box and the data scaled to .9*the size of the window.
        This list deals with the display seen by the user'''
        
        glNewList(self.display_list_idx, GL_COMPILE)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glTranslatef(-.9, .9, .9)
        glScalef(1.8/self.columns, -1.8/self.rows, -1.8/self.layers)
        
        #draw the layers
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        for cnt, normal in zip(contours, normals):
            colors = np.zeros_like(normal)
            colors[...] = np.array(color)
            colors[cnt[:, 0] == self.columns, :] = np.array([1, 0, 0])
            colors[cnt[:, 0] == 0, :] = np.array([1, 0, 0])
            colors[cnt[:, 1] == self.rows, :] = np.array([0, 1, 0])
            colors[cnt[:, 1] == 0, :] = np.array([0, 1, 0])
            colors[cnt[:, 2] == self.layers - 1, :] = np.array([0, 1, 1])
            colors[cnt[:, 2] == 0, :] = np.array([0, 1, 1])
            glVertexPointer(3, GL_INT, 0, cnt)
            glNormalPointer(GL_FLOAT, 0, normal)
            glColorPointer(3, GL_FLOAT, 0, colors)
            glDrawArrays(GL_TRIANGLE_STRIP, 0, cnt.shape[0])
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

        glPopMatrix()
        glEndList()

    def make_box_list(self):
        '''makes a display list to draw the box'''
        glNewList(1, GL_COMPILE)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        
        #make a box around the image
        glBegin(GL_LINES)
        glColor3f(1.0, 0, 0) #x in red
        glVertex3f(-.9, -.9, -.9)
        glVertex3f(.9, -.9, -.9)
        glVertex3f(-.9, .9, -.9)
        glVertex3f(.9, .9, -.9)
        glVertex3f(-.9, -.9, .9)
        glVertex3f(.9, -.9, .9)
        glVertex3f(-.9, .9, .9)
        glVertex3f(.9, .9, .9)
        glColor3f(0,1.0, 0) #y in green
        glVertex3f(-.9, -.9, -.9)
        glVertex3f(-.9, .9, -.9)
        glVertex3f(.9, -.9, -.9)
        glVertex3f(.9, .9, -.9)
        glVertex3f(-.9, .9, .9)
        glVertex3f(-.9, -.9, .9)
        glVertex3f(.9, -.9, .9)
        glVertex3f(.9, .9, .9)
        glColor3f(0,0,1.0) #z in blue
        glVertex3f(-.9, -.9, -.9)
        glVertex3f(-.9, -.9, .9)
        glVertex3f(.9, -.9, -.9)
        glVertex3f(.9, -.9, .9)
        glVertex3f(-.9, .9, -.9)
        glVertex3f(-.9, .9, .9)
        glVertex3f(.9, .9, -.9)
        glVertex3f(.9, .9, .9)
        glEnd()
                
        glColor3f(0.5, 0.5, 0.5)
        
        glRasterPos3f(-.9, .9, .9)
        glutBitmapString(GLUT_BITMAP_TIMES_ROMAN_24, "(0,0,0)")
        glRasterPos3f(.9, .9, .9)
        glutBitmapString(GLUT_BITMAP_TIMES_ROMAN_24, "x=" + str(self.columns-1))
        glRasterPos3f(-.9, -.9, .9)
        glutBitmapString(GLUT_BITMAP_TIMES_ROMAN_24, "y= " + str(self.rows-1))
        glRasterPos3f(-.9, .9, -.9)
        glutBitmapString(GLUT_BITMAP_TIMES_ROMAN_24, "z= " + str(self.layers-1))
        
        glPopMatrix()
        glEndList()
        
    def front_vertex(self, vert_norm):
        '''draws a vertex for the front image'''
        #if the vertex is an edge vertex color it differently
        if (vert_norm[0][0] == self.columns or vert_norm[0][0]==0):
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, (1,0,0))
        elif (vert_norm[0][1] == self.rows or vert_norm[0][1] == 0):
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, (0,1,0))
        elif (vert_norm[0][2] == self.layers-1 or vert_norm[0][2] == 0):
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, (0,0,1))
        else:
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, vert_norm[2])
        glNormal3fv(vert_norm[1])
        glVertex3fv(vert_norm[0])
        
    def back_vertex(self, vert_label):
        '''sets the color of a single vertex and draws it'''
        #scale by dim-1 to include black 
        #multiply by -1 and add 1 to invert color axis
        vertex = vert_label[0]
        sys.stdout.flush()
        glColor4f(1.0*vertex[0]/(self.columns-1), -1.0*vertex[1]/(self.rows-1)+1.0, -1.0*vertex[2]/(self.layers-1)+1.0, vert_label[1])
        glVertex3fv(vertex)
        
    def draw(self, pick=False):
        '''draws an image'''
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fov, self.aspect, 1, 10)
        glMatrixMode(GL_MODELVIEW)
        glDrawBuffer(GL_BACK)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluLookAt(self.center_x,self.center_y, 3, self.center_x, self.center_y, 2, 0,1,0)
        
        #Draw icon after rotating
        glColor3fv(self.icon_color)
        self.loading_icon()
        glColor3f(0.0, 0.0, 0.0)
        glMultMatrixd(self.arcball.matrix().T)
        glCallList(1)#draw the box and loading icon
        
        if not pick:
            #even numbers for display
            for idx in range(2, self.display_list_idx+1, 2):
                glCallList(idx)
            self.draw_marker()
            glutSwapBuffers()
        else:
            #odd numbers for picking
            for idx in range(3, self.display_list_idx+1, 2):
                glCallList(idx)
            glFlush()   
      
        return
        
    def draw_marker(self):
        '''Draws a sphere around the chosen point. Color is inverse of chosen pixel'''
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, (0.0, 0.0, 0.0))
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        location = self.pick_location
        glTranslatef(1.8*location[0]/self.columns-.9,
                    -(1.8*location[1]/self.rows-.9),
                    -(1.8*location[2]/self.layers-.9))
        glColor3fv(self.marker_color)
        glMaterial
        glutSolidSphere(.01, 50, 50)
        glTranslatef(-(1.8*location[0]/self.columns-.9), (1.8*location[1]/self.rows-.9),0)
        
        #draw a square parellel to z plane at z level of marker
        glBegin(GL_LINES)
        glColor3f(1.0, 1.0, 1.0)
        glVertex3f(-.9, -.9, 0)
        glVertex3f(.9, -.9, 0)
        glVertex3f(-.9, .9, 0)
        glVertex3f(.9, .9, 0)
        glVertex3f(-.9, -.9, 0)
        glVertex3f(-.9, .9, 0)
        glVertex3f(.9, -.9, 0)
        glVertex3f(.9, .9, 0)
        glVertex3f(-.9, .9, 0)
        glVertex3f(-.9, -.9, 0)
        glVertex3f(.9, -.9, 0)
        glVertex3f(.9, .9, 0)
        glEnd()
        
        glRasterPos(.9, .9, 0)
        glutBitmapString(GLUT_BITMAP_TIMES_ROMAN_24, "z= " + str(location[2]))
        
        glPopMatrix()
        
    def keyboard(self, key, x, y):
        if key == chr(27): #escape to quit
            sys.exit()
        if key == chr(8): #backspace to refresh/clear
            self.refresh()
        if key == chr(117): #u to undo
            self.undo()
        if key == chr(116): #t to translate to mouse location
            self.translate(x,y)
        if key == chr(99): #c to center the box
            self.reset_translation()
        if (key == chr(105) or key == chr(106) or key == chr(107) or key == chr(108)): #i, j, k, l to translate by increment
            self.shift(key)
        if (key == chr(114)): #r to reset the translation and zoom
            self.reset()
        if (key == chr(122)): #z to reset the zoom
            self.reset_zoom()
            
        return
        
    def on_scroll(self, wheel, direction, x, y):
        '''zooms in and out on mouse scroll wheel'''
        self.fov -= 1 if direction == 1 else -1
        glutPostRedisplay()

    def on_click(self, button, state, x, y):
        #Left click for arcball rotation
        if (button == GLUT_LEFT_BUTTON and state == GLUT_DOWN):
            self.left = True #turn on dragging rotation
            self.arcball.down((x,y))
        #right click to select a pixel location
        elif (button == GLUT_RIGHT_BUTTON and state == GLUT_DOWN):
            self.left = False #turn off dragging rotation
            self.draw(pick=True)
            self.pick_location, self.marker_color, label = self.pick(x,y)
            print self.pick_location#send the pick location to mojo
            print label #send the label location to mojo
            sys.stdout.flush()
            self.has_marker = True
    
    def pick(self, x,y):
        '''gets the (x,y,z) location in the full volume of a chosen pixel'''
        click_color = None
        glReadBuffer(GL_BACK)
        temp = glReadPixels(x,self.win_h-y, 1,1, GL_RGBA, GL_FLOAT)[0][0]
        click_color = temp[:3]
        label_idx = int(temp[3])
        sys.stdout.flush()
        label = self.label_dict[label_idx]
        if np.all(click_color!=0):
            location  = [int(click_color[0]*(self.columns-1)), 
                        int(-(click_color[1]-1)*(self.rows-1)), int(-(click_color[2]-1)*((self.layers-1)))]
            glReadBuffer(GL_FRONT)
            marker_color_neg = glReadPixels(x,self.win_h-y, 1,1, GL_RGB, GL_FLOAT)[0][0]
            marker_color = 1-marker_color_neg
            return location, marker_color, label
        return self.pick_location, self.marker_color, label
        
    def on_drag(self, x, y):
        '''rotates image on dragging with left mouse down'''
        if self.left:
            self.arcball.drag((x,y))
            glutPostRedisplay()
        
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
    viewer.extractor_dict[ids[0][0]] = extr
    handler = handler.Input_Handler(display_queue)
    
    viewer.set_dimensions(extr.rows, extr.columns, extr.layers, extr.w)
    
    viewer.label_dict[0/255.0] = ids[0][0]
    
    extracting_worker = threading.Thread(target = extr.run, name = "extr")
    input_worker = threading.Thread(target = handler.run, name = "input_worker")
    
    input_worker.daemon =True
    extracting_worker.daemon = True
    extracting_worker.start()
    input_worker.start()
    
    viewer.main()
