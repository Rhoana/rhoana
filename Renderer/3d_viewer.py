#-------------------------
#3d Renderer
#Daniel Miron
#7/5/2013
#
#-------------------------

import h5py
import numpy as np
import sys
sys.path.append(r'c:\Python27\Lib\site-packages')
import pickle
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
import arcball as arc

import matplotlib.pyplot as plt

import cv2

class Viewer:
    def __init__(self, label_file, chunk_file):
        self.arcball = None
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
        
        self.chunk_map = self.read_chunk_map(chunk_file)
        
        self.keys = 0
        self.rotation_x = 0
        self.rotation_y = 0
        
        self.win_h = 0
        self.win_w = 0
        
        self.contours = []
        
        self.left = None
        self.slice = None
        self.pick_location = (0,0,0)
        
        self.picking_file = open(r"C:\Users\DanielMiron\Documents\3d_rendering\picking.txt", "w")
        
    def create_arcball(self):
        arcball = arc.Arcball()
        #locate the arcball center at center of window with radius half the width
        arcball.place([self.win_w/2, self.win_h/2], self.win_w/2)
        return arcball
        
    def make_display_list(self):
        '''Creates a display list to draw a box and the data scaled to .9*the size of the window'''
        
        self.display_list = glGenLists(1)
        glNewList(self.display_list, GL_COMPILE)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glTranslatef(-.9, .9, .9)
        glScalef(1.8/self.columns, -1.8/self.rows, -1.8/self.layers)
        
        #draw the layers
        for cnt in self.contours:
            gluTessBeginPolygon(self.tesselator, None)
            gluTessBeginContour(self.tesselator)
            for vtx in cnt:
                gluTessVertex(self.tesselator, vtx, vtx)
            gluTessEndContour(self.tesselator)
            gluTessEndPolygon(self.tesselator)
            
        #make a box around the image
        self.axes()
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
        
        #make a back panel for easy orientation
        glColor3f(.5, .5, .5)
        glBegin(GL_POLYGON)
        glVertex3f(*self.x_axis[2][0])
        glVertex3f(*self.x_axis[2][1])
        glVertex3f(*self.x_axis[3][1])
        glVertex3f(*self.x_axis[3][0])
        glEnd()
        
        glPopMatrix()
        
        glEndList()
        
    def vertex_callback(self, vertex):
        '''sets the color of a single vertex and draws it'''
        #scale by dim-1 to include black 
        #multiply by -1 and add 1 to invert color axis
        glColor3f(1.0*vertex[0]/(self.columns-1), -1.0*vertex[1]/(self.rows-1)+1.0, -1.0*vertex[2]/(self.layers-1)+1.0)
        glVertex3f(*vertex)
        
    def draw(self):
        '''draws an image'''
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluLookAt(0, 0, 3, 0, 0, 2, 0,1,0)
        glMultMatrixd(self.arcball.matrix().T)
        
        self.draw_marker()
    
        glCallList(self.display_list)
        glutSwapBuffers()
        return
        
    def draw_marker(self):
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        location = self.pick_location
        glTranslatef(float(1.8*location[0])/self.columns-.9,
                    -(float(1.8*location[1])/self.rows-.9),
                    -(float(1.8*location[2])/self.layers-.9))
        glScalef(1.8/self.layers, 1.8/self.layers, 1.8/self.layers)
        location = self.pick_location
        glColor3f(1-(1.0*location[0]/(self.columns-1)),
            1-(-1.0*location[1]/(self.rows-1)+1.0), 
            1-(-1.0*location[2]/(self.layers-1)+1.0))
        glutSolidSphere(10, 50, 50)
        
        glPopMatrix()
        
    def keyboard(self, key, x, y):
        return
        
    def on_click(self, button, state, x, y):
        #Left click for arcball rotation
        if (button == GLUT_LEFT_BUTTON and state == GLUT_DOWN):
            self.left = True #turn on dragging rotation
            self.arcball.down((x,y))
        #right click to select a pixel location
        elif (button == GLUT_RIGHT_BUTTON and state == GLUT_DOWN):
            self.left = False #turn off dragging rotation
            self.pick_location = self.pick(x,y)
            self.has_marker = True
            self.slice = self.show_slice(self.pick_location).astype(np.uint8)
        
        
    def show_slice(self, location):
        '''displays a single selected z slice in 2-d'''
        full_layer = self.ds[:, :, location[2]][...]
        layer =np.zeros(np.shape(full_layer))
        max_key = 0 #used to scale colors to 256 colors
        for key in self.keys:
            if key > max_key:
                max_key = key
            layer[full_layer == key] = key
        layer = 255*layer/key
        plt.imshow(layer)
        return layer
        
    def draw_slice(self):
        '''draws a single z slice'''
        if self.slice != None:
            width = float(np.shape(self.slice)[1])
            height = float(np.shape(self.slice)[0])
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            #scale window to size of slice
            glPixelZoom(self.win_w/width, self.win_h/height)
            glDrawPixels(width, height, GL_LUMINANCE, GL_UNSIGNED_BYTE, self.slice)
            glutSwapBuffers()
    
    def pick(self, x,y):
        '''gets the (x,y,z) location in the full volume of a chosen pixel'''
        click_color = None
        click_color = glReadPixels(x,self.win_h-y, 1,1, GL_RGB, GL_FLOAT)[0][0]
        location  = [int(click_color[0]*(self.columns-1)), 
                    int(-(click_color[1]-1)*(self.rows-1)), int(-(click_color[2]-1)*((self.layers-1)))]
        return location
        
    def on_drag(self, x, y):
        if self.left:
            self.arcball.drag((x,y))
            self.draw()
    
    def main(self, window_height, window_width, keys, contour_file):
        self.keys = keys
        self.contours = self.load_contours(contour_file)
        self.win_h = window_height
        self.win_w = window_width
        self.arcball = self.create_arcball()
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(self.win_w, self.win_h) #width, height
        glutCreateWindow("Nerve Cord")
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(65, 1, 1, 10)
        glMatrixMode(GL_MODELVIEW)
        
        self.tesselator = gluNewTess()
        gluTessCallback(self.tesselator, GLU_TESS_BEGIN, glBegin)
        gluTessCallback(self.tesselator, GLU_TESS_END, glEnd)
        gluTessCallback(self.tesselator, GLU_TESS_VERTEX, self.vertex_callback) 
        
        glEnable(GL_DEPTH_TEST)
        
        self.make_display_list()
        glutDisplayFunc(self.draw)
        glutKeyboardFunc(self.keyboard)
        glutMouseFunc(self.on_click)
        glutMotionFunc(self.on_drag)
        
        glutCreateWindow("single layer")
        glutDisplayFunc(self.draw_slice)
        
        
        glutMainLoop()
        return
        
        
    def get_contours(self, keys, contour_file):
        chunk_list = self.organize_chunks(keys)
        self.keys = keys
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
                                    cnt_3d += [[vtx[0][0]-1+chunk[1],vtx[0][1]-1+chunk[0], layer+chunk[2]]] #subtract 1 to adjust back after buffer
                                contours_3d += [cnt_3d]
                            self.contours +=contours_3d
        self.save_contours(contour_file)
        
        
    def organize_chunks(self, keys):
        chunk_list = []
        for key in keys:
            chunk_list += self.chunk_map[key]
        chunk_list.sort(key=lambda x: x[2]) #sort w/respect to z
        chunk_list.reverse() #make back to front
        return chunk_list
        
    def read_chunk_map(self, chunk_file):
        return pickle.load(open(chunk_file, "rb"))
        
    def save_contours(self, contour_file):
        pickle.dump(self.contours, open(contour_file, "wb"))
        return
        
    def load_contours(self, contour_file):
        return pickle.load(open(contour_file, "rb"))
        
    def axes(self):
        self.x_axis = [[[0,0,0], [self.columns, 0,0]], [[0,self.rows,0], [self.columns, self.rows, 0]],
                        [[0,0,self.layers], [self.columns,0,self.layers]], [[0, self.rows, self.layers], [self.columns, self.rows, self.layers]]]
        self.y_axis = [[[0,0,0], [0, self.rows,0]], [[self.columns,0,0],[self.columns, self.rows, 0]], [[0,0,self.layers], [0,self.rows, self.layers]],
                        [[self.columns, 0, self.layers],[self.columns, self.rows, self.layers]]]
        self.z_axis = [[[0,0,0], [0,0,self.layers]], [[self.columns,0,0],[self.columns, 0, self.layers]],
                        [[0, self.rows,0], [0, self.rows, self.layers]],[[self.columns, self.rows, 0],[self.columns, self.rows, self.layers]]]
        
        '''for lines in [self.x_axis, self.y_axis, self.z_axis]:
            for line in lines:
                for vtx in line:
                    vtx[0] = 1.8*(float(vtx[0])/float(self.columns)-0.5)
                    vtx[1] = -1.8*(float(vtx[1])/float(self.rows)-0.5)
                    vtx[2] = -1.8*(float(vtx[2])/float(self.layers)-0.5)'''
    
        
        

viewer = Viewer(r'C:\Users\DanielMiron\Documents\3d_rendering\labels_full.h5',
                r'C:\Users\DanielMiron\Documents\3d_rendering\label_full_chunk_map.p')
viewer.main(1000,1000, [6642,4627], r'C:\Users\DanielMiron\Documents\3d_rendering\contours_full.p')
