 # -------------------------
 # 3d Renderer
 # Daniel Miron
 # 7/5/2013
 # 
 # Allows 3d viewing of nerve cord or neuron stacks.
 # Includes ability to fully rotate image in 3 dimensions and to mark locations in 3-space
 # 
 # Version Date: 7/26 10:30
 # -------------------------

import sys
sys.path.append('.')
import numpy as np

from optparse import OptionParser

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

import extractor
import input_handler as handler

try:
    from OpenGL.platform import win32
except AttributeError:
    pass

class Viewer:
    def __init__(self, location, start_label, in_q, directory, max_x, max_y, z_spacing):
        self.st = time.time()
        self.win_h = 1000
        self.win_w = 1000
        self.arcball = self.create_arcball()

        self.directory = directory
        self.in_q = in_q
        self.max_x = max_x  # highest resolution
        self.max_y = max_y  # highest resolution
        self.z_spacing = z_spacing

        self.rows = 0
        self.columns = 0
        self.layers = 0

        self.fov = 60
        self.aspect = float(self.win_w) / self.win_h

        self.left = None  # keep track of left button status
        self.pick_location = location
        self.label = start_label

        self.display_list_idx = 2  # count from 1 and use first index for box
        self.display_list_dict = dict()  # COLOR as key, display_list indices as value
        self.marker_color = [1., 1., 1.]  # initial marker is white

        self.first = True  # used to control display list flow
        self.icon_color = np.array((0.0, 1.0, 0.0))
        self.st = time.time()

        self.center_x = 0
        self.center_y = 0

        self.extractor_dict = dict()  # keys are indices, values are extractor threads
        self.make_lists = True

        self.num_labels = 0
        self.label_dict = dict()  # keys are float indices, values are labels

    def set_dimensions(self, rows, columns, layers, w):
        '''sets the dimensions of the viewing box'''
        self.rows = rows
        self.columns = columns
        self.layers = layers
        self.xyscale = 1.0 / max(rows, columns)
        self.zscale = self.xyscale

        self.pick_location = (self.pick_location[0]/pow(2, w) - 1, self.pick_location[1]/pow(2, w) - 1, self.pick_location[2])

    def main(self):
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_ALPHA | GLUT_MULTISAMPLE)
        glutInitWindowSize(self.win_w, self.win_h)  # width, height
        glutCreateWindow("3D View")

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fov, self.aspect, 1, 10)
        glMatrixMode(GL_MODELVIEW)

        glShadeModel(GL_SMOOTH)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_NORMALIZE)
        glColorMaterial(GL_FRONT_AND_BACK, GL_EMISSION)
        glEnable(GL_COLOR_MATERIAL)

        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
        glEnable(GL_LINE_SMOOTH);
        try:
            glEnable(GL_MULTISAMPLE)
        except:
            pass

        glLightfv(GL_LIGHT0, GL_SPECULAR, (0,0,0))

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
        glViewport(0, 0, w, h)
        self.arcball.place([self.win_w/2, self.win_h/2], self.win_w/2)

    def translate(self, x, y):
        '''translate the viewing box based on mouse location'''
        self.center_x = self.center_x+((float(x)/self.win_w)-.5)*2
        self.center_y = self.center_y-((float(y)/self.win_h)-.5)*2

    def shift(self, key):
        '''translate the viewing box based on keyboard input'''
        # may want to tune the translation levels better
        if key == chr(105):
            self.center_y -=float(self.fov)**2/10000
        elif key == chr(106):
            self.center_x +=float(self.fov)**2/10000
        elif key == chr(107):
            self.center_y += float(self.fov)**2/10000
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
        timer = time.time()
        while(not self.in_q.empty() and time.time()-timer<.1):
            self.icon_color = (self.icon_color + .01)%1  # resets to black when icon is green since 1.0 and 0.0 %1 are equal
            temp = self.in_q.get()
            if temp[0] == "marker":
                self.pick_location = temp[1:][0]
                self.pick_location[0] = int(float(self.pick_location[0]*self.columns)/self.max_x)
                self.pick_location[1] = int(float(self.pick_location[1]*self.rows)/self.max_y)
            elif temp[0] == "ids":
                self.num_labels += 1
                label_idx = self.num_labels
                self.label_dict[label_idx] = temp[1:][0][0][0]
                extr = extractor.Extractor(self.in_q, self.directory, temp[1:][0], self.pick_location, self.max_x, self.max_y, label_idx)
                self.extractor_dict[temp[1][0][0]] = extr
                extracting_worker = threading.Thread(target = extr.run, name = "extr")
                extracting_worker.daemon = True
                extracting_worker.start()
            elif temp[0] == "contours":
                _, contours, color, primary_label, label_idx, z = temp
                if self.make_lists:
                    self.make_display_lists(contours, color/255.0, primary_label, label_idx, z)
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
        # set icon to green if processes are done
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
        # first display list is for the box
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
            glDeleteLists(display_list, 1)  # delete back and front lists
            glutPostRedisplay()

    def remove_label(self, ids):
        '''remove a single contour'''
        for display_list in self.display_list_dict[ids[0]]:
            glDeleteLists(display_list, 1)
            self.extractor_dict[ids[0]].stop()
            glutPostRedisplay()

    def create_arcball(self):
        arcball = arc.Arcball()
        # locate the arcball center at center of window with radius half the width
        arcball.place([self.win_w/2, self.win_h/2], self.win_w/2)
        return arcball

    def make_display_lists(self, contours, color, label, label_idx, z):
        '''Generates display lists to draw both the front and back buffered images'''
        if self.first:  # make the box
            self.axes()
            self.make_box_list()
            self.first = False
        self.display_lists = glGenLists(2)  # first list for front, second for back
        if label in self.display_list_dict:
            self.display_list_dict[label] = self.display_list_dict[label] + [self.display_list_idx, self.display_list_idx+1]
        else:
            self.display_list_dict[label] = [self.display_list_idx, self.display_list_idx+1]
        self.make_front_list(contours, color, z)
        self.make_back_list(contours, label_idx, z)
        self.display_list_idx +=2

    def make_back_list(self, contours, label_idx, z):
        '''Creates a display list to encode color for image. Not seen by user'''
        glNewList(self.display_list_idx+1, GL_COMPILE)
        glDisable(GL_LIGHTING)  # don't use lighting for color encoding

        glMatrixMode(GL_MODELVIEW)

        colors = [[]] * len(contours)
        for idx, cnt in enumerate(contours):
            clr = np.zeros((cnt.shape[0], 4), np.float)
            clr[:, :2] = cnt
            clr[:, 2] = z
            clr[:, 0] /= self.rows - 1
            clr[:, 1] /= self.columns - 1
            clr[:, 2] /= self.layers - 1
            clr[:, 3] = label_idx/255.0
            colors[idx] = clr

        for offset in range(max(1, int(self.z_spacing))):
            glPushMatrix()
            glTranslatef(-.9, .9, .9)
            glScalef(1.8 * self.xyscale, -1.8 * self.xyscale, -1.8 * self.zscale)
            # draw the layers

            glTranslatef(0, 0, z * self.z_spacing + offset)  # shift by Z offset

            glLineWidth(1.5)
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_COLOR_ARRAY)
            for cnt, clrs in zip(contours, colors):
                glVertexPointer(2, GL_INT, 0, cnt)
                glColorPointer(4, GL_FLOAT, 0, clrs)
                glDrawArrays(GL_LINE_LOOP, 0, cnt.shape[0])

            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_COLOR_ARRAY)

            glPopMatrix()

        glEnable(GL_LIGHTING)

        glEndList()

    def make_front_list(self, contours, color, z):
        '''Creates a display list to draw a box and the data scaled to .9*the size of the window.
        This list deals with the display seen by the user'''

        glNewList(self.display_list_idx, GL_COMPILE)

        glMatrixMode(GL_MODELVIEW)

        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color)

        normals = [[]] * len(contours)
        for idx, cnt in enumerate(contours):
            norms = np.zeros((cnt.shape[0], 3), dtype=np.float)
            norms[:, 0] = - (np.roll(cnt[:, 1], -1) - cnt[:, 1])
            norms[:, 1] = (np.roll(cnt[:, 0], -1) - cnt[:, 0])
            norms = norms + np.roll(norms, 1, axis=0)
            norms = 0.5 * norms + 0.25 * (np.roll(norms, 1, axis=0) + np.roll(norms, -1, axis=0))
            norms = 0.5 * norms + 0.25 * (np.roll(norms, 1, axis=0) + np.roll(norms, -1, axis=0))
            normals[idx] = norms

        for offset in range(max(1, int(self.z_spacing))):
            glPushMatrix()
            glTranslatef(-.9, .9, .9)
            glScalef(1.8 * self.xyscale, -1.8 * self.xyscale, -1.8 * self.zscale)

            # draw the layers
            glTranslatef(0, 0, z * self.z_spacing + offset)  # shift by Z offset
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_NORMAL_ARRAY)
            glLineWidth(1.5)

            for cnt, norms in zip(contours, normals):
                glVertexPointer(2, GL_INT, 0, cnt)
                glNormalPointer(GL_FLOAT, 0, norms)
                glDrawArrays(GL_LINE_LOOP, 0, cnt.shape[0])

            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_NORMAL_ARRAY)
            glPopMatrix()
        glEndList()

    def make_box_list(self):
        '''makes a display list to draw the box'''
        x = self.rows
        y = self.columns
        z = self.layers * self.z_spacing

        glNewList(1, GL_COMPILE)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()

        glTranslatef(-.9, .9, .9)
        glScalef(1.8 * self.xyscale, -1.8 * self.xyscale, -1.8 * self.zscale)

        glDisable(GL_LIGHTING)


        glBegin(GL_LINES)
        # make a box around the image
        glColor3f(1.0, 0, 0)  # x in red
        for yoff in [0, y]:
            for zoff in [0, z]:
                glVertex3f(0, yoff, zoff)
                glVertex3f(x, yoff, zoff)

        glColor3f(0,1.0, 0)  # y in green
        for xoff in [0, x]:
            for zoff in [0, z]:
                glVertex3f(xoff, 0, zoff)
                glVertex3f(xoff, y, zoff)

        glColor3f(0,0,1.0)  # z in blue
        for xoff in [0, x]:
            for yoff in [0, y]:
                glVertex3f(xoff, yoff, 0)
                glVertex3f(xoff, yoff, z)
        glEnd()

        glColor3f(0.5, 0.5, 0.5)


        glRasterPos3f(0, 0, 0)
        glutBitmapString(GLUT_BITMAP_TIMES_ROMAN_24, "(0,0,0)")
        glRasterPos3f(x, 0, 0)
        glutBitmapString(GLUT_BITMAP_TIMES_ROMAN_24, "x=" + str(self.rows-1))
        glRasterPos3f(0, y, 0)
        glutBitmapString(GLUT_BITMAP_TIMES_ROMAN_24, "y= " + str(self.columns-1))
        glRasterPos3f(0, 0, z)
        glutBitmapString(GLUT_BITMAP_TIMES_ROMAN_24, "z= " + str(self.layers-1))
        glEnable(GL_LIGHTING)
        glPopMatrix()
        glEndList()

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

         # Draw icon after rotating
        glColor3fv(self.icon_color)
        self.loading_icon()
        glColor3f(0.0, 0.0, 0.0)
        glMultMatrixd(self.arcball.matrix().T)
        glCallList(1) # draw the box and loading icon

        if not pick:
             # even numbers for display
            for idx in range(2, self.display_list_idx+1, 2):
                glCallList(idx)
            self.draw_marker()
            glutSwapBuffers()
        else:
             # odd numbers for picking
            for idx in range(3, self.display_list_idx+1, 2):
                glCallList(idx)
            glFlush()

        return

    def draw_marker(self):
        '''Draws a sphere around the chosen point. Color is inverse of chosen pixel'''
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glTranslatef(-.9, .9, .9)
        glScalef(1.8 * self.xyscale, -1.8 * self.xyscale, -1.8 * self.zscale)

        location = self.pick_location
        glPushMatrix()
        glTranslatef(location[0],
                     location[1],
                     location[2] * self.z_spacing)
        glColor3fv(self.marker_color)

        glEnable(GL_LIGHTING)
        glutSolidSphere(3, 8, 8)
        glPopMatrix()

         # draw a square parellel to z plane at z level of marker
        glBegin(GL_LINE_LOOP)
        glColor3f(1.0, 1.0, 1.0)
        glVertex3f(0, 0, location[2] * self.z_spacing)
        glVertex3f(self.rows, 0, location[2] * self.z_spacing)
        glVertex3f(self.rows, self.columns, location[2] * self.z_spacing)
        glVertex3f(0, self.columns, location[2] * self.z_spacing)
        glEnd()


        glColor3f(1.0, 1.0, 0)
        glRasterPos(0, 0, location[2])
        glDisable(GL_LIGHTING)
        glutBitmapString(GLUT_BITMAP_TIMES_ROMAN_24, "z= " + str(location[2]))
        glEnable(GL_LIGHTING)

        glPopMatrix()

    def keyboard(self, key, x, y):
        key = key.lower()
        if key == chr(27):  # escape to quit
            sys.exit()
        if key == chr(8):  # backspace to refresh/clear
            self.refresh()
        if key == 'u':  # u to undo
            self.undo()
        if key == 't':  # t to translate to mouse location
            self.translate(x,y)
        if key == 'c':  # c to center the box
            self.reset_translation()
        if key in 'ijkl':
            self.shift(key)
        if (key == 'r'):  # r to reset the translation and zoom
            self.reset()
        if (key == 'z'):  # z to reset the zoom
            self.reset_zoom()
        if (key == '+'):
            self.fov -= 1
            glutPostRedisplay()
        if (key == '-'):
            self.fov += 1
            glutPostRedisplay()
        return

    def on_scroll(self, wheel, direction, x, y):
        '''zooms in and out on mouse scroll wheel'''
        self.fov -= 1 if direction == 1 else -1
        glutPostRedisplay()

    def on_click(self, button, state, x, y):
         # Left click for arcball rotation
        if (button == GLUT_LEFT_BUTTON and state == GLUT_DOWN):
            self.left = True  # turn on dragging rotation
            self.arcball.down((x,y))
         # right click to select a pixel location
        elif (button == GLUT_RIGHT_BUTTON and state == GLUT_DOWN):
            self.left = False  # turn off dragging rotation
            self.draw(pick=True)
            self.pick_location, self.marker_color, self.label = self.pick(x,y)
            print "location", self.pick_location[0], self.pick_location[1], self.pick_location[2], self.label  # send the label location to mojo
            sys.stdout.flush()
            self.has_marker = True

    def pick(self, x,y):
        '''gets the (x,y,z) location in the full volume of a chosen pixel'''
        click_color = None
        glReadBuffer(GL_BACK)
        temp = glReadPixels(x,self.win_h-y, 1,1, GL_RGBA, GL_FLOAT)[0][0]
        click_color = temp[:3]
        label_idx = int(temp[3]*255.0 + .5)
        label = self.label_dict[label_idx]
        if not np.all(click_color==0):

            location = [int(click_color[0]*(self.rows-1)),
                        int(click_color[1]*(self.columns-1)),
                        int(click_color[2]*(self.layers-1))]
            glReadBuffer(GL_FRONT)
            marker_color_neg = glReadPixels(x,self.win_h-y, 1,1, GL_RGB, GL_FLOAT)[0][0]
            marker_color = 1-marker_color_neg
            return location, marker_color, label
        return self.pick_location, self.marker_color, self.label

    def on_drag(self, x, y):
        '''rotates image on dragging with left mouse down'''
        if self.left:
            self.arcball.drag((x,y))
            glutPostRedisplay()

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

    progname = sys.argv[0]

    # Option parsing
    parser = OptionParser()
    parser.add_option("-d", "--directory", dest="directory",
                      help="Mojo directory")
    parser.add_option("--xyz", dest="location",
                      help="Minimum X,Y,Z (comma separated, no spaces)")
    parser.add_option("--max_x", dest="max_x",
                      help="Maximum X", type="int")
    parser.add_option("--max_y", dest="max_y",
                      help="Maximum Y", type="int")
    parser.add_option("--z_spacing", dest="z_spacing", default=1.0,
                      help="Z spacing, in same units as XY pixels", type="float")
    (options, args) = parser.parse_args()

    directory = options.directory
    location = [int(v) for v in options.location.split(',')]
    max_x = options.max_x
    max_y = options.max_y
    z_spacing = options.z_spacing

    ids = []
    for label_set in args:
        primary_id = []
        secondary_ids = []
        split_str = re.split(":", label_set)
        primary_id = [int(split_str[0])]
        if split_str[1] != "":
            secondary_ids = [int(label) for label in re.split(',', split_str[1])]
        ids += [primary_id + secondary_ids]

    extr = extractor.Extractor(display_queue, directory, ids, location, max_x, max_y, 0)
    viewer  = Viewer(location, ids[0][0], display_queue, directory, max_x, max_y, z_spacing)

    viewer.extractor_dict[ids[0][0]] = extr
    handler = handler.Input_Handler(display_queue)

    viewer.set_dimensions(extr.rows, extr.columns, extr.layers, extr.w)

    viewer.label_dict[0] = ids[0][0]

    extracting_worker = threading.Thread(target = extr.run, name = "extr")
    input_worker = threading.Thread(target = handler.run, name = "input_worker")

    input_worker.daemon =True
    extracting_worker.daemon = True
    extracting_worker.start()
    input_worker.start()

    viewer.main()
