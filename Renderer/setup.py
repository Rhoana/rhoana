import sys
sys.path.append(r'c:\Python27\Lib\site-packages')
from distutils.core import setup
import numpy
import py2exe
import os
import glob
import zmq

os.environ["PATH"] =\
    os.environ["PATH"] + \
        os.path.pathsep+os.path.split(zmq.__file__)[0]

#folder = r'C:\Python27epd64\Scripts'
folder = r'C:\Python27\Scripts'
data_files = [(r'.',glob.glob(os.path.join(folder,'*mk2*.dll')))]

setup(console =['viewer_3d.py'], options = {"py2exe":{"excludes":["TkCommands", "Tkinter", "tcl", "OpenGL"], 
                                                    "includes": ["zmq.utils", "zmq.utils.jsonapi", "zmq.utils.strtypes", "h5py.*",
                                                                "numpy", "numpy.*", "numpy.linalg", "numpy.core", "scipy",
                                                                "ctypes", "logging"] }},
                                                                data_files = data_files)
"""
setup(console =['viewer_3d.py'], options = {"py2exe":{"excludes":["TkCommands", "Tkinter", "tcl", "OpenGL"], 
                                                    "includes": ["zmq.utils", "zmq.utils.jsonapi", "zmq.utils.strtypes", "h5py.*",
                                                                "numpy", "numpy.*", "numpy.linalg", "numpy.core", "scipy",
                                                                "win32", "ctypes", "logging"] }},
                                                                data_files = data_files)"""
