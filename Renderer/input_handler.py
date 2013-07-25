#---------------------
#Input handler for 3D Viewer
#Daniel Miron
#719/13
#
#Version Date: 7/25 5:00
#---------------------


import sys
from Queue import Queue
import time
import re

class Input_Handler:
    def __init__(self, out_q):
        self.out_q = out_q #queue to viewer
        
    def run(self):
        while True:
            line = sys.stdin.readline()
            time.sleep(0.001)
            if line!="":
                args = line.split()
                if args[0] == "marker":
                    location = [int(args[1]), int(args[2]), int(args[3])]
                    self.out_q.put(["marker", location])
                elif args[0] == "ids":
                    ids = self.parse_ids(args[1:])
                    self.out_q.put(["ids" ,ids])
                elif args[0] == "limits":
                    limits = (int(args[1]), int(args[2]), int(args[3]))
                    self.out_q.put(["limits", limits])                    
                elif args[0] == "refresh":
                    self.out_q.put(["refresh"])
                elif args[0] == "undo":
                    self.out_q.put(["undo"])
                elif args[0] == "remove":
                    ids = self.parse_ids(args[1:])
                    self.out_q.put(["remove", ids]) 
                           
    def parse_ids(self, args):
        '''reformat ids into a list with the primary id the first element'''
        primary_id = []
        secondary_ids = []
        split_str = re.split(":", args[0])
        primary_id = [int(split_str[0])]
        if split_str[1] != "":
            secondary_ids = [int(label) for label in re.split(',', split_str[1])]
        ids = [primary_id + secondary_ids]
        return ids