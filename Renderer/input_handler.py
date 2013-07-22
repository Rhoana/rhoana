import sys
from Queue import Queue
import time

class Input_Handler:
    def __init__(self, out_q):
        self.out_q = out_q
        
    def run(self):
        while True:
            line = sys.stdin.readline()
            if line!="":
                self.out_q.put(line)