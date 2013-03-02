import time

class Timer(object):
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        print "{0} took {1} seconds".format(self.name, int(time.time() - self.start))

        
