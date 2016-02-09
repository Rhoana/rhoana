import multiprocessing

class LocalRunner(object):
    def __init__(self, storage, num_cpus=None):
        self.storage = storage
        self.num_cpus = num_cpus or multiprocessing.cpu_count()
