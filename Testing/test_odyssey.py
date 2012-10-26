import os
import os.path
import subprocess

class Runner(object):
    def __init__(self):
        self.jobs = []

    def start(self, jobargs):
        new_job = subprocess.Popen(jobargs)
        self.jobs += [new_job]

    def wait_all(self):
        while self.jobs:
            j = self.jobs.pop()
            j.wait()
            if j.returncode:
                raise RuntimeError("bad code: %d" % j.returncode)

if __name__ == '__main__':

    runner = Runner()

    # set up output directories
    for d in 'segmentations dicedblocks fusedblocks overlap_maps'.split(' '):
        if not os.path.exists(d):
            os.mkdir(d)

    input_volume = os.path.join(os.path.dirname(__file__),
                               '..', '0Control', 'Swift',
                               'connectome', 'Volume_odyssey.txt')

    # label the first 20 slices
    segmentation_files = [os.path.join('segmentations', 'segmentation%d.hdf5' % sliceidx) for sliceidx in range(20)]
    for sliceidx in range(20):
        output = segmentation_files[sliceidx]
        if not os.path.exists(output):
            runner.start([os.path.join('..', 'CubeDicing', 'segment_image_bsub.sh'),
                          input_volume, str(sliceidx), output])
    runner.wait_all()

    # dice out two cubes
    if not os.path.exists(os.path.join('dicedblocks', 'block1.hdf5')):
        runner.start([os.path.join('..', 'CubeDicing', 'dice_block_bsub.sh'),
                      '80', '80',
                      '220', '220'] + segmentation_files +
                     [os.path.join('dicedblocks', 'block1.hdf5')])
    if not os.path.exists(os.path.join('dicedblocks', 'block2.hdf5')):
        runner.start([os.path.join('..', 'CubeDicing', 'dice_block_bsub.sh'),
                      '180', '80',
                      '320', '220'] + segmentation_files +
                     [os.path.join('dicedblocks', 'block2.hdf5')])
    runner.wait_all()

    # fuse the cubes
    for idx in range(1, 3):
        runner.start([os.path.join('..', 'WindowFusion', 'window_fusion_bsub.sh'),
                      os.path.join('dicedblocks', 'block%d.hdf5' % idx),
                      str(idx),
                      os.path.join('fusedblocks', 'fused%d.hdf5' % idx)])
    runner.wait_all()
