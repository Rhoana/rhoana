import os
import os.path
import subprocess

class Runner(object):
    def __init__(self):
        self.jobs = []

    def start(self, jobargs):
        jobargs = [str(j) for j in jobargs]
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
    for d in 'segmentations dicedblocks fusedblocks matchedblocks overlap_maps'.split(' '):
        if not os.path.exists(d):
            os.mkdir(d)

    input_volume = os.path.join(os.path.dirname(__file__),
                               '..', '0Control', 'Swift',
                               'connectome', 'Volume_odyssey.txt')

    xy_size = 384
    xy_halo = 64
    z_size = 20
    z_halo = 6
    num_slices = z_size + 2 * z_halo
    # label the first num_slices slices
    segmentation_files = [os.path.join('segmentations', 'segmentation%d.hdf5' % sliceidx) for sliceidx in range(num_slices)]
    for sliceidx in range(num_slices):
        output = segmentation_files[sliceidx]
        if not os.path.exists(output):
            runner.start([os.path.join('..', 'CubeDicing', 'segment_image_bsub.sh'),
                          input_volume, str(sliceidx), output])
    runner.wait_all()

    # dice out two cubes
    if not os.path.exists(os.path.join('dicedblocks', 'block1.hdf5')):
        runner.start([os.path.join('..', 'CubeDicing', 'dice_block_bsub.sh'),
                      1, 1,
                      xy_size + 2 * xy_halo, xy_size + 2 * xy_halo,
                      ] + segmentation_files +
                     [os.path.join('dicedblocks', 'block1.hdf5')])
    if not os.path.exists(os.path.join('dicedblocks', 'block2.hdf5')):
        runner.start([os.path.join('..', 'CubeDicing', 'dice_block_bsub.sh'),
                      xy_size + 1, 1,
                      2 * xy_size + 2 * xy_halo, xy_size + 2 * xy_halo,
                      ] + segmentation_files +
                     [os.path.join('dicedblocks', 'block2.hdf5')])
    runner.wait_all()

    # fuse the cubes
    for idx in range(1, 3):
        runner.start([os.path.join('..', 'WindowFusion', 'window_fusion_bsub.sh'),
                      os.path.join('dicedblocks', 'block%d.hdf5' % idx),
                      str(idx),
                      os.path.join('fusedblocks', 'fused%d.hdf5' % idx)])
    runner.wait_all()

    # pairwise join the cubes
    runner.start([os.path.join('..', 'PairwiseMatching', 'pairwise_match_labels_bsub.sh'),
                  os.path.join('fusedblocks', 'fused1.hdf5'),
                  os.path.join('fusedblocks', 'fused2.hdf5'),
                  '1', xy_halo,
                  os.path.join('matchedblocks', 'matchedblock1.hdf5'),
                  os.path.join('matchedblocks', 'matchedblock2.hdf5')])
    runner.wait_all()
