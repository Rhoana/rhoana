import sys
import os.path
import os
import subprocess
import datetime

class Job(object):
    all_jobs = []

    def __init__(self):
        self.name = self.__class__.__name__ + str(len(Job.all_jobs)) + '_' + datetime.datetime.now().isoformat()
        Job.all_jobs.append(self)

    def run(self):
        # Make sure output directories exist
        out = self.output
        if isinstance(out, basestring):
            out = [out]
        for f in out:
            if not os.path.isdir(os.path.dirname(f)):
                os.mkdir(os.path.dirname(f))
        subprocess.check_call(["bsub",
                               "-q", "short_serial",
                               "-J", self.name,
                               "-w", self.dependency_string()] +
                              self.command())

    def dependency_string(self):
        return " && ".join("done(%s)" % d.name for d in self.dependencies)

    @classmethod
    def run_all(cls):
        for j in cls.all_jobs:
            j.run()

class JobSplit(object):
    '''make a multi-output job object look like a single output job'''
    def __init__(self, job, idx):
        self.job = job
        self.idx = idx
        self.name = job.name

    @property
    def output(self):
        return self.job.output[self.idx]

class SegmentedSlice(Job):
    def __init__(self, raw_image, index):
        Job.__init__(self)
        self.raw_image = raw_image
        self.dependencies = []
        self.output = os.path.join('segmentations', 'slice_%d.hdf5' % index)

    def command(self):
        return ['./segment_image.sh', self.raw_image, self.output]

class Block(Job):
    def __init__(self, segmented_slices, indices, *args):
        Job.__init__(self)
        self.segmented_slices = segmented_slices
        self.dependencies = segmented_slices
        self.args = [str(a) for a in args]
        self.output = os.path.join('dicedblocks', 'block_%d_%d_%d.hdf5' % indices)

    def command(self):
        return ['./dice_block.sh'] + self.args + [s.output for s in self.dependencies] + [self.output]

class FusedBlock(Job):
    def __init__(self, block, indices, global_block_number):
        Job.__init__(self)
        self.block = block
        self.global_block_number = global_block_number
        self.dependencies = [block]
        self.output = os.path.join('fusedblocks', 'fusedblock_%d_%d_%d.hdf5' % indices)

    def command(self):
        return ['./window_fusion.sh',
                self.block.output,
                str(self.global_block_number),
                self.output]

class PairwiseMatching(Job):
    def __init__(self, fusedblock1, fusedblock2, direction, even_or_odd, halo_width):
        Job.__init__(self)
        self.direction = direction
        self.even_or_odd = even_or_odd
        self.halo_width = halo_width
        self.dependencies = [fusedblock1, fusedblock2]
        outdir = 'pairwise_matches_%s_%s' % (['X', 'Y', 'Z',][direction], even_or_odd)
        self.output = (os.path.join(outdir, os.path.basename(fusedblock1.output)),
                       os.path.join(outdir, os.path.basename(fusedblock2.output)))

    def command(self):
        return ['./pairwise_match_labels.sh'] + [d.output for d in self.dependencies] + \
            [str(self.direction + 1), # matlab
             str(self.halo_width)] + list(self.output)

if __name__ == '__main__':
    image_size = 1024
    xy_size = 384
    xy_halo = 64
    z_size = 20
    z_halo = 6

    assert 'CONNECTOME' in os.environ

    # Label all slices
    segmentations = [SegmentedSlice(f, idx) for idx, f in
                     enumerate(f.rstrip() for f in open(sys.argv[1]))]

    # Dice full volume
    blocks = {}
    for block_idx_z in range((len(segmentations) - 2 * z_halo) / z_size):
        lo_slice = block_idx_z * z_size
        hi_slice = lo_slice + z_size + 2 * z_halo
        for block_idx_x in range((image_size - 2 * xy_halo) / xy_size):
            xlo = block_idx_x * xy_size
            xhi = xlo + xy_size + 2 * xy_halo
            for block_idx_y in range((image_size - 2 * xy_halo) / xy_size):
                ylo = block_idx_y * xy_size
                yhi = ylo + xy_size + 2 * xy_halo
                blocks[block_idx_x, block_idx_y, block_idx_z] = \
                    Block(segmentations[lo_slice:hi_slice],
                          (block_idx_x, block_idx_y, block_idx_z),
                          xlo + 1, ylo + 1, xhi + 1, yhi + 1)  # matlab indexing

    # Window fuse all blocks
    fused_blocks = dict((idxs, FusedBlock(block, idxs, num)) for num, (idxs, block) in enumerate(blocks.iteritems()))

    # Pairwise match all blocks.
    #
    # We overwrite each block in fused_blocks with the output of the pairwise
    # matching, and work in non-overlapping sets (even-to-odd, then odd-to-even)
    for direction in range(3):  # X, Y, Z
        for wpidx, which_pairs in enumerate(['even', 'odd']):
            for idx in fused_blocks:
                if (idx[direction] % 2) == wpidx:  # merge even-to-odd, then odd-to-even
                    neighbor_idx = list(idx)
                    neighbor_idx[direction] += 1  # check neighbor exists
                    neighbor_idx = tuple(neighbor_idx)
                    if neighbor_idx in fused_blocks:
                        pw = PairwiseMatching(fused_blocks[idx], fused_blocks[neighbor_idx],
                                              direction,  # matlab
                                              which_pairs,
                                              xy_halo if direction < 2 else z_halo)
                        # we can safely overwrite because of nonoverlapping even/odd sets
                        fused_blocks[idx] = JobSplit(pw, 0)
                        fused_blocks[idx] = JobSplit(pw, 1)

    Job.run_all()
