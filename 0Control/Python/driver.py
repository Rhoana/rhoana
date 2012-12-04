import sys
import os.path
import os
import subprocess
import datetime
from itertools import product

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
        print "RUN", self.command()
        subprocess.check_call(["bsub",
                               "-g", "/diced_connectome",
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

class Reassemble(Job):
    '''reassemble a diced job'''
    def __init__(self, dataset, output_sizes, joblist, output):
        Job.__init__(self)
        self.output_sizes = output_sizes
        self.dataset = dataset
        self.dependencies = joblist
        self.output = output

    def command(self):
        return ['./reassemble.sh', self.dataset,
                str(len(self.output_sizes))] + \
            [str(s) for s in self.output_sizes] + \
            [j.output for j in self.dependencies] + \
            [self.output]

class Subimage_ProbabilityMap(Job):
    def __init__(self, raw_image, idx, xlo, ylo, xhi, yhi, xlo_core, ylo_core, xhi_core, yhi_core):
        Job.__init__(self)
        self.raw_image = raw_image
        self.dependencies = []
        self.coords = [str(c) for c in (xlo, ylo, xhi, yhi)]
        self.core_coords = [str(c) for c in (xlo_core, ylo_core, xhi_core, yhi_core)]
        self.output = os.path.join('subimage_probabilities',
                                   'probs_%d_%s.hdf5' % (idx, '_'.join(self.coords)))

    def command(self):
        return ['./compute_probabilities.sh', self.raw_image, self.output] + \
            self.coords + self.core_coords

class Subimage_SegmentedSlice(Job):
    def __init__(self, idx, probability_map, raw_image, xlo, ylo, xhi, yhi, xlo_core, ylo_core, xhi_core, yhi_core):
        Job.__init__(self)
        self.probability_map = probability_map
        self.raw_image = raw_image
        self.dependencies = [self.probability_map]
        self.coords = [str(c) for c in (xlo, ylo, xhi, yhi)]
        self.core_coords = [str(c) for c in (xlo_core, ylo_core, xhi_core, yhi_core)]
        self.output = os.path.join('subimage_segmentations',
                                   'segs_%d_%s.hdf5' % (idx, '_'.join(self.coords)))

    def command(self):
        return ['./segment_image.sh', self.raw_image, self.probability_map.output, self.output] + \
            self.coords + self.core_coords

class ProbabilityMap(Job):
    def __init__(self, raw_image, index):
        Job.__init__(self)
        self.raw_image = raw_image
        self.dependencies = []
        self.output = os.path.join('probabilities', 'probs_%d.hdf5' % index)

    def command(self):
        return ['./compute_probabilities.sh', self.raw_image, self.output]

class SegmentedSlice(Job):
    def __init__(self, probability_image, index):
        Job.__init__(self)
        self.probability_image = probability_image.output
        self.raw_image = probability_image.raw_image
        self.dependencies = [probability_image]
        self.output = os.path.join('segmentations', 'slice_%d.hdf5' % index)

    def command(self):
        return ['./segment_image.sh', self.raw_image, self.probability_image, self.output]

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

class JoinConcatenation(Job):
    def __init__(self, outfilename, inputs):
        Job.__init__(self)
        self.dependencies = inputs
        self.output = os.path.join('joins', outfilename)

    def command(self):
        return ['./concatenate_joins.sh'] + \
            [s.output for s in self.dependencies] + \
            [self.output]

class GlobalRemap(Job):
    def __init__(self, outfilename, joinjob):
        Job.__init__(self)
        self.dependencies = [joinjob]
        self.joinfile = joinjob.output
        self.output = os.path.join('joins', outfilename)

    def command(self):
        return ['./create_global_map.sh', self.joinfile, self.output]

class RemapBlock(Job):
    def __init__(self, blockjob, build_remap_job, indices):
        Job.__init__(self)
        self.dependencies = [blockjob, build_remap_job]
        self.inputfile = blockjob.output
        self.mapfile = build_remap_job.output
        self.output = os.path.join('relabeledblocks', 'block_%d_%d_%d.hdf5' % indices)

    def command(self):
        return ['./remap_block.sh', self.inputfile, self.mapfile, self.output]

###############################
# Helper functions
###############################
def dice_iter(full_size, core_size, halo_size):
    # we produce two sets of bounds: halo+core+halo and core alone
    for lo in range(0, full_size, core_size):
        yield (max(0, lo - halo_size),
               min(full_size - 1, lo + core_size + 2 * halo_size),
               lo,
               min(full_size - 1, lo + core_size))

def dice(job_builder, args, full_sizes, core_sizes, halo_sizes):
    iters = [dice_iter(*sizes) for sizes in zip(full_sizes, core_sizes, halo_sizes)]
    jobs = []
    for coords in product(*iters):
        # coords is a tuples of (lo, hi)
        lovals, hivals, locore, hicore = zip(*coords)
        jobs.append(job_builder(*(args + lovals + hivals + locore + hicore)))
    return jobs


###############################
# Driver
###############################
if __name__ == '__main__':
    image_size = 5120

    probability_subimage_size = 1024
    probability_subimage_halo = 32

    segmentation_subimage_size = 1024
    segmentation_subimage_halo = 128

    block_xy_size = 384
    block_xy_halo = 64
    block_z_size = 20
    block_z_halo = 6

    assert 'CONNECTOME' in os.environ
    assert 'VIRTUAL_ENV' in os.environ

    images = [f.rstrip() for f in open(sys.argv[1])]

    # Dice raw images and cmopute probability maps
    subimage_probability_maps = [dice(Subimage_ProbabilityMap,
                                      (f, idx),
                                      (image_size, image_size),
                                      (probability_subimage_size, probability_subimage_size),
                                      (probability_subimage_halo, probability_subimage_halo))
                                 for idx, f in enumerate(images)]

    print subimage_probability_maps[0][0].command()

    # Reassemble full probability maps
    probability_maps = [Reassemble('improb', (image_size, image_size),
                                   subs, os.path.join('probabilities', 'prob_%d.hdf5' % idx))
                        for idx, subs in enumerate(subimage_probability_maps)]

    Job.run_all()
    sys.exit(0)

    # Dice probability maps and compute segmentations
    subimage_segmentations = [dice(Subimage_SegmentedSlice,
                                   (idx, pm, raw_image),
                                   (image_size, image_size),
                                   (segmentation_subimage_size, segmentation_subimage_size),
                                   (segmentation_subimage_halo, segmentation_subimage_halo))
                              for idx, (pm, raw_image) in enumerate(zip(probability_maps, images))]

    # Reassemble full segmentation maps
    segmentations = [Reassemble('segs', (image_size, image_size, -1),
                                subs, os.path.join('segmentations', 'segs_%d.hdf5' % idx))
                     for idx, subs in enumerate(subimage_segmentations)]

    Job.run_all()
    sys.exit(0)

    # Dice full volume
    blocks = {}
    for block_idx_z in range((len(segmentations) - 2 * block_z_halo) / block_z_size):
        lo_slice = block_idx_z * block_z_size
        hi_slice = lo_slice + block_z_size + 2 * block_z_halo
        for block_idx_x in range((image_size - 2 * block_xy_halo) / block_xy_size):
            xlo = block_idx_x * block_xy_size
            xhi = xlo + block_xy_size + 2 * block_xy_halo
            for block_idx_y in range((image_size - 2 * block_xy_halo) / block_xy_size):
                ylo = block_idx_y * block_xy_size
                yhi = ylo + block_xy_size + 2 * block_xy_halo
                blocks[block_idx_x, block_idx_y, block_idx_z] = \
                    Block(segmentations[lo_slice:hi_slice],
                          (block_idx_x, block_idx_y, block_idx_z),
                          xlo + 1, ylo + 1, xhi, yhi)  # matlab indexing

    # Window fuse all blocks
    fused_blocks = dict((idxs, FusedBlock(block, idxs, num)) for num, (idxs, block) in enumerate(blocks.iteritems()))

    # Pairwise match all blocks.
    #
    # We overwrite each block in fused_blocks (the python dict, not the file)
    # with the output of the pairwise matching, and work in non-overlapping
    # sets (even-to-odd, then odd-to-even)
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
                                              block_xy_halo if direction < 2 else block_z_halo)
                        # we can safely overwrite because of nonoverlapping even/odd sets
                        fused_blocks[idx] = JobSplit(pw, 0)
                        fused_blocks[neighbor_idx] = JobSplit(pw, 1)

    # Contatenate the joins from all the blocks to a single file, for building
    # the global remap.  Work first on XY planes, to add some parallelism and
    # limit number of command arguments.
    plane_joins_lists = {}
    for idxs, block in fused_blocks.iteritems():
        plane_joins_lists[idxs[2]] = plane_joins_lists.get(idxs[2], []) + [block]
    plane_join_jobs = [JoinConcatenation('concatenate_Z_%d' % idx, plane_joins_lists[idx])
                       for idx in plane_joins_lists]
    full_join = JoinConcatenation('concatenate_full', plane_join_jobs)

    # build the global remap
    remap = GlobalRemap('globalmap', full_join)

    # and apply it to every block
    remapped_blocks = [RemapBlock(fb, remap, idx) for idx, fb in fused_blocks.iteritems()]

    Job.run_all()
