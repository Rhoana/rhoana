import sys
import os.path
import os
import subprocess
import datetime
from itertools import product
from collections import defaultdict

FUSED_QUEUE = "normal_serial"

class Job(object):
    all_jobs = []

    def __init__(self):
        self.name = self.__class__.__name__ + str(len(Job.all_jobs)) + '_' + datetime.datetime.now().isoformat()
        self.already_done = False
        self.processors = 1
        self.memory = 1000
        Job.all_jobs.append(self)

    def run(self):
        # Make sure output directories exist
        out = self.output
        if isinstance(out, basestring):
            out = [out]
        for f in out:
            if not os.path.isdir(os.path.dirname(f)):
                os.mkdir(os.path.dirname(f))
        if self.already_done:
           return
        print "RUN", self.name
        print " ".join(self.command())

        command_list = ["sbatch",
            "-J", self.name,                   # Job name
            "-p", "serial_requeue",            # Work queue
            "--requeue",
            "-n", str(self.processors),        # Number of processors
            "-t", "180",                       # Time in munites 1440 = 24 hours
            "--mem-per-cpu", str(self.memory), # Max memory in MB (strict - attempts to allocate more memory will fail)
            "-o", "logs/out." + self.name,     # Standard out file
            "-e", "logs/error." + self.name]   # Error out file

        if len(self.dependencies) > 0:
            #print command_list
            #print self.dependency_strings()
            command_list = command_list + self.dependency_strings()

        print command_list

        process = subprocess.Popen(command_list,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        sbatch_out, sbatch_err = process.communicate("#!/bin/bash\n{0}".format(" ".join(self.command())))

        print sbatch_out
        print sbatch_err

        if len(sbatch_err) == 0:
            self.jobid = sbatch_out.split()[3]

    def dependency_strings(self):
        dependency_string = ":".join(d.jobid for d in self.dependencies if not d.already_done)
        if len(dependency_string) > 0:
            return ["-d", "afterany:" + dependency_string]
        return []

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

    def get_done(self):
        return self.job.already_done

    def set_done(self, val):
        self.job.already_done = val

    already_done = property(get_done, set_done)
    
    @property
    def output(self):
        return self.job.output[self.idx]

    @property
    def indices(self):
        return self.job.indices[self.idx]

    @property
    def jobid(self):
        return self.job.jobid
    @jobid.setter
    def jobid(self, value):
        self.job.jobid = value
    

class Reassemble(Job):
    '''reassemble a diced job'''
    def __init__(self, dataset, output_sizes, joblist, output):
        Job.__init__(self)
        self.output_sizes = output_sizes
        self.dataset = dataset
        self.dependencies = joblist
        self.memory = 4000
        self.output = output
        self.already_done = False

    def command(self):
        return ['./reassemble.sh', self.dataset,
                str(len(self.output_sizes))] + \
            [str(s) for s in self.output_sizes] + \
            [j.output for j in self.dependencies] + \
            [self.output]

class Subimage_ProbabilityMap(Job):
    def __init__(self, raw_image, idx, xlo, ylo, xhi, yhi, xlo_core, ylo_core, xhi_core, yhi_core):
        Job.__init__(self)
        self.already_done = False
        self.raw_image = raw_image
        self.dependencies = []
        self.memory = 4000
        self.coords = [str(c) for c in (xlo, ylo, xhi, yhi)]
        self.core_coords = [str(c) for c in (xlo_core, ylo_core, xhi_core, yhi_core)]
        self.output = os.path.join('subimage_probabilities',
                                   'probs_%d_%s.hdf5' % (idx, '_'.join(self.coords)))

    def command(self):
        return ['python', 'compute_probabilities.py', self.raw_image, self.output] + \
            self.coords + self.core_coords

class Subimage_SegmentedSlice(Job):
    def __init__(self, idx, probability_map, raw_image, xlo, ylo, xhi, yhi, xlo_core, ylo_core, xhi_core, yhi_core):
        Job.__init__(self)
        self.already_done = False
        self.probability_map = probability_map
        self.raw_image = raw_image
        self.dependencies = [self.probability_map]
        self.memory = 4000
        self.coords = [str(c) for c in (xlo, ylo, xhi, yhi)]
        self.core_coords = [str(c) for c in (xlo_core, ylo_core, xhi_core, yhi_core)]
        self.output = os.path.join('subimage_segmentations',
                                   'segs_%d_%s.hdf5' % (idx, '_'.join(self.coords)))

    def command(self):
        return ['python', 'segment_image.py', self.raw_image, self.probability_map.output, self.output] + \
            self.coords + self.core_coords

class ClassifySegement_Image(Job):
    def __init__(self, idx, raw_image, classifier_file):
        Job.__init__(self)
        self.already_done = False
        self.raw_image = raw_image
        self.classifier_file = classifier_file
        self.dependencies = []
        self.memory = 4000
        self.prob_file = os.path.join('segmentations',
                                      'probs_%d.hdf5' % (idx))
        self.output = os.path.join('segmentations',
                                   'segs_%d.hdf5' % (idx))

    def command(self):
        return ['python',
                os.path.join(os.environ['CONNECTOME'], 'Control', 'segment_image.py'),
                self.raw_image, self.classifier_file, self.prob_file, self.output]


class Block(Job):
    def __init__(self, segmented_slices, indices, *args):
        Job.__init__(self)
        self.already_done = False
        self.segmented_slices = segmented_slices
        self.dependencies = []#segmented_slices
        self.memory = 4000
        self.output = os.path.join('bigdicedblocks', 'block_%d_%d_%d.hdf5' % indices)
        self.args = [str(a) for a in args] + [self.output]
        self.already_done = os.path.exists(self.output)

    def command(self):
        return ['python', os.path.join(os.environ['CONNECTOME'], 'Control', 'dice_block.py')] + self.args + [s for s in self.segmented_slices]

class FusedBlock(Job):
    def __init__(self, block, indices, global_block_number):
        Job.__init__(self)
        self.already_done = False
        self.block = block
        self.global_block_number = global_block_number
        self.dependencies = [block]
        self.processors = 4
        self.memory = 16000
        self.indices = indices
        self.output = os.path.join('bigfusedblocks', 'fusedblock_%d_%d_%d.hdf5' % indices)
        self.already_done = os.path.exists(self.output)

    def command(self):
        return ['python',
                os.path.join(os.environ['CONNECTOME'], 'WindowFusion', 'window_fusion_cpx.py'),
                self.block.output,
                str(self.global_block_number),
                self.output]

class PairwiseMatching(Job):
    def __init__(self, fusedblock1, fusedblock2, direction, even_or_odd, halo_width):
        Job.__init__(self)
        self.direction = direction
        self.already_done = False
        self.even_or_odd = even_or_odd
        self.halo_width = halo_width
        self.indices = (fusedblock1.indices, fusedblock2.indices)
        self.dependencies = [fusedblock1, fusedblock2]
        self.memory = 4000
        outdir = 'pairwise_matches_%s_%s' % (['X', 'Y', 'Z',][direction], even_or_odd)
        self.output = (os.path.join(outdir, os.path.basename(fusedblock1.output)),
                       os.path.join(outdir, os.path.basename(fusedblock2.output)))
        self.already_done = os.path.exists(self.output[0]) and os.path.exists(self.output[1])

    def command(self):
        return [os.path.join(os.environ['CONNECTOME'], 'Control', 'pairwise_match_labels.sh')] + [d.output for d in self.dependencies] + \
            [str(self.direction + 1), # matlab
             str(self.halo_width)] + list(self.output)

class JoinConcatenation(Job):
    def __init__(self, outfilename, inputs):
        Job.__init__(self)
        self.already_done = False
        self.dependencies = inputs
        self.memory = 16000
        self.output = os.path.join('joins', outfilename)

    def command(self):
        return [os.path.join(os.environ['CONNECTOME'], 'Control', 'concatenate_joins.sh')] + \
            [s.output for s in self.dependencies] + \
            [self.output]

class GlobalRemap(Job):
    def __init__(self, outfilename, joinjob):
        Job.__init__(self)
        self.already_done = False
        self.dependencies = [joinjob]
        self.memory = 32000
        self.joinfile = joinjob.output
        self.output = os.path.join('joins', outfilename)
        self.already_done = os.path.exists(self.output)

    def command(self):
        return [os.path.join(os.environ['CONNECTOME'], 'Control', 'create_global_map.sh'), self.joinfile, self.output]

class RemapBlock(Job):
    def __init__(self, blockjob, build_remap_job, indices):
        Job.__init__(self)
        self.already_done = False
        self.dependencies = [blockjob, build_remap_job]
        self.memory = 4000
        self.inputfile = blockjob.output
        self.mapfile = build_remap_job.output
        self.indices = indices
        self.output = os.path.join('relabeledblocks', 'block_%d_%d_%d.hdf5' % indices)
        self.already_done = os.path.exists(self.output)

    def command(self):
        return [os.path.join(os.environ['CONNECTOME'], 'Control', 'remap_block.sh'), self.inputfile, self.mapfile, self.output]

class CopyImage(Job):
    def __init__(self, input, idx):
        Job.__init__(self)
        self.already_done = False
        self.dependencies = []
        self.memory = 4000
        self.inputfile = input
        self.idx = idx
        self.output = os.path.join('output_images', 'image_%05d.tif' % idx)

    def command(self):
        return ['/bin/cp', self.inputfile, self.output]

class ExtractLabelPlane(Job):
    def __init__(self, zplane, xy_halo, remapped_blocks, zoffset, image_size, xy_block_size):
        Job.__init__(self)
        self.already_done = False
        self.dependencies = remapped_blocks
        self.memory = 4000
        self.zoffset = zoffset
        self.xy_halo = xy_halo
        self.image_size = image_size
        self.xy_block_size = xy_block_size
        self.output = os.path.join('output_labels', 'labels_%05d.tif' % zplane)
        self.already_done = os.path.exists(self.output)

    def generate_args(self):
        for block in self.dependencies:
            # XY corner followed by filename
            yield str(block.indices[0] * self.xy_block_size)
            yield str(block.indices[1] * self.xy_block_size)
            yield block.output

    def command(self):
        return [os.path.join(os.environ['CONNECTOME'], 'Control', 'extract_label_plane.sh'), self.output, str(self.image_size), str(self.zoffset), str(self.xy_halo)] + \
            list(self.generate_args())


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

    block_xy_halo = 64
    block_xy_size = 512 - (2 * 64)
    block_z_size = 52
    block_z_halo = 6

    assert 'CONNECTOME' in os.environ
    #assert 'VIRTUAL_ENV' in os.environ

    images = [f.rstrip() for f in open(sys.argv[1])]
    classifier_file = os.path.join(os.environ['CONNECTOME'], 'ClassifyMembranes', 'GB_classifier.txt')

    segmentations = [ClassifySegement_Image(idx, im, classifier_file)
                    for idx, im in enumerate(images)]

    print(sys.argv[1])
    segmentations = [f.rstrip() for f in open(sys.argv[1])]

    #print segmentations
    
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
                          xlo, ylo, xhi, yhi)

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
                        # we can safely overwrite (variables, not files)
                        # because of nonoverlapping even/odd sets
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
    remapped_blocks_by_plane = defaultdict(list)
    for bl in remapped_blocks:
        remapped_blocks_by_plane[bl.indices[2]] += [bl]

    # finally, extract the images and output labels
    # output_images = [CopyImage(i, idx) for idx, i in enumerate(images)]
    max_zslab = max(remapped_blocks_by_plane.keys())
    output_labels = [ExtractLabelPlane(idx, block_xy_halo,
                                       remapped_blocks_by_plane[min(idx / block_z_size, max_zslab)],
                                       idx - block_z_size * min(idx / block_z_size, max_zslab),  # offset within block
                                       image_size, block_xy_size)
                     for idx, _ in enumerate(segmentations)]

    # # Render fused blocks directly
    # fused_blocks_by_plane = defaultdict(list)
    # for idx, fb in fused_blocks.iteritems():
    #     fused_blocks_by_plane[fb.indices[2]] += [fb]
    # max_zslab = max(fused_blocks_by_plane.keys())
    # fused_output_labels = [ExtractLabelPlane(idx,
    #                                    fused_blocks_by_plane[min(idx / block_z_size, max_zslab)],
    #                                    idx - block_z_size * min(idx / block_z_size, max_zslab),  # offset within block
    #                                    image_size, block_xy_size)
    #                  for idx, _ in enumerate(segmentations)]

    if len(sys.argv) == 2:
        Job.run_all()
    else:
        for j in Job.all_jobs:
            if j.output == sys.argv[2] or sys.argv[2] in j.output or sys.argv[2] in j.output[0]:
                for k in j.dependencies:
                    if k.output != sys.argv[2] and sys.argv[2] not in k.output and sys.argv[2] not in k.output[0]:
                        k.already_done = True
                j.run()
                
