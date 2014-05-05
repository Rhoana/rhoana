import sys
import os.path
import os
import subprocess
import datetime
import time
from itertools import product
from collections import defaultdict

USE_SBATCH = True

USE_QSUB = False
QSUB_WORK_QUEUE = 'normal'

MEASURE_PERFORMANCE = False
MAX_TRIES = 10
RUN_LOCAL = False

#Multicore settings
MAX_CORES = 16
MAX_MEMORY_MB = 64000
MIN_TIME = 600
MAX_JOBS_TO_SUBMIT = 100
TIME_FACTOR = 4

class Job(object):
    block_count = 0
    all_jobs = []

    def __init__(self):
        self.name = self.__class__.__name__ + str(len(Job.all_jobs)) + '_' + datetime.datetime.now().isoformat()
        self.jobid = None
        self.output = []
        self.already_done = False
        self.processors = 1
        self.time = 60
        self.memory = 1000
        self.try_count = 0
        Job.all_jobs.append(self)

    def get_done(self):
        if self.already_done:
            return True
        all_outputs = self.output if isinstance(self.output, (list, tuple)) else [self.output]
        if all([os.path.exists(f) for f in all_outputs]):
            self.already_done = True
            return True
        return False

    def dependendencies_done(self):
        for d in self.dependencies:
            if not d.get_done():
                return False
        return True

    def run(self):
        # Make sure output directories exist
        out = self.output
        if isinstance(out, basestring):
            out = [out]
        for f in out:
            if not os.path.isdir(os.path.dirname(f)):
                os.mkdir(os.path.dirname(f))
        if self.get_done():
           return 0
        if self.try_count >= MAX_TRIES:
            return 0

        print "RUN", self.name
        print " ".join(self.command())
        self.try_count += 1

        if RUN_LOCAL:
            subprocess.check_call(self.command())
        elif USE_SBATCH:
            command_list = ["sbatch",
                "-J", self.name,                   # Job name
                "-p", "serial_requeue",            # Work queue (partition) = general / unrestricted / interactive / serial_requeue
                #"-p", "general",                   # Work queue (partition) = general / unrestricted / interactive / serial_requeue
                "--requeue",
                #"--exclude=holy2b05105,hp1301,hp0403",           # Exclude some bad nodes - holy2b05105 did not have scratch2 mapped.
                "-n", str(self.processors),        # Number of processors
                "-t", str(self.time),              # Time in munites 1440 = 24 hours
                "--mem-per-cpu", str(self.memory), # Max memory in MB (strict - attempts to allocate more memory will fail)
                "--open-mode=append",              # Append to log files
                "-o", "logs/out." + self.name,     # Standard out file
                "-e", "logs/error." + self.name]   # Error out file

            if len(self.dependencies) > 0:
                #print command_list
                #print self.dependency_strings()
                command_list = command_list + self.dependency_strings()

            print command_list

            process = subprocess.Popen(command_list,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if MEASURE_PERFORMANCE:
                sbatch_out, sbatch_err = process.communicate("#!/bin/bash\nperf stat -o logs/perf.{0} {1}".format(self.name, " ".join(self.command())))
            else:
                sbatch_out, sbatch_err = process.communicate("#!/bin/bash\n{0}".format(" ".join(self.command())))

            if len(sbatch_err) == 0:
                self.jobid = sbatch_out.split()[3]
                #print 'jobid={0}'.format(self.jobid)

        else:
            subprocess.check_call(["bsub",
                                   "-Q", "all ~0",
                                   "-r",
                                   "-R", "rusage[mem=" + str(self.memory) + "]",
                                   "-g", "/diced_connectome",
                                   "-q", "normal_serial" ,
                                   "-J", self.name,
                                   "-o", "logs/out." + self.name,
                                   "-e", "logs/error." + self.name,
                                   "-w", self.dependency_strings()] +
                                  self.command())
        return 1

    def dependency_strings(self):
        if USE_SBATCH:
            dependency_string = ":".join(d.jobid for d in self.dependencies if not d.get_done())
            if len(dependency_string) > 0:
                return ["-d", "afterok:" + dependency_string]
            return []
        else:
            return " && ".join("done(%s)" % d.name for d in self.dependencies if not d.get_done())

    @classmethod
    def run_all(cls):
        for j in cls.all_jobs:
            j.run()

    @classmethod
    def run_job_blocks(cls, job_block_list, required_cores, required_memory, required_full_time):
        block_name = 'JobBlock{0}.'.format(cls.block_count) + job_block_list[0][0].name
        cls.block_count += 1
        print "RUNNING JOB BLOCK: " + block_name
        print "{0} blocks, {1} jobs, {2} cores, {3}MB memory, {4}m time.".format(
            len(job_block_list), [len(jb) for jb in job_block_list], required_cores, required_memory, required_full_time)
        full_command = "#!/bin/bash\n"
        dependency_set = set()

        # Find all dependencies for all jobs
        for job_block in job_block_list:
            for j in job_block:
                for d in j.dependencies:
                    if not d.get_done() and d.jobid is not None:
                        if USE_SBATCH or USE_QSUB:
                            dependency_set.add(d.jobid)
                        # else:
                        #     dependency_set.add(d.name)

        if USE_SBATCH:
            command_list = ["sbatch",
                "-J", block_name,                   # Job name
                "-p", "serial_requeue",            # Work queue (partition) = general / unrestricted / interactive / serial_requeue
                #"-p", "general",                   # Work queue (partition) = general / unrestricted / interactive / serial_requeue
                "--requeue",
                #"--exclude=holy2b05105,hp1301,hp0403",           # Exclude some bad nodes - holy2b05105 did not have scratch2 mapped.
                "-n", str(required_cores),        # Number of processors
                "-t", str(required_full_time),              # Time in munites 1440 = 24 hours
                "--mem-per-cpu", str(required_memory), # Max memory in MB (strict - attempts to allocate more memory will fail)
                "--open-mode=append",              # Append to log files
                "-o", "logs/out." + block_name,     # Standard out file
                "-e", "logs/error." + block_name]   # Error out file

        elif USE_QSUB:
            command_list = ["qsub"]#,
                # "-N", block_name,                   # Job name
                # "-A", 'hvd113',                    # XSEDE Allocation
                # "-q", QSUB_WORK_QUEUE,                    # Work queue (partition) = general / unrestricted / interactive / serial_requeue
                # "-l", 'nodes=1:ppn={0},walltime={1}:00'.format(str(required_cores), required_full_time),  # Number of processors
                # #"-l", 'walltime={0}:00'.format(self.time),             # Time in munites 1440 = 24 hours
                # #"-l", '-mppmem={0}'.format(self.memory),               # Max memory per cpu in MB (strict - attempts to allocate more memory will fail)
                # "-e", "logs/outerror." + block_name('_')[0],      # Error out file
                # "-j", "eo"]                                            # Join standard out file to error file

            # Better to use file input rather than command line inputs (according to XSEDE helpdesk)
            # Request MAX_CORES so that memory requirement is also met
            full_command += (
               "#PBS -N {0}\n".format(block_name) +
               "#PBS -A hvd113\n" +
               "#PBS -q {0}\n".format(QSUB_WORK_QUEUE) +
               "#PBS -l nodes=1:ppn={0}:native,walltime={1}:00\n".format(str(MAX_CORES), required_full_time) +
               "#PBS -e logs/outerror.{0}\n".format(block_name.split('_')[0]) +
               "#PBS -j eo\n")

        if len(dependency_set) > 0:
            if USE_SBATCH:
                dependency_string = ":".join(d for d in dependency_set)
                if len(dependency_string) > 0:
                    print "depends on jobs:" + dependency_string
                    command_list += ["-d", "afterok:" + dependency_string]
            elif USE_QSUB:
                dependency_string = ":".join(d for d in dependency_set)
                if len(dependency_string) > 0:
                    print "depends on jobs:" + dependency_string
                    full_command += "#PBS -W depend=afterok:" + dependency_string + "\n"
            else:
                command_list += " && ".join("done(%s)" % d for d in dependency_set)

        if USE_SBATCH:
            full_command += "date\n"
        elif USE_QSUB:
            full_command += "cd $PBS_O_WORKDIR\ndate\n"

        # Generate job block commands
        for job_block in job_block_list:
            block_commands = ''
            for j in job_block:
                block_commands += '{0} &\n'.format(' '.join(j.command()))
                print j.name
            full_command += '{0}wait\ndate\n'.format(block_commands)

        # # Test job ids
        # for job_block in job_block_list:
        #     for j in job_block:
        #         j.jobid = str(cls.block_count - 1)

        # print command_list
        # print full_command

        # Submit job
        process = subprocess.Popen(command_list, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        submit_out, submit_err = process.communicate(full_command)

        # Process output
        if len(submit_err) == 0:
            if USE_SBATCH:
                new_jobid = submit_out.split()[3]
            elif USE_QSUB:
                new_jobid = submit_out.split('.')[0]
            print 'jobid={0}'.format(new_jobid)
            for job_block in job_block_list:
                for j in job_block:
                    j.jobid = new_jobid

    @classmethod
    def multicore_run_all(cls):
        jobs_submitted = 0
        job_block_list = [[]]
        required_cores = 0
        required_memory = 0
        required_full_time = 0
        required_block_time = 0
        for j in cls.all_jobs:

            # Make sure output directories exist
            out = j.output
            if isinstance(out, basestring):
                out = [out]
            for f in out:
                if not os.path.isdir(os.path.dirname(f)):
                    os.mkdir(os.path.dirname(f))

            if j.get_done():
                continue

            # Check dependencies
            contains_dependent = False
            for d in j.dependencies:
                if isinstance(d, JobSplit):
                    d = d.job
                if d in job_block_list[-1]:
                    contains_dependent = True
                    break

            # See if we can fit this job into the current multicore job
            if (not contains_dependent and required_cores + j.processors <= MAX_CORES and
                required_memory + j.memory <= MAX_MEMORY_MB):
                # Add this job to the job list
                required_cores += j.processors
                required_memory += j.memory
                required_block_time = max(required_block_time, j.time)
                job_block_list[-1].append(j)
            else:
                #print (contains_dependent, required_cores, required_memory)
                #print (j.processors, j.memory)
                # This block is full - run it or add another
                required_full_time += required_block_time / TIME_FACTOR
                # See if we need more jobs to fill the time
                if (not contains_dependent and required_full_time < MIN_TIME):
                    # Start a new block of jobs
                    job_block_list.append([j])
                    required_cores = j.processors
                    required_memory = j.memory
                    required_block_time = j.time
                else:
                    # Run the current job block list
                    Job.run_job_blocks(job_block_list, required_cores, required_memory, required_full_time)

                    # Limit number of jobs submitted at once
                    jobs_submitted += 1
                    if MAX_JOBS_TO_SUBMIT > 0 and jobs_submitted >= MAX_JOBS_TO_SUBMIT:
                        return

                    # Reset for next block
                    job_block_list = [[j]]
                    required_cores = j.processors
                    required_memory = j.memory
                    required_full_time = 0
                    required_block_time = j.time

        # Run the final (possibly partial) job block list
        if len(job_block_list[0]) > 0:
            required_full_time += required_block_time
            Job.run_job_blocks(job_block_list, required_cores, required_memory, required_full_time)


    @classmethod
    def keep_running(cls):
        all_jobs_complete = False
        cancelled_jobs = {}
        cancelled_requeue_iters = 3
        while not all_jobs_complete:

            all_job_names = {}
            # Generate dictionary of jobs
            for j in cls.all_jobs:
                all_job_names[j.name] = True

            # Find running jobs
            sacct_output = subprocess.check_output(['sacct', '-n', '-o', 'JobID,JobName%100,State%20'])

            pending_running_complete_jobs = {}
            pending = 0
            running = 0
            complete = 0
            failed = 0
            cancelled = 0
            timeout = 0
            other_status = 0
            non_matching = 0

            for job_line in sacct_output.split('\n'):

                job_split = job_line.split()
                if len(job_split) == 0:
                    continue

                job_id = job_split[0]
                job_name = job_split[1]
                job_status = ' '.join(job_split[2:])
                
                if job_name in all_job_names:
                    if job_status in ['PENDING', 'RUNNING', 'COMPLETED']:
                        if job_name in pending_running_complete_jobs:
                            print 'Found duplicate job: ' + job_name
                            dup_job_id, dup_job_status = pending_running_complete_jobs[job_name]
                            print job_id, job_status, dup_job_id, dup_job_status

                            job_to_kill = None
                            if job_status == 'PENDING':
                                job_to_kill = job_id
                            elif dup_job_status == 'PENDING':
                                job_to_kill = dup_job_id
                                pending_running_complete_jobs[job_name] = (job_id, job_status)    

                            if job_to_kill is not None:
                                print 'Canceling job ' + job_to_kill
                                try:
                                    scancel_output = subprocess.check_output(['scancel', '{0}'.format(job_to_kill)])
                                    print scancel_output
                                except:
                                    print "Error canceling job:", sys.exc_info()[0]
                        else:
                            pending_running_complete_jobs[job_name] = (job_id, job_status)
                            if job_status == 'PENDING':
                                pending += 1
                            elif job_status == 'RUNNING':
                                running += 1
                            elif job_status == 'COMPLETED':
                                complete += 1
                    elif job_status in ['FAILED', 'NODE_FAIL']:
                        failed += 1
                    elif job_status in ['CANCELLED', 'CANCELLED+'] or job_status.startswith('CANCELLED'):
                        cancelled += 1

                        # This job could requeued after preemption
                        # Wait cancelled_requeue_iters before requeueing
                        cancelled_iters = 0
                        if job_id in cancelled_jobs:
                            cancelled_iters = cancelled_jobs[job_id]

                        if cancelled_iters < cancelled_requeue_iters:
                            pending_running_complete_jobs[job_name] = (job_id, job_status)
                            cancelled_jobs[job_id] = cancelled_iters + 1

                    elif job_status in ['TIMEOUT']:
                        timeout += 1
                    else:
                        print "Unexpected status: {0}".format(job_status)
                        other_status += 1
                elif job_name not in ['batch', 'true', 'prolog']:
                    non_matching += 1

            run_count = 0
            for j in cls.all_jobs:
                if j.name not in pending_running_complete_jobs and j.dependendencies_done():
                    run_count += j.run()

            print 'Found {0} pending, {1} running, {2} complete, {3} failed, {4} cancelled, {5} timeout, {6} unknown status and {7} non-matching jobs.'.format(
                pending, running, complete, failed, cancelled, timeout, other_status, non_matching)

            print "Queued {0} job{1}.".format(run_count, '' if run_count == 1 else 's')

            if pending > 0 or running > 0 or run_count > 0:
                time.sleep(60)
            else:
                all_jobs_complete = True

class JobSplit(object):
    '''make a multi-output job object look like a single output job'''
    def __init__(self, job, idx):
        self.job = job
        self.idx = idx
        self.name = job.name

    def get_done(self):
        return self.job.get_done()

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

class ClassifySegment_Image(Job):
    def __init__(self, idx, raw_image, classifier_file):
        Job.__init__(self)
        self.already_done = False
        self.raw_image = raw_image
        self.stump_image = raw_image.replace('input_images', 'stump_images')
        self.classifier_file = classifier_file
        self.dependencies = []
        self.memory = 8000
        self.time = 300
        self.features_file = os.path.join('segmentations',
                                      'features_%d.hdf5' % (idx))
        self.prob_file = os.path.join('segmentations',
                                      'probs_%d.hdf5' % (idx))
        self.output = os.path.join('segmentations',
                                   'segs_%d.hdf5' % (idx))
        #self.already_done = os.path.exists(self.output)

    def command(self):
        return ['python',
                os.path.join(os.environ['CONNECTOME'], 'Control', 'segment_image.py'),
                self.raw_image, self.classifier_file, self.stump_image, self.prob_file, self.output]

class Block(Job):
    def __init__(self, segmented_slices, indices, *args):
        Job.__init__(self)
        self.already_done = False
        self.segmented_slices = segmented_slices
        self.dependencies = segmented_slices
        self.memory = 500
        self.time = 30
        self.output = os.path.join('bigdicedblocks', 'block_%d_%d_%d.hdf5' % indices)
        self.args = [str(a) for a in args] + [self.output]
        #self.already_done = os.path.exists(self.output)

    def command(self):
        return ['python', os.path.join(os.environ['CONNECTOME'], 'Control', 'dice_block.py')] + self.args + [s.output for s in self.segmented_slices]

class FusedBlock(Job):
    def __init__(self, block, indices, global_block_number):
        Job.__init__(self)
        self.already_done = False
        self.block = block
        self.global_block_number = global_block_number
        self.dependencies = [block]
        self.processors = 4
        self.memory = 16000
        # memory is per proc, so we are requesting 64GB here (and sometimes use it)
        #self.time = 360
        self.time = 480
        self.indices = indices
        self.output = os.path.join('bigfusedblocks', 'fusedblock_%d_%d_%d.hdf5' % indices)
        #self.already_done = os.path.exists(self.output)

    def command(self):
        return ['python',
                os.path.join(os.environ['CONNECTOME'], 'WindowFusion', 'window_fusion_cpx.py'),
                self.block.output,
                str(self.global_block_number),
                self.output]

class CleanBlock(Job):
    def __init__(self, fusedblock):
        Job.__init__(self)
        self.already_done = False
        self.indices = fusedblock.indices
        self.block = fusedblock.block
        self.global_block_number = fusedblock.global_block_number
        self.dependencies = [fusedblock]
        self.memory = 6000
        #self.memory = 8000
        self.time = 60
        self.inputlabels = fusedblock.output
        self.inputprobs = fusedblock.block.output
        self.output = os.path.join('cleanedblocks', 'block_%d_%d_%d.hdf5' % self.indices)
        #self.already_done = os.path.exists(self.output)

    def command(self):
        return [os.path.join(os.environ['CONNECTOME'], 'Control', 'clean_block.sh'), self.inputlabels, self.inputprobs, self.output]

class PairwiseMatching(Job):
    def __init__(self, fusedblock1, fusedblock2, direction, even_or_odd, halo_width):
        Job.__init__(self)
        self.direction = direction
        self.already_done = False
        self.even_or_odd = even_or_odd
        self.halo_width = halo_width
        self.indices = (fusedblock1.indices, fusedblock2.indices)
        self.dependencies = [fusedblock1, fusedblock2]
        #self.memory = 16000
        self.memory = 8000
        #self.memory = 4000
        self.time = 60
        outdir = 'pairwise_matches_%s_%s' % (['X', 'Y', 'Z',][direction], even_or_odd)
        self.output = (os.path.join(outdir, os.path.basename(fusedblock1.output)),
                       os.path.join(outdir, os.path.basename(fusedblock2.output)))
        #self.already_done = os.path.exists(self.output[0]) and os.path.exists(self.output[1])

    def command(self):
        return [os.path.join(os.environ['CONNECTOME'], 'Control', 'pairwise_match_labels.sh')] + [d.output for d in self.dependencies] + \
            [str(self.direction + 1), # matlab
             str(self.halo_width)] + list(self.output)

class JoinConcatenation(Job):
    def __init__(self, outfilename, inputs):
        Job.__init__(self)
        self.already_done = False
        self.dependencies = inputs
        self.memory = 1000
        self.time = 60
        self.output = os.path.join('joins', outfilename)
        #self.already_done = os.path.exists(self.output)
        
    def command(self):
        return [os.path.join(os.environ['CONNECTOME'], 'Control', 'concatenate_joins.sh')] + \
            [s.output for s in self.dependencies] + \
            [self.output]

class GlobalRemap(Job):
    def __init__(self, outfilename, joinjob):
        Job.__init__(self)
        self.already_done = False
        self.dependencies = [joinjob]
        self.memory = 1000
        self.time = 60
        self.joinfile = joinjob.output
        self.output = os.path.join('joins', outfilename)
        #self.already_done = os.path.exists(self.output)

    def command(self):
        return [os.path.join(os.environ['CONNECTOME'], 'Control', 'create_global_map.sh'), self.joinfile, self.output]

class RemapBlock(Job):
    def __init__(self, blockjob, build_remap_job, indices):
        Job.__init__(self)
        self.already_done = False
        self.dependencies = [blockjob, build_remap_job]
        #self.memory = 2000
        self.memory = 4000
        #self.memory = 8000
        self.time = 60
        self.inputfile = blockjob.output
        self.mapfile = build_remap_job.output
        self.indices = indices
        self.output = os.path.join('relabeledblocks', 'block_%d_%d_%d.hdf5' % indices)
        #self.already_done = os.path.exists(self.output)

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
        self.memory = 1000
        self.time = 60
        self.zoffset = zoffset
        self.xy_halo = xy_halo
        self.image_size = image_size
        self.xy_block_size = xy_block_size
        self.output = os.path.join('output_labels', 'labels_%05d.tif' % zplane)
        #self.already_done = os.path.exists(self.output)

    def generate_args(self):
        for block in self.dependencies:
            # XY corner followed by filename
            yield str(block.indices[0] * self.xy_block_size)
            yield str(block.indices[1] * self.xy_block_size)
            yield block.output

    def command(self):
        return [os.path.join(os.environ['CONNECTOME'], 'Control', 'extract_label_plane.sh'), self.output, str(self.image_size), str(self.zoffset), str(self.xy_halo)] + \
            list(self.generate_args())

class ExtractOverlayPlane(Job):
    def __init__(self, zplane, xy_halo, remapped_blocks, zoffset, image_size, xy_block_size, input_image_path):
        Job.__init__(self)
        self.already_done = False
        self.dependencies = remapped_blocks
        self.memory = 4000
        self.time = 60
        self.zoffset = zoffset
        self.xy_halo = xy_halo
        self.image_size = image_size
        self.xy_block_size = xy_block_size
        self.input_image_path = input_image_path
        self.output = os.path.join('output_overlay', 'overlay_%05d.png' % zplane)
        #self.already_done = os.path.exists(self.output)

    def generate_args(self):
        for block in self.dependencies:
            # XY corner followed by filename
            yield str(block.indices[0] * self.xy_block_size)
            yield str(block.indices[1] * self.xy_block_size)
            yield block.output

    def command(self):
        return [os.path.join(os.environ['CONNECTOME'], 'Control', 'extract_overlay_plane.sh'), self.output, self.input_image_path, str(self.image_size), str(self.zoffset), str(self.xy_halo)] + \
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

    assert 'CONNECTOME' in os.environ
    #assert 'VIRTUAL_ENV' in os.environ

    # Default settings
    image_size = 2048
    probability_subimage_size = 1024
    probability_subimage_halo = 32
    segmentation_subimage_size = 1024
    segmentation_subimage_halo = 128
    block_xy_halo = 64
    block_xy_size = 512 - (2 * 64)
    block_z_size = 52
    block_z_halo = 6

    classifier_file = os.path.join(os.environ['CONNECTOME'], 'DeepNets', 'deep_net_combo3_13November2013.h5')

    settings_file = sys.argv[1]
    os.environ['CONNECTOME_SETTINGS'] = settings_file
    execfile(settings_file)

    images = [f.rstrip() for f in open(sys.argv[2])]

    segmentations = [ClassifySegment_Image(idx, im, classifier_file)
                    for idx, im in enumerate(images)]

    #segmentations = [f.rstrip() for f in open(sys.argv[2])]
    #print segmentations
    
    # Dice full volume
    blocks = {}
    nblocks_x = (image_size - 2 * block_xy_halo) / block_xy_size
    nblocks_y = (image_size - 2 * block_xy_halo) / block_xy_size
    nblocks_z = (len(segmentations) - 2 * block_z_halo) / block_z_size
    for block_idx_z in range(nblocks_z):
        lo_slice = block_idx_z * block_z_size
        hi_slice = lo_slice + block_z_size + 2 * block_z_halo
        for block_idx_x in range(nblocks_x):
            xlo = block_idx_x * block_xy_size
            xhi = xlo + block_xy_size + 2 * block_xy_halo
            for block_idx_y in range(nblocks_y):
                ylo = block_idx_y * block_xy_size
                yhi = ylo + block_xy_size + 2 * block_xy_halo
                print "Making block {0}, slice {1}, crop {2}.".format(
                    (block_idx_x, block_idx_y, block_idx_z),
                    (lo_slice, hi_slice),
                    (xlo, ylo, xhi, yhi))
                blocks[block_idx_x, block_idx_y, block_idx_z] = \
                    Block(segmentations[lo_slice:hi_slice],
                          (block_idx_x, block_idx_y, block_idx_z),
                          xlo, ylo, xhi, yhi)

    # Window fuse all blocks
    # Generate block id based on on block index with z as most significant (allows additional slabs to be added later)
    fused_blocks = dict((idxs, FusedBlock(block, idxs,
        idxs[0] + idxs[1] * nblocks_x + idxs[2] * nblocks_x * nblocks_y)) for (idxs, block) in blocks.iteritems())

    # Cleanup all blocks (remove small or completely enclosed segments)
    cleaned_blocks = dict((idxs, CleanBlock(fb)) for idxs, fb in fused_blocks.iteritems())
    #cleaned_blocks = fused_blocks

    # Pairwise match all blocks.
    #
    # We overwrite each block in cleaned_blocks (the python dict, not the file)
    # with the output of the pairwise matching, and work in non-overlapping
    # sets (even-to-odd, then odd-to-even)
    for direction in range(3):  # X, Y, Z
        for wpidx, which_pairs in enumerate(['even', 'odd']):
            for idx in cleaned_blocks:
                if (idx[direction] % 2) == wpidx:  # merge even-to-odd, then odd-to-even
                    neighbor_idx = list(idx)
                    neighbor_idx[direction] += 1  # check neighbor exists
                    neighbor_idx = tuple(neighbor_idx)
                    if neighbor_idx in cleaned_blocks:
                        pw = PairwiseMatching(cleaned_blocks[idx], cleaned_blocks[neighbor_idx],
                                              direction,  # matlab
                                              which_pairs,
                                              block_xy_halo if direction < 2 else block_z_halo)
                        # we can safely overwrite (variables, not files)
                        # because of nonoverlapping even/odd sets
                        cleaned_blocks[idx] = JobSplit(pw, 0)
                        cleaned_blocks[neighbor_idx] = JobSplit(pw, 1)

    # Contatenate the joins from all the blocks to a single file, for building
    # the global remap.  Work first on XY planes, to add some parallelism and
    # limit number of command arguments.
    plane_joins_lists = {}
    for idxs, block in cleaned_blocks.iteritems():
        plane_joins_lists[idxs[2]] = plane_joins_lists.get(idxs[2], []) + [block]
    plane_join_jobs = [JoinConcatenation('concatenate_Z_%d' % idx, plane_joins_lists[idx])
                       for idx in plane_joins_lists]
    full_join = JoinConcatenation('concatenate_full', plane_join_jobs)

    # build the global remap
    remap = GlobalRemap('globalmap', full_join)

    # and apply it to every block
    remapped_blocks = [RemapBlock(fb, remap, idx) for idx, fb in cleaned_blocks.iteritems()]
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

    # optional, render overlay images
    output_labels = [ExtractOverlayPlane(idx, block_xy_halo,
                                       remapped_blocks_by_plane[min(idx / block_z_size, max_zslab)],
                                       idx - block_z_size * min(idx / block_z_size, max_zslab),  # offset within block
                                       image_size, block_xy_size, im)
                    for idx, im in enumerate(images)]

    # # Render fused blocks directly
    # cleaned_blocks_by_plane = defaultdict(list)
    # for idx, fb in cleaned_blocks.iteritems():
    #     cleaned_blocks_by_plane[fb.indices[2]] += [fb]
    # max_zslab = max(cleaned_blocks_by_plane.keys())
    # fused_output_labels = [ExtractLabelPlane(idx,
    #                                    cleaned_blocks_by_plane[min(idx / block_z_size, max_zslab)],
    #                                    idx - block_z_size * min(idx / block_z_size, max_zslab),  # offset within block
    #                                    image_size, block_xy_size)
    #                  for idx, _ in enumerate(segmentations)]

    if '-l' in sys.argv:
        RUN_LOCAL = True
        sys.argv.remove('-l')
    if '--local' in sys.argv:
        RUN_LOCAL = True
        sys.argv.remove('--local')
    if len(sys.argv) == 3:
        Job.run_all()
    elif '-k' in sys.argv or '--keeprunning' in sys.argv:
        # Monitor job status and requeue as necessary
        Job.keep_running()
    elif '-m' in sys.argv or '--multicore' in sys.argv:
        if RUN_LOCAL:
            print "ERROR: --local cannot be used with --multicore (not yet implemented)."
            return
        # Bundle jobs for multicore nodes
        Job.multicore_run_all()
    else:
        for j in Job.all_jobs:
            if j.output == sys.argv[3] or sys.argv[3] in j.output or sys.argv[3] in j.output[0] or sys.argv[3] in j.name:
                for k in j.dependencies:
                    if k.output != sys.argv[3] and sys.argv[3] not in k.output and sys.argv[3] not in k.output[0] and sys.argv[3] not in k.name:
                        k.already_done = True
                j.run()

    
