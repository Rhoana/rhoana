import sys
import os.path
import os
import glob
import subprocess
import datetime
import time
import random
from itertools import product
from collections import defaultdict
import numpy as np

USE_SBATCH = True
# SBATCH_PARTITION_LIST = ['general']
SBATCH_PARTITION_LIST = ['serial_requeue']
#SBATCH_PARTITION_LIST = ['general', 'serial_requeue']
#SBATCH_PARTITION_LIST = ['general', 'general', 'serial_requeue']
SBATCH_GPU_PARTITION_LIST = ["gpgpu", "resonance"]
SBATCH_ACCOUNT = None #(Use default account for current user)
#SBATCH_ACCOUNT = 'pfister_lab'

USE_QSUB = False
QSUB_WORK_QUEUE = 'normal'

MEASURE_PERFORMANCE = False
MAX_TRIES = 10
RUN_LOCAL = False
MEMORY_FACTOR = 1

#Multicore settings
MAX_CORES = 4
MAX_MEMORY_MB = 16000
MIN_TIME = 180
MAX_JOBS_TO_SUBMIT = 1000
TIME_FACTOR = 1

######## filename cache start ########
# - The Odyssey /n/regal file system is slow to report os.path.exists() but fast to return simple ls results.
# - (Because os.path.exists() invokes os.stat(), requesting full file info including size etc.)
# - os.stat() is only slow the first time for each file and is not so bad for a small number (1-10k) of files.
# - For large numbers (100k+) of files this makes checking which jobs have completed very slow.
# - Here we maintain a dictionary of files that exist on startup using the faster os.listdir()

subdirs = [
    '../lgn8000/bigfusedblocks',
    '../lgn8000/relabeledblocks',
    '../lgn8000v2/bigfusedblocks',
    'bigdicedblocks',
    'bigfusedblocks',
    'combofusedblocks',
    'cleanedblocks',
    'joins',
    'output_labels',
    'output_overlay',
    'output_index',
    'oversegmentations',
    'pairwise_matches_X_even',
    'pairwise_matches_X_odd',
    'pairwise_matches_Y_even',
    'pairwise_matches_Y_odd',
    'pairwise_matches_Z_even',
    'pairwise_matches_Z_odd',
    'relabeledblocks',
    'segmentations']

existing_files_dict = {}
non_existing_files_dict = {}
file_recheck_time_seconds = 60

def refresh_file_cache():
    start_time = time.time()
    for subdir in subdirs:
        if os.path.exists(subdir):
            files = os.listdir(subdir)
            for f in files:
                existing_files_dict[os.path.join(subdir, f)] = None
    print "loaded existing files in {0} seconds.".format(time.time() - start_time)

#refresh_file_cache()

def fast_exists(fname, check_time, cache_only=False):
    if fname in existing_files_dict:
        return True
    elif cache_only:
        return False
    elif fname in non_existing_files_dict:
        if non_existing_files_dict[fname] > check_time:
            return False
        elif os.path.exists(fname):
            del non_existing_files_dict[fname]
            existing_files_dict[fname] = None
            return True
        else:
            non_existing_files_dict[fname] = check_time + file_recheck_time_seconds
            return False
    elif os.path.exists(fname):
        existing_files_dict[fname] = None
        return True
    else:
        non_existing_files_dict[fname] = check_time + file_recheck_time_seconds
        return False

######## filename cache end ########


class Job(object):
    run_name = os.path.split(os.getcwd())[1]
    block_count = 0
    all_jobs = []

    def __init__(self):
        # Allow for recovery if driver script fails - use deterministic job names.
        self.name = Job.run_name + '.' + self.__class__.__name__ + str(len(Job.all_jobs)) # + '_' + datetime.datetime.now().isoformat()
        self.jobid = None
        self.output = []
        self.already_done = False
        self.processors = 1
        self.gpus = 0
        self.time = 60
        self.memory = 1000
        self.try_count = 0
        Job.all_jobs.append(self)

    def get_done(self, check_time=time.time(), cache_only=False):
        if self.already_done:
            return True
        all_outputs = self.output if isinstance(self.output, (list, tuple)) else [self.output]
        for f in all_outputs:
            if not fast_exists(f, check_time, cache_only):
                return False
        self.already_done = True
        return True

    def dependendencies_done(self, check_time=time.time(), cache_only=False):
        for d in self.dependencies:
            if not d.get_done(check_time, cache_only):
                return False
        return True

    def run(self, partition_list=SBATCH_PARTITION_LIST, gpu_partition_list=SBATCH_GPU_PARTITION_LIST):
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

            partition = random.choice(partition_list)
            if self.gpus > 0:
                #partition = "gpgpu"
                partition = random.choice(gpu_partition_list)

            command_list = ["sbatch",
                "-J", self.name,                   # Job name
                "-p", partition,                   # Work queue (partition) = general / unrestricted / interactive / serial_requeue
                "--requeue",
                #"--exclude=holy2b05105,hp1301,hp0403",           # Exclude some bad nodes - holy2b05105 did not have scratch2 mapped.
                "-n", str(self.processors),        # Number of processors
                "-t", str(self.time),              # Time in munites 1440 = 24 hours
                "--mem-per-cpu", str(int(self.memory * MEMORY_FACTOR)), # Max memory in MB (strict - attempts to allocate more memory will fail)
                "--open-mode=append",              # Append to log files
                "-o", "logs/outerror." + self.name,     # Standard out file
                "-e", "logs/outerror." + self.name]   # Error out file

            if self.gpus > 0:
                command_list += ["--gres=gpu:{0}".format(self.gpus)]
                #TODO configure to run on aagpu[01-08] nodes
                if partition == "gpgpu":
                    #command_list += ["--nodelist=holygpu[01-16]"]
                    command_list += ["--exclude=aaggpu[01-08]"]

            if SBATCH_ACCOUNT is not None:
                command_list += ["--account={0}".format(SBATCH_ACCOUNT)]

            if len(self.dependencies) > 0:
                #print command_list
                #print self.dependency_strings()
                command_list = command_list + self.dependency_strings()

            print command_list

            process = subprocess.Popen(command_list,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            job_command = " ".join(self.command())

            # Setup GPU environment
            if self.gpus > 0:
                job_command = "source ~/dev/setup_env_generic.sh\nwhich python\nhostname\nnvcc --version\n" + job_command

            if MEASURE_PERFORMANCE:
                sbatch_out, sbatch_err = process.communicate("#!/bin/bash\nperf stat -o logs/perf.{0} {1}".format(self.name, job_command))
            else:
                sbatch_out, sbatch_err = process.communicate("#!/bin/bash\n{0}".format(job_command))

            if len(sbatch_err) == 0:
                self.jobid = sbatch_out.split()[3]
                #print 'jobid={0}'.format(self.jobid)

        else:
            subprocess.check_call(["bsub",
                                   "-Q", "all ~0",
                                   "-r",
                                   "-R", "rusage[mem=" + str(int(self.memory * MEMORY_FACTOR)) + "]",
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
    def run_job_blocks(cls, job_block_list, required_cores, required_gpus, required_memory, required_full_time,
        partition_list=SBATCH_PARTITION_LIST, gpu_partition_list=SBATCH_GPU_PARTITION_LIST):
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

            partition = random.choice(partition_list)
            if required_gpus > 0:
                #partition = "gpgpu"
                partition = random.choice(gpu_partition_list)

            command_list = ["sbatch",
                "-J", block_name,                   # Job name
                "-p", partition,            # Work queue (partition) = general / unrestricted / interactive / serial_requeue
                #"-p", "general",                   # Work queue (partition) = general / unrestricted / interactive / serial_requeue
                "--requeue",
                #"--exclude=holy2b05105,hp1301,hp0403",           # Exclude some bad nodes - holy2b05105 did not have scratch2 mapped.
                "-n", str(required_cores),        # Number of processors
                "-t", str(required_full_time),              # Time in munites 1440 = 24 hours
                "--mem-per-cpu", str(int(required_memory * MEMORY_FACTOR)), # Max memory in MB (strict - attempts to allocate more memory will fail)
                "--open-mode=append",              # Append to log files
                "-o", "logs/out." + block_name,     # Standard out file
                "-e", "logs/error." + block_name]   # Error out file

            if required_gpus > 0:
                command_list += ["--gres=gpu:{0}".format(required_gpus)]
                #TODO configure to run on aagpu[01-08] nodes
                if partition == "gpgpu":
                    #command_list += ["--nodelist=holygpu[01-16]"]
                    command_list += ["--exclude=aaggpu[01-08]"] 

            if SBATCH_ACCOUNT is not None:
                command_list += ["--account={0}".format(SBATCH_ACCOUNT)]

        elif USE_QSUB:
            command_list = ["qsub"]#,
                # "-N", block_name,                   # Job name
                # "-A", 'hvd113',                    # XSEDE Allocation
                # "-q", QSUB_WORK_QUEUE,                    # Work queue (partition) = general / unrestricted / interactive / serial_requeue
                # "-l", 'nodes=1:ppn={0},walltime={1}:00'.format(str(required_cores), required_full_time),  # Number of processors
                # #"-l", 'walltime={0}:00'.format(self.time),             # Time in munites 1440 = 24 hours
                # #"-l", '-mppmem={0}'.format(int(self.memory * MEMORY_FACTOR)),               # Max memory per cpu in MB (strict - attempts to allocate more memory will fail)
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

        # Setup GPU environment and print gpu debug messages
        if required_gpus > 0:
            full_command += "source ~/dev/setup_env_generic.sh\nwhich python\nhostname\nnvcc --version\n"

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

        return block_name

    @classmethod
    def multicore_run_list(cls, runnable_jobs, run_partial_blocks=True, partition_list=SBATCH_PARTITION_LIST, gpu_partition_list=SBATCH_GPU_PARTITION_LIST):
        submit_count = 0
        submitted_job_blocks = {}
        job_block_list = [[]]
        required_cores = 0
        required_memory = 0
        required_full_cores = 0
        required_full_memory = 0
        required_full_time = 0
        required_block_time = 0
        required_gpus = 0
        for j in runnable_jobs:

            # Make sure output directories exist
            out = j.output
            if isinstance(out, basestring):
                out = [out]
            for f in out:
                if not os.path.isdir(os.path.dirname(f)):
                    os.mkdir(os.path.dirname(f))

            if j.get_done(cache_only=True):
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
            if (not contains_dependent and
                (required_cores + j.processors <= MAX_CORES or required_cores == 0) and
                (required_memory + j.memory <= MAX_MEMORY_MB or required_memory == 0) and
                j.gpus == 0) or len(job_block_list[0]) == 0):
                # Add this job to the job list
                required_cores += j.processors
                required_gpus = j.gpus
                required_memory += j.memory
                required_block_time = max(required_block_time, j.time)
                job_block_list[-1].append(j)
            else:
                #print (contains_dependent, required_cores, required_memory)
                #print (j.processors, j.memory)
                # This block is full - run it or add another
                required_full_cores = max(required_full_cores, required_cores)
                required_full_memory = max(required_full_memory, required_memory)
                required_full_time += required_block_time / TIME_FACTOR
                # See if we need more jobs to fill the time
                if (not contains_dependent and required_full_time < MIN_TIME and
                    (required_gpus == j.gpus or len(job_block_list[0]) == 0)):
                    # Start a new block of jobs
                    job_block_list.append([j])
                    required_cores = j.processors
                    required_gpus = j.gpus
                    required_memory = j.memory
                    required_block_time = j.time
                else:
                    # Run the current job block list
                    block_name = Job.run_job_blocks(job_block_list, required_full_cores, required_gpus, required_full_memory, required_full_time,
                        partition_list=partition_list, gpu_partition_list=gpu_partition_list)
                    submitted_job_blocks[block_name] = job_block_list

                    # Limit number of jobs submitted at once
                    submit_count += 1
                    if MAX_JOBS_TO_SUBMIT > 0 and submit_count >= MAX_JOBS_TO_SUBMIT:
                        job_block_list = [[]]
                        break

                    # Reset for next block
                    job_block_list = [[j]]
                    required_cores = j.processors
                    required_gpus = j.gpus
                    required_memory = j.memory
                    required_full_cores = 0
                    required_full_memory = 0
                    required_full_time = 0
                    required_block_time = j.time
                    
        # Run the final (possibly partial) job block list
        if run_partial_blocks and len(job_block_list[0]) > 0:
            required_full_cores = max(required_full_cores, required_cores)
            required_full_memory = max(required_full_memory, required_memory)
            required_full_time += required_block_time
            block_name = Job.run_job_blocks(job_block_list, required_full_cores, required_gpus, required_full_memory, required_full_time,
                partition_list=partition_list, gpu_partition_list=gpu_partition_list)
            submitted_job_blocks[block_name] = job_block_list
            submit_count += 1

        return submitted_job_blocks

    @classmethod
    def multicore_run_all(cls):
        Job.multicore_run_list(cls.all_jobs)

    @classmethod
    def multicore_keep_running(cls):
        all_jobs_complete = False
        cancelled_jobs = {}
        cancelled_requeue_iters = 5
        submitted_job_blocks = {}

        while not all_jobs_complete:

            # Find running job blocks
            sacct_output = subprocess.check_output(['sacct', '-n', '-o', 'JobID,Partition%15,JobName%100,State%20'])

            pending_jobs_per_partition = dict(zip(SBATCH_PARTITION_LIST, [0]*len(SBATCH_PARTITION_LIST)))
            pending_jobs_per_gpu_partition = dict(zip(SBATCH_GPU_PARTITION_LIST, [0]*len(SBATCH_GPU_PARTITION_LIST)))

            pending_running_complete_job_blocks = {}
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
                job_partition = job_split[1]
                job_name = job_split[2]
                job_status = ' '.join(job_split[3:])
                
                if job_name in submitted_job_blocks:
                    if job_name in cancelled_jobs:
                        # Ignore previous cancelled job
                        #print 'Ignoring previous cancelled job {0}.'.format(job_name)
                        del cancelled_jobs[job_name]
                        if job_name in pending_running_complete_job_blocks:
                            old_status = pending_running_complete_job_blocks[job_name][1]
                            #print old_status
                            if old_status.startswith('CANCELLED'):
                                del pending_running_complete_job_blocks[job_name]
                                #print('Removed from prc list.')
                    if job_status in ['PENDING', 'RUNNING', 'COMPLETED']:
                        if job_name in pending_running_complete_job_blocks:
                            print 'Found duplicate job: ' + job_name
                            dup_job_id, dup_job_status = pending_running_complete_job_blocks[job_name]
                            print job_id, job_status, dup_job_id, dup_job_status

                            job_to_kill = None
                            if job_status == 'PENDING':
                                job_to_kill = job_id
                            elif dup_job_status == 'PENDING':
                                job_to_kill = dup_job_id
                                pending_running_complete_job_blocks[job_name] = (job_id, job_status)    

                            if job_to_kill is not None:
                                print 'Canceling job ' + job_to_kill
                                try:
                                    scancel_output = subprocess.check_output(['scancel', '{0}'.format(job_to_kill)])
                                    print scancel_output
                                except:
                                    print "Error canceling job:", sys.exc_info()[0]
                        else:
                            pending_running_complete_job_blocks[job_name] = (job_id, job_status)
                            if job_status == 'PENDING':
                                pending += 1
                                if job_partition in pending_jobs_per_partition:
                                    pending_jobs_per_partition[job_partition] += 1
                                if job_partition in pending_jobs_per_gpu_partition:
                                    pending_jobs_per_gpu_partition[job_partition] += 1                               
                            elif job_status == 'RUNNING':
                                running += 1
                            elif job_status == 'COMPLETED':
                                complete += 1
                    elif job_status in ['FAILED', 'NODE_FAIL']:
                        failed += 1
                    elif job_status.startswith('CANCELLED'):
                        cancelled += 1

                        # This job might be requeued automatically after preemption
                        # Wait cancelled_requeue_iters before requeueing manually
                        cancelled_iters = 0
                        if job_name in cancelled_jobs:
                            cancelled_iters = cancelled_jobs[job_name]

                        if cancelled_iters < cancelled_requeue_iters:
                            pending_running_complete_job_blocks[job_name] = (job_id, job_status)
                            cancelled_jobs[job_name] = cancelled_iters + 1

                    elif job_status in ['TIMEOUT']:
                        timeout += 1
                    else:
                        print "Unexpected status: {0}".format(job_status)
                        other_status += 1
                elif job_name not in ['batch', 'true', 'prolog']:
                    non_matching += 1

            #print 'Found {0} running job blocks.'.format(len(pending_running_complete_job_blocks))

            # Find running jobs
            pending_running_complete_jobs = {}
            for job_block_name in pending_running_complete_job_blocks:
                job_id, job_status = pending_running_complete_job_blocks[job_block_name]
                job_block_list = submitted_job_blocks[job_block_name]
                for job_list in job_block_list:
                    for job in job_list:
                        pending_running_complete_jobs[job.name] = (job_id, job_status)

            #print '== {0} running jobs.'.format(len(pending_running_complete_jobs))

            # Make a list of runnable jobs
            run_count = 0
            block_count = 0
            runnable_jobs = []
            check_time = time.time()

            # refresh file cache
            refresh_file_cache()
            
            for j in cls.all_jobs:
                if j.name not in pending_running_complete_jobs and j.dependendencies_done(check_time, True) and not j.get_done(check_time, True):
                    runnable_jobs.append(j)
                    run_count += 1

            # Try to balance partitions
            min_pending_jobs = min(pending_jobs_per_partition.values())
            min_pending_gpu_jobs = min(pending_jobs_per_gpu_partition.values())

            # remove any partitions with too many pending jobs
            for partition in pending_jobs_per_partition.keys():
                if pending_jobs_per_partition[partition] > min_pending_jobs + (MAX_JOBS_TO_SUBMIT / len(pending_jobs_per_partition) / 2):
                    del(pending_jobs_per_partition[partition])

            partition_list = pending_jobs_per_partition.keys()
            if len(partition_list) == 0:
                print ('WARNING: empty partition_list')
                partition_list = SBATCH_PARTITION_LIST

            for partition in pending_jobs_per_gpu_partition.keys():
                if pending_jobs_per_gpu_partition[partition] > min_pending_jobs + (MAX_JOBS_TO_SUBMIT / len(pending_jobs_per_gpu_partition) / 2):
                    del(pending_jobs_per_gpu_partition[partition])

            gpu_partition_list = pending_jobs_per_gpu_partition.keys()
            if len(gpu_partition_list) == 0:
                print ('WARNING: empty gpu_partition_list')
                gpu_partition_list = SBATCH_GPU_PARTITION_LIST

            print partition_list
            print gpu_partition_list

            new_job_blocks = Job.multicore_run_list(runnable_jobs, run_partial_blocks=pending==0, partition_list=partition_list, gpu_partition_list=gpu_partition_list)
            block_count += len(new_job_blocks)
            submitted_job_blocks.update(new_job_blocks)

            print 'Found {0} pending, {1} running, {2} complete, {3} failed, {4} cancelled, {5} timeout, {6} unknown status and {7} non-matching job blocks.'.format(
                pending, running, complete, failed, cancelled, timeout, other_status, non_matching)

            print "Queued {0} job{1} in {2} block{3}.".format(
                run_count, '' if run_count == 1 else 's',
                block_count, '' if block_count == 1 else 's')

            if pending > 0 or running > 0 or run_count > 0:
                time.sleep(60)
            else:
                all_jobs_complete = True

    @classmethod
    def keep_running(cls):
        all_jobs_complete = False
        cancelled_jobs = {}
        cancelled_requeue_iters = 3
        first_iteration = True
        while not all_jobs_complete:

            all_job_names = {}
            # Generate dictionary of jobs
            for j in cls.all_jobs:
                all_job_names[j.name] = True

            # Find running jobs
            sacct_output = subprocess.check_output(['sacct', '-n', '-o', 'JobID,Partition%15,JobName%100,State%20'])

            pending_jobs_per_partition = dict(zip(SBATCH_PARTITION_LIST, [0]*len(SBATCH_PARTITION_LIST)))
            pending_jobs_per_gpu_partition = dict(zip(SBATCH_GPU_PARTITION_LIST, [0]*len(SBATCH_GPU_PARTITION_LIST)))

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
                job_partition = job_split[1]
                job_name = job_split[2]
                job_status = ' '.join(job_split[3:])
                
                if job_name in all_job_names:
                    if job_id in cancelled_jobs:
                        # Ignore previous cancelled job
                        del cancelled_jobs[job_id]
                        if job_name in pending_running_complete_jobs:
                            old_status = pending_running_complete_jobs[job_name][1]
                            if old_status.startswith('CANCELLED'):
                                del pending_running_complete_jobs[job_name]
                    if job_status in ['PENDING', 'RUNNING', 'COMPLETED']:
                        if job_name in pending_running_complete_jobs:
                            print 'Found duplicate job: ' + job_name
                            dup_job_id, dup_job_status = pending_running_complete_jobs[job_name]
                            print job_id, job_status, dup_job_id, dup_job_status

                            job_to_kill = None
                            if dup_job_status.startswith('CANCELLED'):
                                job_to_kill = None
                            elif job_status == 'PENDING':
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
                                if job_partition in pending_jobs_per_partition:
                                    pending_jobs_per_partition[job_partition] += 1
                                if job_partition in pending_jobs_per_gpu_partition:
                                    pending_jobs_per_gpu_partition[job_partition] += 1                               
                            elif job_status == 'RUNNING':
                                running += 1
                            elif job_status == 'COMPLETED':
                                complete += 1
                    elif job_status in ['FAILED', 'NODE_FAIL']:
                        failed += 1
                    elif job_status.startswith('CANCELLED'):
                        cancelled += 1
                        if not first_iteration:
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

            # Try to balance partitions
            min_pending_jobs = min(pending_jobs_per_partition.values())
            min_pending_gpu_jobs = min(pending_jobs_per_gpu_partition.values())

            # remove any partitions with too many pending jobs
            for partition in pending_jobs_per_partition.keys():
                if pending_jobs_per_partition[partition] > min_pending_jobs + (MAX_JOBS_TO_SUBMIT / len(pending_jobs_per_partition) / 2):
                    del(pending_jobs_per_partition[partition])

            partition_list = pending_jobs_per_partition.keys()
            if len(partition_list) == 0:
                print ('WARNING: empty partition_list')
                partition_list = SBATCH_PARTITION_LIST

            for partition in pending_jobs_per_gpu_partition.keys():
                if pending_jobs_per_gpu_partition[partition] > min_pending_jobs + (MAX_JOBS_TO_SUBMIT / len(pending_jobs_per_gpu_partition) / 2):
                    del(pending_jobs_per_gpu_partition[partition])

            gpu_partition_list = pending_jobs_per_gpu_partition.keys()

            print partition_list
            print gpu_partition_list

            run_count = 0
            for j in cls.all_jobs:
                if j.name not in pending_running_complete_jobs and j.dependendencies_done():
                    run_count += j.run(partition_list=partition_list, gpu_partition_list=gpu_partition_list)

            print 'Found {0} pending, {1} running, {2} complete, {3} failed, {4} cancelled, {5} timeout, {6} unknown status and {7} non-matching jobs.'.format(
                pending, running, complete, failed, cancelled, timeout, other_status, non_matching)

            print "Queued {0} job{1}.".format(run_count, '' if run_count == 1 else 's')

            if pending > 0 or running > 0 or run_count > 0:
                time.sleep(60)
            else:
                all_jobs_complete = True

            first_iteration = False

class JobSplit(object):
    '''make a multi-output job object look like a single output job'''
    def __init__(self, job, idx):
        self.job = job
        self.idx = idx
        self.name = job.name

    def get_done(self, check_time=time.time(), cache_only=False):
        return self.job.get_done(check_time, cache_only)

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
                                   'probs_%05d_%s.hdf5' % (idx, '_'.join(self.coords)))

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
                                   'segs_%05d_%s.hdf5' % (idx, '_'.join(self.coords)))

    def command(self):
        return ['python', 'segment_image.py', self.raw_image, self.probability_map.output, self.output] + \
            self.coords + self.core_coords

class Classify_Image(Job):
    def __init__(self, idx, raw_image, classifier_file, classify_prog):
        Job.__init__(self)
        self.already_done = True
        self.raw_image = raw_image
        self.classify_prog = classify_prog
        self.classifier_file = classifier_file
        self.dependencies = []
        self.gpus = 1
        self.memory = 4000
        self.time = 180
        self.output = os.path.join('segmentations', 'probs_%05d.hdf5' % (idx))
        #self.already_done = os.path.exists(self.output)

    def command(self):
        return ['python', self.classify_prog, self.classifier_file, self.raw_image, self.output]

class Segment_Image(Job):
    def __init__(self, idx, classifier, segmentation_prog):
        Job.__init__(self)
        self.already_done = True
        self.classifier = classifier
        self.segmentation_prog = segmentation_prog
        self.dependencies = [classifier]
        self.memory = 2000
        self.time = 120
        self.output = os.path.join('segmentations', 'segs_%05d.hdf5' % (idx))
        #self.already_done = os.path.exists(self.output)

    def command(self):
        return ['python', self.segmentation_prog, self.classifier.output, self.classifier.raw_image, self.output]

class Block(Job):
    def __init__(self, segmented_slices, indices, *args):
        Job.__init__(self)
        self.already_done = True
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
    def __init__(self, block, indices, global_block_number, fusion_cplex_threads=4):
        Job.__init__(self)
        self.already_done = False
        self.block = block
        self.global_block_number = global_block_number
        self.dependencies = [block]
        #self.processors = 4
        self.processors = fusion_cplex_threads
        #self.memory = 20000
        self.memory = 16000
        # memory is per proc, so we are requesting 64GB here (and sometimes use it)
        #self.time = 360
        self.time = 180
        self.indices = indices
        self.output = os.path.join('bigfusedblocks', 'fusedblock_%d_%d_%d.hdf5' % indices)
        #self.already_done = os.path.exists(self.output)

    def command(self):
        return ['python',
                os.path.join(os.environ['CONNECTOME'], 'WindowFusion', 'window_fusion_cpx.py'),
                self.block.output,
                str(self.global_block_number),
                self.output]

class ComboBlock(Job):
    def __init__(self, block, indices, global_block_number, v1_block_folder, v1_remapped_block_folder, v2_block_folder, v1_segment_sizes):
        Job.__init__(self)
        self.already_done = False
        self.block = block
        self.global_block_number = global_block_number
        self.v1_block_folder = v1_block_folder
        self.v1_remapped_block_folder = v1_remapped_block_folder
        self.v2_block_folder = v2_block_folder
        self.v1_segment_sizes = v1_segment_sizes
        self.dependencies = [block]
        #self.memory = 2000
        self.memory = 1000
        self.time = 5
        self.indices = indices
        self.output = os.path.join('combofusedblocks', 'comboblock_%d_%d_%d.hdf5' % indices)
        #self.already_done = os.path.exists(self.output)

    def command(self):

        # combo segment objects over a given size
        return ['python',
                os.path.join(os.environ['CONNECTOME'], 'Cleanup', 'resegment_block.py'),
                os.path.join(self.v1_block_folder, 'fusedblock_%d_%d_%d.hdf5' % self.indices),
                os.path.join(self.v2_block_folder, 'fusedblock_%d_%d_%d.hdf5' % self.indices),
                os.path.join(self.v1_remapped_block_folder, 'block_%d_%d_%d.hdf5' % self.indices),
                self.v1_segment_sizes,
                str(self.global_block_number),
                self.output]

        # combo segment all objects
        # return ['python',
        #         os.path.join(os.environ['CONNECTOME'], 'Cleanup', 'combo_segment_block.py'),
        #         os.path.join(self.v1_block_folder, 'fusedblock_%d_%d_%d.hdf5' % self.indices),
        #         os.path.join(self.v2_block_folder, 'fusedblock_%d_%d_%d.hdf5' % self.indices),
        #         str(self.global_block_number),
        #         self.output]

class CleanBlock(Job):
    def __init__(self, fusedblock):
        Job.__init__(self)
        self.already_done = False
        self.indices = fusedblock.indices
        self.block = fusedblock.block
        self.global_block_number = fusedblock.global_block_number
        self.v1_block_folder = fusedblock.v1_block_folder
        self.dependencies = [fusedblock]
        self.memory = 6000
        #self.memory = 8000
        #self.time = 60
        self.time = 30
        self.inputlabels = fusedblock.output
        self.inputprobs = os.path.join(self.v1_block_folder.replace('bigfusedblocks', 'bigdicedblocks'), 'block_%d_%d_%d.hdf5' % self.indices)
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
        #self.memory = 8000
        #self.memory = 4000
        #self.memory = 2000
        self.memory = 500
        #self.time = 60
        #self.time = 30
        self.time = 3
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
        self.memory = 4000
        self.time = 240
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
        self.memory = 64000
        self.time = 960
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
        #self.memory = 4000
        #self.memory = 8000
        self.memory = 2000
        #self.time = 60
        #self.time = 30
        self.time = 2
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
        self.memory = 2000
        self.time = 60
        self.zoffset = zoffset
        self.xy_halo = xy_halo
        self.image_size = image_size
        self.xy_block_size = xy_block_size
        # Specify .tif for 32-bit tif output or .png for adaptive 24-bit (RGB) / 32-bit (RGBA) png output
        #self.output = os.path.join('output_labels', 'labels_%05d.tif' % zplane)
        self.output = os.path.join('output_labels', 'labels_%05d.png' % zplane)
        #self.already_done = os.path.exists(self.output)

    def generate_args(self):
        for block in self.dependencies:
            # XY corner followed by filename
            yield str(block.indices[0] * self.xy_block_size)
            yield str(block.indices[1] * self.xy_block_size)
            yield block.output

    def command(self):
        return [os.path.join(os.environ['CONNECTOME'], 'Control', 'extract_label_plane.sh'), self.output, str(self.image_size[0]), str(self.image_size[1]), str(self.zoffset), str(self.xy_halo)] + \
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
        return [os.path.join(os.environ['CONNECTOME'], 'Control', 'extract_overlay_plane.sh'), self.output, self.input_image_path, str(self.image_size[0]), str(self.image_size[1]), str(self.zoffset), str(self.xy_halo)] + \
            list(self.generate_args())

class CalculateIndexPlane(Job):
    def __init__(self, zplane, label_plane):
        Job.__init__(self)
        self.already_done = False
        self.dependencies = [label_plane]
        self.label_image = label_plane.output
        self.memory = 20000
        self.time = 60
        self.zplane = zplane
        self.output = os.path.join('output_index', 'index_%05d.h5' % zplane)
        #self.already_done = os.path.exists(self.output)

    def command(self):
        return [os.path.join(os.environ['CONNECTOME'], 'Control', 'calculate_index_plane.sh'), self.label_image, str(self.zplane), self.output]

class MergeDownsampleIndex(Job):
    def __init__(self, outfilename, inputs):
        Job.__init__(self)
        self.already_done = False
        self.dependencies = inputs
        self.memory = 4000
        self.time = 240
        self.output = os.path.join('output_index', outfilename)
        #self.already_done = os.path.exists(self.output)
        
    def command(self):
        return [os.path.join(os.environ['CONNECTOME'], 'Control', 'merge_index_ds884.sh')] + \
            [s.output for s in self.dependencies] + \
            [self.output]

class MergeIndex(Job):
    def __init__(self, outfilename, inputs):
        Job.__init__(self)
        self.already_done = False
        self.dependencies = inputs
        self.memory = 4000
        self.time = 240
        self.output = os.path.join('output_index', outfilename)
        #self.already_done = os.path.exists(self.output)
        
    def command(self):
        return [os.path.join(os.environ['CONNECTOME'], 'Control', 'merge_index.sh')] + \
            [s.output for s in self.dependencies] + \
            [self.output]


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
    image_size = (2048, 2048) # image (height, width)
    probability_subimage_size = 1024
    probability_subimage_halo = 32
    segmentation_subimage_size = 1024
    segmentation_subimage_halo = 128
    block_xy_halo = 64
    block_xy_size = 512 - (2 * 64)
    block_z_size = 52
    block_z_halo = 6
    fusion_cplex_threads = 4

    classify_prog = os.path.join(os.environ['CONNECTOME'], 'DeepNets', 'classify_image_slow_h5.py')
    classifier_file = os.path.join(os.environ['CONNECTOME'], 'DeepNets', 'lgn49m10aamega_ds4_200k_seed11_continued.pkl')
    segmentation_prog = os.path.join(os.environ['CONNECTOME'], 'Segment', 'segment_ws.py')

    settings_file = sys.argv[1]
    os.environ['CONNECTOME_SETTINGS'] = settings_file
    execfile(settings_file)

    if '-n' in sys.argv:
        Job.run_name = sys.argv[sys.argv.index('-n') + 1]

    images = [f.rstrip() for f in open(sys.argv[2])]

    classifications = [Classify_Image(idx, im, classifier_file, classify_prog) for idx, im in enumerate(images)]

    segmentations = [Segment_Image(idx, imc, segmentation_prog) for idx, imc in enumerate(classifications)]

    #segmentations = [f.rstrip() for f in open(sys.argv[2])]
    #print segmentations
    
    # Dice full volume
    blocks = {}
    nblocks_x = (image_size[0] - 2 * block_xy_halo) / block_xy_size
    nblocks_y = (image_size[1] - 2 * block_xy_halo) / block_xy_size
    nblocks_z = (len(segmentations) - 2 * block_z_halo) / block_z_size
    print "Making {0}x{1}x{2} combo block volume.".format(nblocks_x, nblocks_y, nblocks_z)
    block_order = []
    hi_slice = 0
    for block_idx_z in range(nblocks_z):
        lo_slice = block_idx_z * block_z_size
        hi_slice = lo_slice + block_z_size + 2 * block_z_halo
        for block_idx_y in range(nblocks_y):
            ylo = block_idx_y * block_xy_size
            yhi = ylo + block_xy_size + 2 * block_xy_halo
            for block_idx_x in range(nblocks_x):
                xlo = block_idx_x * block_xy_size
                xhi = xlo + block_xy_size + 2 * block_xy_halo
                # print "Making block {0}, slice {1}, crop {2}.".format(
                #     (block_idx_x, block_idx_y, block_idx_z),
                #     (lo_slice, hi_slice),
                #     (xlo, ylo, xhi, yhi))
                blocks[block_idx_x, block_idx_y, block_idx_z] = \
                    Block(segmentations[lo_slice:hi_slice],
                          (block_idx_x, block_idx_y, block_idx_z),
                          xlo, ylo, xhi, yhi)
                block_order.append((block_idx_x, block_idx_y, block_idx_z))

    # Combo fuse all blocks
    # Takes results and sizes from gala / fusion and resegments large objects.
    fused_blocks = dict((idxs, ComboBlock(blocks[idxs], idxs,
        idxs[0] + idxs[1] * nblocks_x + idxs[2] * nblocks_x * nblocks_y,
        v1_block_folder, v1_remapped_block_folder, v2_block_folder, v1_segment_sizes)) for idxs in block_order)

    # Cleanup all blocks (remove small or completely enclosed segments)
    cleaned_blocks = dict((idxs, CleanBlock(fused_blocks[idxs])) for idxs in block_order)
    #cleaned_blocks = fused_blocks

    # Pairwise match all blocks.
    #
    # We overwrite each block in cleaned_blocks (the python dict, not the file)
    # with the output of the pairwise matching, and work in non-overlapping
    # sets (even-to-odd, then odd-to-even)
    for direction in range(3):  # X, Y, Z
        for wpidx, which_pairs in enumerate(['even', 'odd']):
            for idx in block_order:
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
    for idxs in block_order:
        plane_joins_lists[idxs[2]] = plane_joins_lists.get(idxs[2], []) + [cleaned_blocks[idxs]]
    plane_join_jobs = [JoinConcatenation('concatenate_Z_%d.hdf5' % idx, plane_joins_lists[idx])
                       for idx in plane_joins_lists]
    full_join = JoinConcatenation('concatenate_full.hdf5', plane_join_jobs)

    # build the global remap
    remap = GlobalRemap('globalmap.hdf5', full_join)

    # and apply it to every block
    remapped_blocks = [RemapBlock(cleaned_blocks[idx], remap, idx) for idx in block_order]
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
                    for idx, _ in enumerate(segmentations[:hi_slice])]

    # optional, render overlay images
    output_overlays = [ExtractOverlayPlane(idx, block_xy_halo,
                                       remapped_blocks_by_plane[min(idx / block_z_size, max_zslab)],
                                       idx - block_z_size * min(idx / block_z_size, max_zslab),  # offset within block
                                       image_size, block_xy_size, im)
                    for idx, im in enumerate(images[:hi_slice])]

    # optional, calculate pixel index and merge into single index file
    if os.path.exists(index_stats_file):
        output_index = [CalculateIndexPlane(idx, im)
                        for idx, im in enumerate(output_labels)]

        all_merges = []
        current_merge = []
        merge_phase = 0
        merge_n = 12
        for merge_from in range(0, len(output_index), merge_n):
            merge_to = np.minimum(merge_from + merge_n, len(output_index))
            current_merge.append(MergeDownsampleIndex('merge{0}_{1}-{2}.h5'.format(merge_phase, merge_from, merge_to-1), output_index[merge_from:merge_to]))
        all_merges.append(current_merge)

        while(len(current_merge) > 1):
            merge_phase += 1
            current_merge = []
            for merge_from in range(0, len(all_merges[-1]), merge_n):
                merge_to = np.minimum(merge_from + merge_n, len(all_merges[-1]))
                current_merge.append(MergeIndex('merge{0}_{1}-{2}.h5'.format(merge_phase, merge_from, merge_to-1), all_merges[-1][merge_from:merge_to]))
            all_merges.append(current_merge)

    # # Render fused / cleaned blocks directly (for debugging)
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
        else:
            # Bundle jobs for multicore nodes
            Job.multicore_run_all()
    elif '-mk' in sys.argv or '--multicore-keeprunning' in sys.argv:
        if RUN_LOCAL:
            print "ERROR: --local cannot be used with --multicore-keeprunning (not yet implemented)."
        else:
            # Bundle jobs for multicore nodes
            Job.multicore_keep_running()
    else:
        for j in Job.all_jobs:
            if j.output == sys.argv[3] or sys.argv[3] in j.output or sys.argv[3] in j.output[0] or sys.argv[3] in j.name:
                for k in j.dependencies:
                    if k.output != sys.argv[3] and sys.argv[3] not in k.output and sys.argv[3] not in k.output[0] and sys.argv[3] not in k.name:
                        k.already_done = True
                j.run()

    
