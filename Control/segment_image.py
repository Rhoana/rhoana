import os
import sys
import subprocess

job_repeat_attempts = 5

def check_file(filename):
    if not os.path.exists(filename):
        return False
    # verify the file has the expected data
    import h5py
    f = h5py.File(filename, 'r')
    fkeys = f.keys()
    f.close()
    if set(fkeys) != set(['segmentations', 'probabilities']):
        os.unlink(filename)
        return False
    return True

# Default settings
features_prog = os.path.join(os.environ['CONNECTOME'], 'ClassifyMembranes', 'compute_features')
classify_prog = os.path.join(os.environ['CONNECTOME'], 'DeepNets', 'full_image_combo_classify_with_stumps.py')
segmentation_prog = os.path.join(os.environ['CONNECTOME'], 'Segment', 'segment_ws.py')

if 'CONNECTOME_SETTINGS' in os.environ:
    settings_file = os.environ['CONNECTOME_SETTINGS']
    execfile(settings_file)

args = sys.argv[1:]

image_file = args.pop(0)
classifier = args.pop(0)
stump_file = args.pop(0)
#features_file = args.pop(0)
probabilities_file = args.pop(0)
segmentations_file = args.pop(0)

if os.path.exists(segmentations_file):
    print segmentations_file, "already exists"
    if check_file(segmentations_file):
        sys.exit(0)

repeat_attempt_i = 0
while repeat_attempt_i < job_repeat_attempts and not check_file(segmentations_file):

    repeat_attempt_i += 1

    try:

        # if not os.path.exists(features_file):
        #     temp_features_file = features_file + "_tmp"
        #     print "Computing features:", features_prog, image_file, temp_features_file
        #     subprocess.check_call([features_prog, image_file, temp_features_file], env=os.environ)
        #     os.rename(temp_features_file, features_file)

        if not os.path.exists(probabilities_file):
            print "Computing probabilities:", classify_prog, image_file, stump_file, classifier, probabilities_file
            subprocess.check_call(['python', classify_prog, image_file, stump_file, classifier, probabilities_file], env=os.environ)

        print "Computing segmentations:", segmentation_prog, probabilities_file, segmentations_file
        subprocess.check_call(['python', segmentation_prog, probabilities_file, segmentations_file], env=os.environ)

    except IOError as e:
        print "I/O error({0}): {1}".format(e.errno, e.strerror)
    except KeyboardInterrupt:
        pass
    except:
        print "Unexpected error:", sys.exc_info()[0]
        if repeat_attempt_i == job_repeat_attempts:
            pass

assert check_file(segmentations_file), "Output file could not be verified after {0} attempts, exiting.".format(job_repeat_attempts)

