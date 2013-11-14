import os
import sys
import subprocess

def check_file(filename):
    # verify the file has the expected data
    import h5py
    f = h5py.File(filename, 'r')
    if set(f.keys()) != set(['segmentations', 'probabilities']):
        os.unlink(filename)
        return False
    return True

try:

    # Default settings
    features_prog = os.path.join(os.environ['CONNECTOME'], 'ClassifyMembranes', 'compute_features')
    classify_prog = os.path.join(os.environ['CONNECTOME'], 'ClassifyMembranes', 'classify_image')
    segmentation_prog = os.path.join(os.environ['CONNECTOME'], 'Segment', 'segment.py')

    settings_file = os.environ['CONNECTOME_SETTINGS']
    execfile(settings_file)

    args = sys.argv[1:]

    image_file = args.pop(0)
    classifier = args.pop(0)
    features_file = args.pop(0)
    probabilities_file = args.pop(0)
    segmentations_file = args.pop(0)

    if os.path.exists(segmentations_file):
        print segmentations_file, "already exists"
        if check_file(segmentations_file):
            sys.exit(0)

    if not os.path.exists(features_file):
        temp_features_file = features_file + "_tmp"
        print "Computing features:", features_prog, image_file, temp_features_file
        subprocess.check_call([features_prog, image_file, temp_features_file], env=os.environ)
        os.rename(temp_features_file, features_file)

    if not os.path.exists(probabilities_file):
        print "Computing probabilities:", classify_prog, features_file, classifier, probabilities_file
        subprocess.check_call(['python', classify_prog, features_file, classifier, probabilities_file], env=os.environ)

    print "Computing segmentations:", segmentation_prog, probabilities_file, segmentations_file
    subprocess.check_call(['python', segmentation_prog, probabilities_file, segmentations_file], env=os.environ)

    assert check_file(segmentations_file), "Bad data in file, exiting!"

except KeyboardInterrupt:
    pass
