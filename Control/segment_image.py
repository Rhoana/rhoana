import os
import sys
import subprocess

def check_file(filename):
    # verify the file has the expected data
    import h5py
    f = h5py.File(filename, 'r')
    if set(f.keys()) != set(['segmentations']):
        os.unlink(filename)
        return False
    return True

try:
    args = sys.argv[1:]

    image_file = args.pop(0)
    classifier = args.pop(0)
    probabilities_file = args.pop(0)
    segmentations_file = args.pop(0)

    if os.path.exists(segmentations_file):
        print segmentations_file, "already exists"
        if check_file(segmentations_file):
            sys.exit(0)

    classify_prog = os.path.join(os.environ['CONNECTOME'], 'ClassifyMembranes', 'classify_image')
    print "Computing probabilities:", classify_prog, image_file, classifier, probabilities_file
    subprocess.check_call([classify_prog, image_file, classifier, probabilities_file], env=os.environ)

    segmentation_prog = os.path.join(os.environ['CONNECTOME'], 'Segment', 'segment.py')
    print "Computing segmentations:", segmentation_prog, probabilities_file, segmentations_file
    subprocess.check_call(['python', '-i', segmentation_prog, probabilities_file, segmentations_file], env=os.environ)
    assert check_file(segmentations_file), "Bad data in file, exiting!"
except KeyboardInterrupt:
    pass
