from pylearn2.utils import py_integer_types
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
import os
import time
import sys
import numpy as np
from PIL import Image
import h5py
import clahe

job_repeat_attempts = 5

def check_file(filename):
    if not os.path.exists(filename):
        return False
    # verify the file has the expected data
    import h5py
    f = h5py.File(filename, 'r')
    fkeys = f.keys()
    f.close()
    if set(fkeys) != set(['probabilities']):
        print 'Key error in {0}.'.format(filename)
        print fkeys
        os.unlink(filename)
        return False
    return True

def normalize_image_float(original_image, saturation_level=0.005, clahe_level=0):
    if clahe_level != 0:
        norm_image = np.zeros(original_image.shape, dtype=original_image.dtype)
        clahe.clahe(original_image, norm_image, clahe_level)
        return np.float32(norm_image) / 255.0
    else:
        sorted_image = np.sort( np.uint8(original_image).ravel() )
        minval = np.float32( sorted_image[ len(sorted_image) * ( saturation_level / 2 ) ] )
        maxval = np.float32( sorted_image[ len(sorted_image) * ( 1 - saturation_level / 2 ) ] )
        norm_image = np.float32(original_image - minval) * ( 255 / (maxval - minval))
        norm_image[norm_image < 0] = 0
        norm_image[norm_image > 255] = 255
        return norm_image / 255.0

_, img_in, h5_out, batch_size_str = sys.argv[0:4]

repeat_attempt_i = 0
while repeat_attempt_i < job_repeat_attempts and not check_file(h5_out):

    repeat_attempt_i += 1
    try:

        print "Classifying image {0} with:".format(img_in)

        batch_size = int(batch_size_str)

        print "batch size {0}.".format(batch_size)

        raw_image = np.array(Image.open(img_in))
        nx, ny = raw_image.shape

        #import theano.tensor as T

        model_args = sys.argv[4:]
        models = []
        clahe_levels = []

        output_spaces = []

        # Prepare models and output spaces
        while model_args:

            model_path = model_args.pop(0)
            clahe_level = int(model_args.pop(0))
            print model_path

            models.append(serial.load(model_path))
            clahe_levels.append(clahe_level)

            models[-1].set_batch_size(batch_size)
            output_spaces.append(models[-1].get_output_space().make_batch_theano())

        print "{0} models.".format(len(models))
        pad_images = []
        input_spaces = []

        # Padding must be the same for all models
        input_shape = models[0].input_space.shape
        pad_by = np.max(input_shape) / 2

        ymf = None

        # Prepare clahe images and input spaces
        unique_clahe_levels, first_indices = np.unique(clahe_levels, return_index=True)
        print "{0} input clahe levels.".format(len(unique_clahe_levels))
        print unique_clahe_levels

        for i, clahe_level in enumerate(unique_clahe_levels):
            input_image = normalize_image_float(raw_image, clahe_level=clahe_level)
            pad_images.append(np.pad(input_image, ((pad_by, pad_by), (pad_by, pad_by)), 'symmetric'))
            input_spaces.append(models[first_indices[i]].get_input_space().make_batch_theano())

            for neti in np.nonzero(np.array(clahe_levels)==clahe_level)[0]:
                # Add to the output ymf
                if ymf is None:
                    ymf = models[neti].fprop(input_spaces[-1])
                else:
                    ymf = ymf + models[neti].fprop(input_spaces[-1])

        raw_image = None

        if len(models) > 1:
            ymf = ymf / float(len(models))
        ymf.name = 'ymf'

        from theano import function
        batch_predict = function(input_spaces,[ymf])

        X = []
        for i in range(len(input_spaces)):
            X.append(np.zeros((1, input_shape[0], input_shape[1], batch_size), dtype=np.float32))
        y = np.zeros(nx * ny, dtype=np.float32)

        batch_count = 0
        batch_start = 0
        batchi = 0

        assert isinstance(X[0].shape[0], (int, long))
        assert isinstance(batch_size, py_integer_types)

        start_time = time.time()

        for xi in range(nx):
            for yi in range(ny):

                for i in range(len(input_spaces)):
                    X[i][0, :, :, batchi] = pad_images[i][xi : xi + input_shape[0], yi : yi + input_shape[1]]

                batchi += 1

                if batchi == batch_size:
                    # Classify and reset
                    y[batch_start:batch_start + batch_size] = batch_predict(*X)[0][:,0]
                    batch_count += 1
                    batch_start += batch_size
                    batchi = 0

                    if batch_count % 100 == 0:
                        print "Batch {0} done. Up to {1}.".format(batch_count, (xi, yi))

        if batchi > 0:
            y[batch_start:batch_start + batchi] = batch_predict(*X)[0][:batchi,0]

        print 'Complete in {0:1.4f} seconds'.format(time.time() - start_time)

        output_image = y.reshape(input_image.shape)

        temp_path = h5_out + '_partial'

        if os.path.exists(temp_path):
            os.unlink(temp_path)

        f = h5py.File(temp_path)
        f['/probabilities'] = output_image
        f.close()

        print 'Wrote to {0}.'.format(temp_path)
        print os.path.exists(temp_path)

        # move to final location
        if os.path.exists(h5_out):
            os.unlink(h5_out)
        os.rename(temp_path, h5_out)

        print 'Renamed to {0}.'.format(h5_out)
        print os.path.exists(h5_out)

        print "Success"

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print e
        raise

assert check_file(h5_out), "Output file could not be verified after {0} attempts, exiting.".format(job_repeat_attempts)
