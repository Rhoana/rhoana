import time
import os
import sys
import numpy as np
import Image
import h5py

from lib_maxout_theano_batch import *

job_repeat_attempts = 5
saturation_level=0.005

def check_file(filename):
    if not os.path.exists(filename):
        return False
    # verify the file has the expected data
    f = h5py.File(filename, 'r')
    fkeys = f.keys()
    f.close()
    if set(fkeys) != set(['probabilities']):
        os.unlink(filename)
        return False
    return True

if __name__ == '__main__':

    _, model_path, img_in, output_path = sys.argv[0:4]

    repeat_attempt_i = 0
    while repeat_attempt_i < job_repeat_attempts and not check_file(output_path):

        repeat_attempt_i += 1

        try:

            top = int(sys.argv[4])
            left = int(sys.argv[5])
            height = int(sys.argv[6])
            width = int(sys.argv[7])

            # For lib_maxout_theano_batch we can control batch size
            batch_size = 2048
            if len(sys.argv) > 8:
                batch_size = int(sys.argv[8])
            network = DeepNetwork(model_path, batch_size=batch_size)
            #network = DeepNetwork(model_path)

            if (os.path.exists(output_path)):

                print "Output file {0} already exists.".format(output_path)

            else:

                print 'Processing image {0}, {1}.'.format(img_in, (top, left, height, width))
                out_hdf5 = h5py.File(output_path, 'w')

                input_image = np.array(Image.open(img_in))
                nx, ny = input_image.shape

                # Find normalization settings for full image
                sorted_image = np.sort( input_image.ravel() )
                minval = np.float32( sorted_image[ len(sorted_image) * ( saturation_level / 2 ) ] )
                maxval = np.float32( sorted_image[ len(sorted_image) * ( 1 - saturation_level / 2 ) ] )

                # Pad
                pad_by = network.pad_by
                input_image = np.pad(input_image, ((pad_by, pad_by), (pad_by, pad_by)), 'symmetric')

                # Crop
                input_image = input_image[top:top + height + 2 * pad_by, left:left + width + 2 * pad_by]

                # Normalize cropped image as float
                input_image = np.float32(input_image - minval) * ( 255 / (maxval - minval))
                input_image[input_image < 0] = 0
                input_image[input_image > 255] = 255
                input_image = input_image / 255.0

                start_time = time.time()

                output = network.apply_net(input_image, perform_pad=False)

                print 'Deep net classify complete in {0:1.4f} seconds'.format(time.time() - start_time)

                # im = Image.fromarray(np.uint8(output * 255))
                # im.save(output_path.replace('hdf5', 'tif'))
                # print "Image saved."

                out_hdf5.create_dataset('probabilities', data = output, chunks = (64,64), compression = 'gzip')
                out_hdf5.close()
                print "Probabilities saved to: {0}".format(output_path)

        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
        except KeyboardInterrupt:
            raise
        except:
            print "Unexpected error:", sys.exc_info()[0]
            if repeat_attempt_i == job_repeat_attempts:
                raise
            
    assert check_file(output_path), "Output file could not be verified after {0} attempts, exiting.".format(job_repeat_attempts)
