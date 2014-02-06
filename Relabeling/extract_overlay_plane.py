import os
import sys
import numpy as np
import h5py
from libtiff import TIFF
import mahotas

job_repeat_attempts = 5

ncolors = 10000
alpha = 0.5

def check_file(filename):
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        return False
    return True

def sbdm_string_hash( in_string ):
    hash = 0
    for i in xrange(len(in_string)):
        hash = ord(in_string[i]) + (hash << 6) + (hash << 16) - hash
    return np.uint32(hash % 2**32)

if __name__ == '__main__':

    output_path = sys.argv[1]

    repeat_attempt_i = 0
    while repeat_attempt_i < job_repeat_attempts and not check_file(output_path):

        repeat_attempt_i += 1

        try:

            # Parse arguments
            args = sys.argv[2:]
            input_image_path = args.pop(0)
            output_size = int(args.pop(0))
            zoffset = int(args.pop(0))
            xy_halo = int(args.pop(0))

            output_ids = np.zeros((output_size, output_size), np.uint32)

            while args:
                xbase = int(args.pop(0))
                ybase = int(args.pop(0))
                infile = args.pop(0)

                try:
                    data = h5py.File(infile, 'r')['labels'][:, :, :]
                except Exception, e:
                    print e, infile
                    raise

                xend = xbase + data.shape[0]
                yend = ybase + data.shape[1]

                xfrom_base = 0
                xfrom_end = data.shape[0]
                yfrom_base = 0
                yfrom_end = data.shape[1]

                if xbase > 0:
                    xbase = xbase + xy_halo
                    xfrom_base = xfrom_base + xy_halo
                if xend < output_size - 1:
                    xend = xend - xy_halo
                    xfrom_end = xfrom_end - xy_halo

                if ybase > 0:
                    ybase = ybase + xy_halo
                    yfrom_base = yfrom_base + xy_halo
                if yend < output_size - 1:
                    yend = yend - xy_halo
                    yfrom_end = yfrom_end - xy_halo

                print "{0} region ({1}:{2},{3}:{4}) assinged to ({5}:{6},{7}:{8}).".format(
                    infile, xfrom_base, xfrom_end, yfrom_base, yfrom_end, xbase, xend, ybase, yend)

                output_ids[xbase:xend, ybase:yend] = data[xfrom_base:xfrom_end, yfrom_base:yfrom_end, zoffset]

            # Generate a random colormap (should be the same for all nodes)
            np.random.seed(7)
            color_map = np.uint8(np.random.randint(0,256,(ncolors+1)*3)).reshape((ncolors + 1, 3))
            
            overlay_colors = color_map[output_ids % ncolors]

            boundaries = output_ids==1

            for ci in range(3):
                overlay_colors[:,:,ci][boundaries] = 128

            current_image_f = np.float32(mahotas.imread(input_image_path)[:output_size, :output_size])

            overlay_colors[:,:,0] = (1-alpha) * overlay_colors[:,:,0] + alpha * current_image_f
            overlay_colors[:,:,1] = (1-alpha) * overlay_colors[:,:,1] + alpha * current_image_f
            overlay_colors[:,:,2] = (1-alpha) * overlay_colors[:,:,2] + alpha * current_image_f

            mahotas.imsave(output_path, overlay_colors)

            # tif = TIFF.open(output_path, mode='w')
            # tif.write_image(overlay_colors, compression='lzw')
            
        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
        except KeyboardInterrupt:
            pass
        # except:
        #     print "Unexpected error:", sys.exc_info()[0]
        #     if repeat_attempt_i == job_repeat_attempts:
        #         raise
            
    assert check_file(output_path), "Output file could not be verified after {0} attempts, exiting.".format(job_repeat_attempts)
