import os
import sys
import numpy as np
import h5py
# from libtiff import TIFF
import mahotas
import shutil

job_repeat_attempts = 5

def check_file(filename):
    if not os.path.exists(filename) or os.path.getsize(filename) <= 8:
        return False
    return True

if __name__ == '__main__':

    output_path = sys.argv[1]

    repeat_attempt_i = 0
    while repeat_attempt_i < job_repeat_attempts and not check_file(output_path):

        repeat_attempt_i += 1

        try:

            # Parse arguments
            args = sys.argv[2:]
            output_size_x = int(args.pop(0))
            output_size_y = int(args.pop(0))
            zoffset = int(args.pop(0))
            xy_halo = int(args.pop(0))

            output_image = np.zeros((output_size_x, output_size_y), np.uint32)
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
                if xend < output_size_x - 1:
                    xend = xend - xy_halo
                    xfrom_end = xfrom_end - xy_halo

                if ybase > 0:
                    ybase = ybase + xy_halo
                    yfrom_base = yfrom_base + xy_halo
                if yend < output_size_y - 1:
                    yend = yend - xy_halo
                    yfrom_end = yfrom_end - xy_halo

                print "{0} region ({1}:{2},{3}:{4}) assinged to ({5}:{6},{7}:{8}).".format(
                    infile, xfrom_base, xfrom_end, yfrom_base, yfrom_end, xbase, xend, ybase, yend)

                output_image[xbase:xend, ybase:yend] = data[xfrom_base:xfrom_end, yfrom_base:yfrom_end, zoffset]

            temp_path = output_path.replace('.', '_partial.')

            if output_path.lower().endswith('.tif') or output_path.lower().endswith('.tiff'):
                # Output 32-bit tiff (difficult to open for most desktop applications)
                tif = TIFF.open(temp_path, mode='w')
                tif.write_image(np.rot90(output_image), compression='lzw')
                tif.close()
            else:
                # Output in Vast PNG or other format
                if np.max(output_image) < 2**24:
                    vast_export = np.zeros((output_image.shape[0], output_image.shape[1], 3), dtype=np.uint8)
                    vast_export[:,:,0] = np.uint8(output_image // (2**16) % (2**8))
                    vast_export[:,:,1] = np.uint8(output_image // (2**8) % (2**8))
                    vast_export[:,:,2] = np.uint8(output_image % (2**8))
                    mahotas.imsave(temp_path, vast_export)
                else:
                    # Use alpha channel to export 32-bit labels
                    vast_export = np.zeros((output_image.shape[0], output_image.shape[1], 4), dtype=np.uint8)
                    vast_export[:,:,0] = np.uint8(output_image // (2**16) % (2**8))
                    vast_export[:,:,1] = np.uint8(output_image // (2**8) % (2**8))
                    vast_export[:,:,2] = np.uint8(output_image % (2**8))
                    vast_export[:,:,3] = np.uint8(output_image // (2**24) % (2**8))
                    mahotas.imsave(temp_path, vast_export)

            shutil.move(temp_path, output_path)
            
        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
        except KeyboardInterrupt:
            pass
        except:
            print "Unexpected error:", sys.exc_info()[0]
            if repeat_attempt_i == job_repeat_attempts:
                raise
            
    assert check_file(output_path), "Output file could not be verified after {0} attempts, exiting.".format(job_repeat_attempts)
