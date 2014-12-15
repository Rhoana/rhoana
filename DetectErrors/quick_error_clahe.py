import numpy as np
import mahotas
import scipy.ndimage
import clahe
import shutil
import sys
import os

job_repeat_attempts = 5

def check_file(filename):
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        return False
    return True

def generate_disc(disc_rad):
    y,x = np.ogrid[-disc_rad:disc_rad+1, -disc_rad:disc_rad+1]
    return (x*x + y*y <= disc_rad*disc_rad)

if __name__ == '__main__':

    output_path = sys.argv[1]
    input_prev = sys.argv[2]
    input_target = sys.argv[3]
    input_next = sys.argv[4]

    input_images = (input_prev, input_target, input_next)

    input_existing_errors = None
    if len(sys.argv) > 7:
        input_errors_prev = sys.argv[5]
        input_errors_target = sys.argv[6]
        input_errors_next = sys.argv[7]
        input_existing_errors = (input_errors_prev, input_errors_target, input_errors_next)

    # Default settings
    qe_blur_sigma, qe_thresh, qe_min_size = (7, 50, 300)

    qe_margin_radius = 15
    qe_min_gap_size = 5000

    qe_dilate_radius = 30
    qe_erode_radius = 20

    # Load environment settings
    if 'CONNECTOME_SETTINGS' in os.environ:
        settings_file = os.environ['CONNECTOME_SETTINGS']
        execfile(settings_file)
    
    margin_disc = generate_disc(qe_margin_radius)
    dilate_disc = generate_disc(qe_dilate_radius)
    erode_disc = generate_disc(qe_erode_radius)

    repeat_attempt_i = 0
    while repeat_attempt_i < job_repeat_attempts and not check_file(output_path):

        repeat_attempt_i += 1

        try:

            error_vol = None
            existing_error_labels = None
            vol = None
            #debug_image = None

            mask = None

            check_for_errors = True

            if input_existing_errors is not None:

                print "Opening error mask images {0}.".format(input_existing_errors)

                for index in range(3):
                    error_image = mahotas.imread(input_existing_errors[index]) > 0

                    if error_vol is None:
                        error_vol = np.zeros((error_image.shape[0], error_image.shape[1], 3), dtype=np.uint8)

                    if index == 1:
                        mask = error_image

                    error_vol[:,:,index] = error_image

                existing_error = np.sum(error_vol > 0, axis=2) == 3

                if np.any(existing_error):

                    existing_error_labels, nexisting_error = mahotas.label(existing_error)
                    existing_error_sizes = mahotas.labeled.labeled_size(existing_error_labels)
                    too_small = np.nonzero(existing_error_sizes < qe_min_size)
                    existing_error_labels, nexisting_error = mahotas.labeled.relabel(mahotas.labeled.remove_regions(existing_error_labels, too_small))

                    if nexisting_error > 0:
                        print "Found {0} existing error regions.".format(nexisting_error)
                    else:
                        mask = np.zeros(existing_error_labels.shape, dtype=np.bool)
                        check_for_errors = False

            if check_for_errors:

                print "Opening images {0}.".format(input_images)

                for index in range(3):

                    raw_image = mahotas.imread(input_images[index])
                    input_image = np.zeros(raw_image.shape, dtype=raw_image.dtype)
                    clahe.clahe(raw_image, input_image, 3)
                    raw_image = None

                    # if index == 1:
                    #     debug_image = input_image

                    # Blur
                    if qe_blur_sigma != 0:
                        input_image = scipy.ndimage.gaussian_filter(input_image, qe_blur_sigma)

                    if vol is None:
                        vol = np.zeros((input_image.shape[0], input_image.shape[1], 3), dtype=np.float32)

                    vol[:,:,index] = input_image

                diff = np.abs(np.diff(vol, axis=2))
                mindiff = np.min(diff, axis=2)

                # Remove dust
                noise_labels, nnoise = mahotas.label(mindiff > qe_thresh)
                noise_sizes = mahotas.labeled.labeled_size(noise_labels)
                too_small = np.nonzero(noise_sizes < qe_min_size)
                noise_labels, nnoise = mahotas.labeled.relabel(mahotas.labeled.remove_regions(noise_labels, too_small))

                # Close holes
                margin_mask = mahotas.dilate(noise_labels > 0, margin_disc)
                gaps, ngaps = mahotas.label(margin_mask == 0)
                gap_sizes = mahotas.labeled.labeled_size(gaps)
                too_small = np.nonzero(gap_sizes < qe_min_gap_size)
                gaps, ngaps = mahotas.labeled.relabel(mahotas.labeled.remove_regions(gaps, too_small))

                if input_existing_errors is None:
                    mask = gaps==0
                    print "Found {0} error regions ({1} pixels).".format(nnoise, np.sum(mask))
                else:
                    remove_mask = np.logical_and(existing_error_labels > 0, gaps!=0)
                    remove_labels, nremove = mahotas.label(remove_mask)
                    print "Removing {0} error regions ({1} pixels).".format(nremove, np.sum(remove_mask))
                    mask[remove_mask] = 0

                #if np.any(mask):
                    #figure(figsize=(40,20));imshow(debug_image,cmap=cm.gray);title(image_names[test_index])
                    #figure(figsize=(40,20));imshow(debug_image*mask,cmap=cm.gray);title(image_names[test_index])

            mahotas.imsave(output_path.replace('.', '_partial.', 1), np.uint8(mask*255))
            shutil.move(output_path.replace('.', '_partial.', 1), output_path)
            print "Wrote to {0}.".format(output_path)

        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
        except KeyboardInterrupt:
            pass
        # except:
        #     print "Unexpected error:", sys.exc_info()[0]
        #     if repeat_attempt_i == job_repeat_attempts:
        #         raise
            
    assert check_file(output_path), "Output file could not be verified after {0} attempts, exiting.".format(job_repeat_attempts)

