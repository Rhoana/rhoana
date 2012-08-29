Density2d is a reference implementation of the edge detection and membrane alignment algorithms discussed in the accompanying article.

Parameters have been optimised for serial 50nm thick transmission electron microscope (TEM) images of the mushroom body calycal neuropiles of Drosophila Melanogaster, scanned at a resolution of 3.7 nm per pixel.

Usage:
density2d (-f filename | -d directory) -op outputPrefix -os outputSuffix
	[-display] [-verbose] [-thresh #.##] [-step #] [-rot #] [-scale #.##] [-open] [-align]

-f filename
-d directory
Specify filename or directory of image(s) to be segmented. If a directory is specified all files with extentions .png .jpg .jpeg .tif .tiff .bmp  and .gif within the specified directory will be processed.

-op outputPrefix
-os outputSuffix
Append a prefix or suffix to the output images. Output image name is based on the original image file name, with prefix and suffix added. Argument outputPrefix can be used to output to a different directory. Default is an output suffix of '_density2d' so that image 'test.tif' will be segmented in the file 'test_dentity2d.png', in the same directory.
For example to process all images in the test_img directory and save outputs in the test_out directory run the command:
density2d -d test_img -op ..\test_out

-display
Display progress in a figure window.

-verbose
Display verbose messages.

-thresh #.##
Segmentation threshold. Specify a number between 0 and 20. Higher numbers will produce more segments based on parameters optimised to minimise false positive and false negative detections. Default 5. Specify '-thresh r' to use rand index optimised parameters.

-step #
Step size (in pixels) to use when tracing membrane. Specify a number between 4 and 20. Default 10. Use a smaller number for more accurate, slower segmentation.

-rot #
Number of filter rotations to use at each point. Specify any integer, 10 or above. Default 128. Larger numbers produce a more accurate segmentation, but are slower and require more memory. Numbers larger than 128 may encounter memory problems.

-scale #.##
Scaling of images. Parameters were optimised for an image resolution of 3.7nm per pixel. Input images can be scaled to approximately match this resolution using this parameter. For example '-scale 1.5' will enlarge images by a factor of 1.5 before segmentation.

-open
By default unclosed edges detected by the algorithm are closed using a shortest path approach. Use the '-open' argument to skip this step.

-align
Perform alignment on membrane detected in consecutive images. Additional output .csv files will be produced containing x and y offset to the previous image, and the cumulitave offset from the first image.


GPU acceleration has been disabled by default because of platform compatibility problems.
A Windows implementation of GPU convolution acceleration has been included in the MultiConvolve directory.
In order to enable GPU convolution edit the file density2d_draw_lines.m.

Contact Seymour Knowles-Barley (seymour.kb@ed.ac.uk) for further information or to report or fix bugs.
