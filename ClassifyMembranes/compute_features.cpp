#include <stdlib.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <H5Cpp.h>
#include <iostream>
#include <assert.h>
using namespace cv;
using namespace std;

#ifdef WIN32
#include "XGetopt.h"
#else
#include <getopt.h>
#endif

H5::H5File create_feature_file(char *filename, const Mat &image);
void write_feature(H5::H5File h5file, const Mat &image, const char *name);

void adapthisteq(const Mat &in, Mat &out, float regularizer);
void membranes(Mat &image_in, int windowsize, int membranewidth, void (*feature_callback)(const Mat &image, const char *name));
void local_statistics(Mat &image_in, int windowsize, void (*feature_callback)(const Mat &image, const char *name));
void tensor_gradient(Mat &image_in, void (*feature_callback)(const Mat &image, const char *name));
void vesicles(const Mat &image_in, void (*feature_callback)(const Mat &image, const char *name));

static H5::H5File h5f;
// callback for feature writing
void add_feature(const Mat &image, const char *name)
{
    write_feature(h5f, image, name);
}

static int verbose;

///* The options we understand. */
//static struct option long_options[] = {
//  /* These options set a flag. */
//  {"verbose", no_argument, &verbose, 1},
//  /* These options don't set a flag.
//   We distinguish them by their indices. */
//  {"windowsize",  required_argument, 0, 'w'},
//  {"membranewidth",  required_argument, 0, 'm'},
//  {0, 0, 0, 0}
//};

int main(int argc, char** argv) {
  /* Default values. */
  int windowsize = 19;
  int membranewidth = 3;
  
  while (1) {
    int option_index = 0;

    //int c = getopt_long (argc, argv, "w:m:", long_options, &option_index);
    int c = getopt(argc, argv, _T("w:m:"));

    /* Detect the end of the options. */
    if (c == -1)
      break;
    switch (c) {
        
      case 'w':
        windowsize = atoi(optarg);
        assert ((windowsize % 2) == 1);
        break;
        
      case 'm':
        membranewidth = atoi(optarg);
        assert ((membranewidth % 2) == 1);
        break;
        
      case '?':
        /* getopt_long already printed an error message. */
        break;
        
      default:
        abort ();
    }
  }

  assert (argc - optind == 2);  /* 2 required arguments */
  char *input_image = argv[optind];
  char *output_hdf5 = argv[optind + 1];

  cout << "Storing features from " << input_image << " in " << output_hdf5 << endl;
  
  /* Read input, convert to grayscale */
  Mat image;
  image = imread(input_image, 0);
  image.convertTo(image, CV_8U);
  
  /* create the feature file */
  h5f = create_feature_file(output_hdf5, image);

  /* compute and write features */
  
  /* FEATURE: Original image */
  add_feature(image, "original");

  /* FEATURE: normxcorr with small circles */
  vesicles(image, add_feature);

  /* normalize image */
  adapthisteq(image, image, 2);  // max CDF derivative of 2

  /* FEATURE: Normalized image */
  add_feature(image, "adapthisteq");

  /* FEATURE: normalized cross-correlation with membrane template, with statistics */
  membranes(image, windowsize, membranewidth, add_feature);

  /* FEATURE: local statistics: mean, variance, and pixel counts per-decile */
  local_statistics(image, windowsize, add_feature);

  /* FEATURE: successively smoothed versions of (image, anisotropy, gradient magnitude) */
  tensor_gradient(image, add_feature);
}
