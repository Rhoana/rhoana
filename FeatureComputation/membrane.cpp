#include <stdlib.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <H5Cpp.h>

#include "quickmedian.h"

using namespace cv;
using namespace std;

void write_feature(H5::H5File h5file, const Mat &image, const char *name);

#define NUM_ROTATIONS 8

string membrane_feature_name(int windowsize, int membranewidth, int angle)
{
    ostringstream s;
    s << "membrane_" << windowsize << "_" << membranewidth << "_" << angle;
    return string(s.str());
}

void find_membranes(Mat &image_in, int windowsize, int membranewidth, H5::H5File &h5f)
{
  Mat tmplate = Mat::zeros(windowsize, windowsize, CV_8U);
  
  // OpenCV's template matching returns only the "valid" region (if compared to normxcorr2 in matlab).  We pre-pad
  // the image to result in "same" behavior.
  Mat image;
  copyMakeBorder(image_in, image,
                 windowsize / 2, windowsize / 2,
                 windowsize / 2, windowsize / 2,
                 BORDER_REFLECT_101);
  
  for (int i = (windowsize - membranewidth) / 2, ii = 0; ii < membranewidth; i++, ii++)
    for (int j = 0; j < windowsize; j++)
      tmplate.at<uchar>(i, j) = 255;

  // allocate space for all matches, to allow min/max/mean/variance/median computation per-pixel
  Mat all_matches(image_in.total(), NUM_ROTATIONS, CV_32F);
  Mat match;
  Mat rot_tmplate;
  Mat rot_mat(2, 3, CV_32FC1);
  Point center = Point(windowsize/2, windowsize/2);
  for (int step = 0; step < NUM_ROTATIONS; step++) {
      double angle = step * 180.0 / NUM_ROTATIONS;
      rot_mat = getRotationMatrix2D(center, angle, 1.0);
      warpAffine(tmplate, rot_tmplate, rot_mat, tmplate.size());
      matchTemplate(image, rot_tmplate, match, CV_TM_CCORR_NORMED);
      match.reshape(0, match.total()).copyTo(all_matches.col(step));
      write_feature(h5f, match,
                    membrane_feature_name(windowsize, membranewidth, angle).c_str());
  }

  // Compute per-pixel min, max, span, median, mean, variance
  
  Mat temp, temp2;
  reduce(all_matches, temp, 1, CV_REDUCE_MIN);
  write_feature(h5f, temp.reshape(0, image_in.rows), "membrane_min");
  reduce(all_matches, temp2, 1, CV_REDUCE_MAX);
  write_feature(h5f, temp2.reshape(0, image_in.rows), "membrane_max");
  temp = temp2 - temp;
  write_feature(h5f, temp.reshape(0, image_in.rows), "membrane_span");
  float *amptr = all_matches.ptr<float>(), *tptr = temp.ptr<float>();
  for (int i = 0; i < all_matches.rows; i++, tptr++, amptr += NUM_ROTATIONS)
      *tptr = quickmedian(amptr, NUM_ROTATIONS);
  write_feature(h5f, temp.reshape(0, image_in.rows), "membrane_median");

  // mean and variance
  reduce(all_matches, temp, 1, CV_REDUCE_AVG);
  write_feature(h5f, temp.reshape(0, image_in.rows), "membrane_mean");
  multiply(all_matches, all_matches, all_matches);
  multiply(temp, temp, temp);
  reduce(all_matches, temp2, 1, CV_REDUCE_AVG);
  temp2 -= temp;
  write_feature(h5f, temp2.reshape(0, image_in.rows), "membrane_var");
}
