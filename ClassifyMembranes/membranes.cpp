#include <stdlib.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "quickmedian.h"

#include <cilk/cilk.h>

using namespace cv;
using namespace std;

#define NUM_ROTATIONS 8

string membrane_feature_name(int windowsize, int membranewidth, int angle)
{
    ostringstream s;
    s << "membrane_" << windowsize << "_" << membranewidth << "_" << angle;
    return string(s.str());
}

void rot_min_max_span(Mat &image_in, const Mat &all_matches, void (*feature_callback)(const Mat &image, const char *name))
{
    Mat temp, temp2;
    cilk_spawn [&] {
        reduce(all_matches, temp, 1, CV_REDUCE_MIN);
        feature_callback(temp.reshape(0, image_in.rows), "membrane_min");
    }();
    cilk_spawn [&] {
        reduce(all_matches, temp2, 1, CV_REDUCE_MAX);
        feature_callback(temp2.reshape(0, image_in.rows), "membrane_max");
    }();
    cilk_sync;
    temp = temp2 - temp;
    feature_callback(temp.reshape(0, image_in.rows), "membrane_span");
}

void rot_median(Mat &image_in, const Mat &all_matches, void (*feature_callback)(const Mat &image, const char *name))
{
    Mat temp;
    temp.create(image_in.size(), CV_32FC1);
    const float *amptr = all_matches.ptr<float>();
    float *tptr = temp.ptr<float>();
    // can't use the grainsize pragma with gcccilk, for some reason
    int step = (all_matches.rows + 7) / 8;
    cilk_for (int i = 0; i < all_matches.rows; i += step) {
        for (int j = i; j < (i + step) && (j < all_matches.rows); j++)
            tptr[j] = quickmedian(amptr + j * NUM_ROTATIONS, NUM_ROTATIONS);
    }
    feature_callback(temp, "membrane_median");
}

void rot_mean_var(Mat &image_in, const Mat &all_matches, void (*feature_callback)(const Mat &image, const char *name))
{
    // mean and variance
    Mat temp, temp2;
    
    cilk_spawn reduce(all_matches, temp, 1, CV_REDUCE_AVG);
    cilk_spawn [&] {
        Mat all_sq;
        multiply(all_matches, all_matches, all_sq);
        reduce(all_sq, temp2, 1, CV_REDUCE_AVG);
    }();
    cilk_sync;
    cilk_spawn feature_callback(temp.reshape(0, image_in.rows), "membrane_mean");
    Mat var = temp2 - temp.mul(temp);
    feature_callback(var.reshape(0, image_in.rows), "membrane_var");
    cilk_sync;
}

void _membranes(Mat &image_in, int windowsize, int membranewidth, void (*feature_callback)(const Mat &image, const char *name))
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
  Mat matches[NUM_ROTATIONS];
  cilk_for (int step = 0; step < NUM_ROTATIONS; step++) {
      Mat rot_tmplate;
      Mat rot_mat(2, 3, CV_32FC1);
      Point center = Point(windowsize/2, windowsize/2);
      double angle = step * 180.0 / NUM_ROTATIONS;
      rot_mat = getRotationMatrix2D(center, angle, 1.0);
      warpAffine(tmplate, rot_tmplate, rot_mat, tmplate.size());
      matchTemplate(image, rot_tmplate, matches[step], CV_TM_CCORR_NORMED);
      matches[step].reshape(0, matches[step].total()).copyTo(all_matches.col(step));
  }

  // Compute per-pixel min, max, span, median, mean, variance
  cilk_spawn rot_min_max_span(image_in, all_matches, feature_callback);
  cilk_spawn rot_median(image_in, all_matches, feature_callback);
  cilk_spawn rot_mean_var(image_in, all_matches, feature_callback);

  // write features after other computations, since these will be
  // performed almost sequentially because of HDF5 locking.
  cilk_for (int step = 0; step < NUM_ROTATIONS; step++) {
      double angle = step * 180.0 / NUM_ROTATIONS;
      feature_callback(matches[step],
                       membrane_feature_name(windowsize, membranewidth, angle).c_str());
  }
  cilk_sync;
}

void membranes(Mat &image_in, int windowsize, int membranewidth, void (*feature_callback)(const Mat &image, const char *name))
{
    _membranes(image_in, windowsize, membranewidth, feature_callback);
}
