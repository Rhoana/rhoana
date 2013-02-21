#include <stdlib.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <H5Cpp.h>

using namespace cv;
using namespace std;

void write_feature(H5::H5File h5file, const Mat &image, const char *name);

#define NUM_HIST 10

string localhist_feature_name(int histbin)
{
    ostringstream s;
    s << "local_hist_" << histbin;
    return string(s.str());
}

void local_statistics(Mat &image_in, int windowsize, H5::H5File &h5f)
{
    Mat image;
    image_in.convertTo(image, CV_32F);

    // we compute statistics using a Guassian blur, rather than a hard window
    Mat mean;
    GaussianBlur(image, mean, Size(0, 0), windowsize);
    write_feature(h5f, mean, "local_mean");
    
    Mat var;
    GaussianBlur(image.mul(image), var, Size(0, 0), windowsize);
    var = var - mean.mul(mean);
    write_feature(h5f, var, "local_variance");
    
    // histogram features
    Mat smoothed_count;
    for (int i = 0; i < NUM_HIST; i++) {
        float threshold_lo = i * 256.0 / NUM_HIST;
        float threshold_hi = (i+1) * 256.0 / NUM_HIST;
        Mat mask = (image >= threshold_lo).mul(image < threshold_hi);
        mask.convertTo(mask, CV_32F);
        GaussianBlur(mask, smoothed_count, Size(0, 0), windowsize);
        write_feature(h5f, smoothed_count, localhist_feature_name(i).c_str());
    }
}
