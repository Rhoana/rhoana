#include <stdlib.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace cv;
using namespace std;

#define MIN_RADIUS 2
#define MAX_RADIUS 6

string vesicle_feature_name(int radius)
{
    ostringstream s;
    s << "vesicle_" << radius;
    return string(s.str());
}

void vesicles(const Mat &image_in, void (*feature_callback)(const Mat &image, const char *name))
{
    Mat maxv = Mat::zeros(image_in.size(), CV_32F);
    Mat minv = Mat::ones(image_in.size(), CV_32F);
    for (int r = MIN_RADIUS; r <= MAX_RADIUS; r++) {
        int windowsize = 2 * r + 3;
        Mat tmplate = Mat::ones(windowsize, windowsize, CV_8U) * 255;
        circle(tmplate, Point(r + 1, r + 1), r, Scalar(0));
        circle(tmplate, Point(r + 1, r + 1), r + 1, Scalar(0));
        // OpenCV's template matching returns only the "valid" region (if compared to normxcorr2 in matlab).  We pre-pad
        // the image to result in "same" behavior.
        Mat image;
        copyMakeBorder(image_in, image,
                       windowsize / 2, windowsize / 2,
                       windowsize / 2, windowsize / 2,
                       BORDER_REFLECT_101);

        Mat match;
        matchTemplate(image, tmplate, match, CV_TM_CCORR_NORMED);
        feature_callback(match, vesicle_feature_name(r).c_str());
        maxv = max(maxv, match);
        minv = min(minv, match);
    }
    feature_callback(maxv, "vesicle_max");
    feature_callback(minv, "vesicle_min");
    GaussianBlur(maxv, maxv, Size(0, 0), 2);
    GaussianBlur(minv, minv, Size(0, 0), 2);
    feature_callback(maxv, "vesicle_blurmax");
    feature_callback(minv, "vesicle_blurmin");
}

