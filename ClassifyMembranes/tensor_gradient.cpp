#include <stdlib.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace cv;
using namespace std;

#define NUM_SMOOTHINGS 10

string smoothed_name(int sigma, const char *sub)
{
    ostringstream s;
    s << "smoothed_" << sub << "_" << sigma;
    return string(s.str());
}

string dog_name(int sigma1, int sigma2)
{
    ostringstream s;
    s << "DoG_" << sigma1 << "_" << sigma2;
    return string(s.str());
}

void mag_aniso(Mat &image_in, void (*feature_callback)(const Mat &image, const char *name), int smoothing_level)
{
    // Compute images gradients
    Mat gx, gy;
    Scharr(image_in, gx, CV_32F, 1, 0);
    Scharr(image_in, gy, CV_32F, 0, 1);

    // Gradient magnitude
    Mat magnitude;
    sqrt(gx.mul(gx) + gy.mul(gy), magnitude);

    // Compute the structure tensor, and from it, anisotropy
    Mat gx2, gxy, gy2;
    GaussianBlur(gx.mul(gx), gx2, Size(5, 5), 0);
    GaussianBlur(gx.mul(gy), gxy, Size(5, 5), 0);
    GaussianBlur(gy.mul(gy), gy2, Size(5, 5), 0);
    MatExpr trace = gx2 + gy2;
    MatExpr det = gx2.mul(gy2) - gxy.mul(gxy);
    Mat second_term;
    sqrt(trace.mul(trace) / 4 - det, second_term);
    MatExpr eig1 = trace / 2 + second_term;
    MatExpr eig2 = trace / 2 - second_term;
    Mat anisotropy = eig1 / eig2;

    feature_callback(magnitude, smoothed_name(smoothing_level, "magnitude").c_str());
    feature_callback(anisotropy, smoothed_name(smoothing_level, "anisotropy").c_str());
}

void local_difference_variance(Mat &image, void (*feature_callback)(const Mat &image, const char *name), int smoothing_level)
{
    float *base;
    float diffs[4];

    Mat variance;
    variance.create(image.size(), CV_32F);
    assert (image.isContinuous());
    for (int i = 0; i < image.rows; i++) {
        float *base = image.ptr<float>(i);
        float *out = variance.ptr<float>(i);
        for (int j = 0; j < image.cols; j++) {
            diffs[0] = base[j] - base[j - ((i > 0) ? image.cols : 0)];
            diffs[1] = base[j] - base[j + ((i < image.rows - 1) ? image.cols : 0)];
            diffs[2] = base[j] - base[j - ((j > 0) ? 1 : 0)];
            diffs[3] = base[j] - base[j + ((j < image.cols - 1) ? 1 : 0)];
            double mean = (diffs[0] + diffs[1] + diffs[2] + diffs[3]) / 4.0;
            double sq = (diffs[0]*diffs[0] + diffs[1]*diffs[1] + diffs[2]*diffs[2] + diffs[3]*diffs[3]) / 4.0;
            out[j] = sq - mean*mean;
        }
    }
    feature_callback(variance, smoothed_name(smoothing_level, "localvar").c_str());
}

void tensor_gradient(Mat &image_in, void (*feature_callback)(const Mat &image, const char *name))
{
    Mat image;
    image_in.convertTo(image, CV_32F);

    // Compute smoothings of image, gradient magnitude, and anisotropy.
    // Also compute differences of gaussians.
    Mat smoothed_ims[NUM_SMOOTHINGS];
    for (int i = 0; i < NUM_SMOOTHINGS; i++) {
        int smoothing_level = i + 1;
        GaussianBlur(image, smoothed_ims[i], Size(0, 0), smoothing_level);
        feature_callback(smoothed_ims[i], smoothed_name(smoothing_level, "image").c_str());
        mag_aniso(smoothed_ims[i], feature_callback, smoothing_level);
        for (int j = 0; j < i - 1; j+= 2) {
            Mat DoG = smoothed_ims[i] - smoothed_ims[j];
            feature_callback(DoG, dog_name(smoothing_level, j + 1).c_str());
        }
        local_difference_variance(smoothed_ims[i], feature_callback, smoothing_level);
    }

    // Large DoG (woof!)
    Mat g50;
    GaussianBlur(image, g50, Size(0, 0), 50);
    feature_callback(smoothed_ims[1] - g50, "DoG_2_50");
}
