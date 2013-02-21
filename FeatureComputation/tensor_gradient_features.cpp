#include <stdlib.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <H5Cpp.h>

using namespace cv;
using namespace std;

void write_feature(H5::H5File h5file, const Mat &image, const char *name);

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

void tensor_gradient_features(Mat &image_in,  H5::H5File &h5f)
{
    Mat image;
    image_in.convertTo(image, CV_32F);

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
    
    // Compute smoothings of image, gradient magnitude, and anisotropy.
    // Also compute differences of gaussians.
    Mat smoothed_ims[NUM_SMOOTHINGS];
    Mat smoothed_aniso;
    Mat smoothed_mag;
    for (int i = 0; i < NUM_SMOOTHINGS; i++) {
        GaussianBlur(image, smoothed_ims[i], Size(0, 0), i + 1);
        write_feature(h5f, smoothed_ims[i], smoothed_name(i + 1, "image").c_str());
        GaussianBlur(anisotropy, smoothed_aniso, Size(0, 0), i + 1);
        write_feature(h5f, smoothed_aniso, smoothed_name(i + 1, "anisotropy").c_str());
        GaussianBlur(magnitude, smoothed_mag, Size(0, 0), i + 1);
        write_feature(h5f, smoothed_mag, smoothed_name(i + 1, "magnitude").c_str());
        for (int j = 0; j < i - 1; j+= 2) {
            Mat DoG = smoothed_ims[i] - smoothed_ims[j];
            write_feature(h5f, DoG, dog_name(i + 1, j + 1).c_str());
        }
    }

    // Large DoG
    Mat g50;
    GaussianBlur(image, g50, Size(0, 0), 50);
    write_feature(h5f, smoothed_ims[1] - g50, "DoG_2_50");
}
