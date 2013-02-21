#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <assert.h>

using namespace cv;
using namespace std;

#define ABS(x) (((x) >= 0) ? (x) : (-(x)))

void hist(const Mat &in, Mat &out)
{
    int histSize = 256;
    float range[] = { 0, 256 } ;
    const float* histRange = { range };
    bool uniform = true; bool accumulate = false;
    calcHist(&in, 1, 0, Mat(), out, 1, &histSize, &histRange, uniform, accumulate);
}
                
void clamp_and_cumsum(Mat &hist, float regularizer)
{
    float total_pixels = sum(hist)[0];
    float clip_limit = (regularizer * total_pixels) / hist.total();

    assert (hist.depth() == CV_32F);
    assert (hist.total() == 256);
    assert (hist.isContinuous());

    float *ptr = hist.ptr<float>();
    float excess = 0;
    float accum = 0;

    // clamp and compute cumulative sum
    for (int i = 0; i < 256; i++, ptr++) {
        float overage = MAX(*ptr - clip_limit, 0);
        if (overage > 0.0) {
            excess += overage;
            *ptr -= overage;
        }
        // make sure 0 maps to 0
        float temp = *ptr;
        *ptr = accum;
        accum += temp;
    }
    
    // redistribute excess, and normalize to 0-255
    float scale = 255.0 / (hist.at<float>(255) + excess);
    float last = 0.0;
    excess /= 256;
    int i;
    for (i = 0, ptr = hist.ptr<float>(); i < 256; i++, ptr++) {
        *ptr += ((i + 1) * excess); // every bin after 0 has excess added
        *ptr *= scale;
        float tmp = *ptr;
        *ptr = cvRound((last + *ptr) / 2.0);  // map to halfway between low and high values, and round to nearest integer
        last = tmp; // update last to pre-averaged value
    }
}

void adapthisteq(const Mat &in, Mat &out, float regularizer)
{
    // contrast limited adapthive histogram equalization.
    // regularizer = max derivative in the remapping function.
    // 1 = no remapping.
    // higher values = more remapping.
    // 
    // Uses a fixed 8x8 grid of blocks and interpolates between their
    // remappings.  Clips before interpolation.  The regularizer is
    // approximate.  Some points in the remapping may be slightly
    // higher.

    // matrix of local histograms
    Mat localhists[8][8];

    // slightly overestimate to insure coverage
    int block_width = (in.cols + 7) / 8;
    int block_height = (in.rows + 7) / 8;
    Rect imrect(0, 0, in.cols, in.rows);

    // compute histograms in 8x8 subimages
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++) {
            Rect sub(i * block_width, j * block_height,
                     block_width, block_height);
            if ((i == 7) || (j == 7))
                sub &= imrect; // Avoid going off the edge
            hist(in(sub), localhists[i][j]);
        }

    // clamp histograms and compute remapping function (in place)
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            clamp_and_cumsum(localhists[i][j], regularizer);
    
    // weight function for normalization
    Mat weight = Mat::zeros(in.size(), CV_32F);
    Mat output = Mat::zeros(in.size(), CV_32F);
    
    // region weighting mask
    Mat ROIweight(2 * block_height + 1, 2 * block_width + 1, CV_32F);
    for (int j = 0; j < 2 * block_height + 1; j++)
        for (int i = 0; i < 2 * block_width + 1; i++)
            ROIweight.at<float>(j, i) = (block_height - ABS(j - block_height)) *
                (block_width - ABS(i - block_width));

    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++) {
            // upper left corner of histogram window
            int corneri = i * block_width;
            int cornerj = j * block_height;
            // bounds of ROI (histogram region of influence)
            int ROI_lo_i = corneri - (block_width / 2);
            int ROI_lo_j = cornerj - (block_height / 2);

            Rect subrect(ROI_lo_i, ROI_lo_j,
                         2 * block_width + 1, 2 * block_height + 1);

            // clip to actual image
            subrect &= imrect;
            
            // find subimage ROI after clipping
            Rect subROI = Rect(0, 0, 2 * block_width + 1, 2 * block_height + 1) & \
                Rect(-ROI_lo_i, -ROI_lo_j, in.cols, in.rows);

            Mat temp(subrect.height, subrect.width, CV_32F);

            // locally transform
            LUT(in(subrect), localhists[i][j], temp);
            multiply(temp, ROIweight(subROI), temp);
            output(subrect) += temp;
            weight(subrect) += ROIweight(subROI);
        }

    output /= weight;
    output.convertTo(out, CV_8U);
}
