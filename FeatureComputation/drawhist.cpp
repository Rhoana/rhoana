#include <opencv/cv.h>
#include <opencv/highgui.h>
using namespace cv;

void drawhist(const Mat &src, const char *windowname)
{
    int histSize = 256;
    float range[] = { 0, 256 } ;
    const float* histRange = { range };
    bool uniform = true; bool accumulate = false;
    Mat hist;
    calcHist(&src, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );

    // Draw the histograms for B, G and R
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );

    Mat histImage( hist_h, hist_w, CV_8UC1, Scalar( 0,0,0) );
    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    /// Draw for each channel
    for( int i = 1; i < histSize; i++ ) {
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) ,
              Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
              Scalar(255), 2, 8, 0  );
    }

    /// Display
    namedWindow(windowname, CV_WINDOW_AUTOSIZE );
    imshow(windowname, histImage );
}
