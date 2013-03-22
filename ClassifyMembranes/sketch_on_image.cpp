#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <H5Cpp.h>
#include <assert.h>
using namespace cv;
using namespace H5;
using namespace std;

Mat im, overlay;
bool show_overlay = true;

// Implement mouse callback
static void onMouse(int event, int x, int y, int flags, void* ign)
{
    static Point last_draw(0,0);
    
    switch(event) {
    case CV_EVENT_MOUSEMOVE: 
        if (flags & CV_EVENT_FLAG_LBUTTON) {
            if (last_draw != Point(0, 0)) {
                line(im, last_draw, Point(x, y), CV_RGB(255,0,0));
            }
            last_draw = Point(x, y);
        }
        else if (flags & CV_EVENT_FLAG_MBUTTON) {
            if (last_draw != Point(0, 0)) {
                line(im, last_draw, Point(x, y), CV_RGB(0,255,0));
            }
            last_draw = Point(x, y);
        }
        else if (flags & CV_EVENT_FLAG_RBUTTON) {
            if (last_draw != Point(0, 0)) {
                line(im, last_draw, Point(x, y), CV_RGB(0,0,255));
            }
            last_draw = Point(x, y);
        } else {
            last_draw = Point(0, 0);
        }
        
        break;
    };
    if (! show_overlay) {
        imshow("im", im);
    } else {
        Mat tmp;
        add(0.9 * im, overlay, tmp);
        imshow("im", tmp);
    }
}

int main(int argc, char** argv)
{
    assert (argc == 4);
    show_overlay = true;
    im = imread(argv[1]);
    overlay = imread(argv[2]);
    overlay = 0.1 * overlay;
    Mat tmp;
    add(im, overlay, tmp);
    imshow("im", tmp);
    cvSetMouseCallback("im",&onMouse, 0 );
    while (true) {
        int k = cvWaitKey(0);
        if (k == ' ') {
            show_overlay = ! show_overlay;
            if (! show_overlay) {
                imshow("im", im);
            } else {
                Mat tmp;
                add(0.9 * im, overlay, tmp);
                imshow("im", tmp);
            }
        }
        if (k == 27)
            break;
    }
    imwrite(argv[3], im);
}
