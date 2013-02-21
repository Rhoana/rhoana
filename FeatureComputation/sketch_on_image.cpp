#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <H5Cpp.h>
#include <assert.h>
using namespace cv;
using namespace H5;
using namespace std;


Mat im;

// Implement mouse callback
static void onMouse(int event, int x, int y, int flags, void* ign)
{
    static Point last_draw(0,0);
    
    cout << event << " " << flags << endl;
    switch(event) {
    case CV_EVENT_MOUSEMOVE: 
        cout << "move" << y << " " << x << endl;
        if (flags & CV_EVENT_FLAG_LBUTTON) {
            cout << "draw" << last_draw.x << endl;
            if (last_draw != Point(0, 0)) {
                line(im, last_draw, Point(x, y), CV_RGB(0,255,0));
            }
        }
        if (flags & CV_EVENT_FLAG_RBUTTON) {
            cout << "draw" << last_draw.x << endl;
            if (last_draw != Point(0, 0)) {
                line(im, last_draw, Point(x, y), CV_RGB(255,0,0));
            }
        }
        last_draw = Point(x, y);
        break;
    };
    imshow("im", im);
}

int main(int argc, char** argv)
{
    im = imread(argv[1]);
    imshow("im", im);
    cvSetMouseCallback("im",&onMouse, 0 );
    cvWaitKey(0);
    imwrite(argv[2], im);
}
