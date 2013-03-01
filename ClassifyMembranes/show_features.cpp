#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <H5Cpp.h>
#include <assert.h>
using namespace cv;
using namespace H5;
using namespace std;

H5File open_feature_file(char *filename);
vector<string> get_feature_names(H5File h5f);
void read_feature(H5File h5f, Mat &image_out, const char *name, const Rect &rect = Rect(0,0,0,0));



int main(int argc, char** argv)
{
    H5File h5f = open_feature_file(argv[1]);
    vector<string> feature_names = get_feature_names(h5f);
    cout << "READ " << feature_names.size() << endl;
    for (int i = 0; i < feature_names.size(); i++) {
        Mat im;
        read_feature(h5f, im, feature_names[i].c_str());
        double minv, maxv;
        minMaxLoc(im, &minv, &maxv);
        cout << feature_names[i] << " " << minv << " " << maxv << endl;
        normalize(im, im, 0, 1, NORM_MINMAX);
        minMaxLoc(im, &minv, &maxv);
        cout << "    " << minv << " " << maxv << endl;
        
        imshow(feature_names[i], im);
    }
    waitKey(0);
}
