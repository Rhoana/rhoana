// predict_randomforest.cpp - 

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/ml.h>
#include <H5Cpp.h>
#include <assert.h>
using namespace cv;
using namespace H5;
using namespace std;

H5File open_feature_file(char *filename);
vector<string> get_feature_names(H5File h5f);
void read_feature(H5File h5f, Mat &image_out, const char *name, const Rect &roi);
void read_feature_size(H5File h5f, Size &size_out, const char *name);

vector<string> feature_names(H5File h5f)
{
    static vector<string> names;
    static int done = 0;

    if (! done) {
        done = 1;
        names = get_feature_names(h5f);
        sort(names.begin(), names.end());
    }
    return names;
}

int main(int argc, char** argv)
{
    cv::FileStorage fs(argv[1], cv::FileStorage::READ);
    CvRTrees forest;
    forest.read(*fs, *fs["forest"]);

    // Fetch features
    H5File h5f = open_feature_file(argv[2]);
    vector<string> names = feature_names(h5f);
    int num_features = names.size();

    // Find image size
    Size imsize;
    read_feature_size(h5f, imsize, names[0].c_str());

    // figure out how many chunks to break into
    int row_block_size = imsize.height / (imsize.height / 1024) + 1;
    int col_block_size = imsize.width / (imsize.width / 1024) + 1;

    // Output image
    Mat prediction(imsize, CV_32FC1);
    Rect fullrect(0, 0, imsize.width, imsize.height);

    for (int basecol = 0; basecol < imsize.width; basecol += col_block_size) {
        for (int baserow = 0; baserow < imsize.height; baserow += row_block_size) {
            cout << basecol << " " << baserow << endl;
            Rect roi(basecol, baserow, col_block_size, row_block_size);
            roi &= fullrect;

            // Stack columns
            Mat stacked_features(roi.width * roi.height, num_features, CV_32F);
            for (int fnum = 0; fnum < num_features; fnum++) {
                Mat feature;
                Mat dest;
                read_feature(h5f, feature, names[fnum].c_str(), roi);
                feature.reshape(0, roi.width * roi.height).copyTo(stacked_features.col(fnum));
            }
            
            Mat submat = prediction(roi);
            int stacked_row_offset = 0;
            for (int outrow = 0; outrow < roi.height; outrow++) {
                float *dest = submat.ptr<float>(outrow);
                for (int outcol = 0; outcol < roi.width; outcol++, stacked_row_offset++, dest++) {
                    *dest = forest.predict_prob(stacked_features.row(stacked_row_offset));
                }
            }
        }
    }
    imshow("result", prediction);
    waitKey(0);
}
