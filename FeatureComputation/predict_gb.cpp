// predict_randomforest.cpp - 

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/ml.h>
#include <H5Cpp.h>
#include <assert.h>
using namespace cv;
using namespace H5;
using namespace std;

H5::H5File create_feature_file(char *filename, const Mat &image);
H5File open_feature_file(char *filename);
vector<string> get_feature_names(H5File h5f);
void read_feature(H5File h5f, Mat &image_out, const char *name, const Rect &roi=Rect(0,0,0,0));
void read_feature_size(H5File h5f, Size &size_out, const char *name);
void write_feature(H5::H5File h5file, const Mat &image, const char *name);

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
    CvBoost classifier;
    classifier.read(*fs, *fs["classifier"]);

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
                    float sum = classifier.predict(stacked_features.row(stacked_row_offset), Mat(), Range::all(), false, true);
                    // cout << sum << " " << 1 / (1 + exp(-sum)) << endl;
                    *dest = 1 / (1 + exp(-sum));
                }
            }
        }
    }

    if (argc == 3) {
        normalize(prediction, prediction, 0, 1, NORM_MINMAX);
        imshow("result", prediction);
        waitKey(0);
    } else {
        H5File h5fout = create_feature_file(argv[3], prediction);
        write_feature(h5fout, prediction, "probabilities");
        for (int fnum = 0; fnum < num_features; fnum++) {
            if (names[fnum].find("membrane") != string::npos) {
                Mat feature;
                read_feature(h5f, feature, names[fnum].c_str());
                write_feature(h5fout, feature, names[fnum].c_str());
            }
        }
    }
}
