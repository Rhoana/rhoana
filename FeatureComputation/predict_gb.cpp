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
    CvSeq* weak_classifiers = classifier.get_weak_predictors();

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
    Mat prediction = Mat::zeros(imsize, CV_32FC1);
    Rect fullrect(0, 0, imsize.width, imsize.height);

    for (int basecol = 0; basecol < imsize.width; basecol += col_block_size) {
        for (int baserow = 0; baserow < imsize.height; baserow += row_block_size) {
            cout << basecol << " " << baserow << endl;
            Rect roi(basecol, baserow, col_block_size, row_block_size);
            roi &= fullrect;

            vector<Mat> features(num_features);
            // Read features
            for (int fnum = 0; fnum < num_features; fnum++) {
                read_feature(h5f, features[fnum], names[fnum].c_str(), roi);
            }

            // Predict
            Mat GB_sum(features[0].size(), CV_32F);
            int roi_size = roi.width * roi.height;

            CvSeqReader reader;
            cvStartReadSeq( weak_classifiers, &reader );
            cvSetSeqReaderPos( &reader, 0);

            for (int wi = 0; wi < weak_classifiers->total; wi++) {
                CvBoostTree* bt;
                CV_READ_SEQ_ELEM( bt, reader );
                const CvDTreeNode *root = bt->get_root();
                CvDTreeSplit* split = root->split;
                int fnum = split->var_idx;
                float thresh = split->ord.c;
                float left_val = root->left->value;
                float right_val = root->right->value;
                if (split->inversed) {
                    float temp = left_val;
                    left_val = right_val;
                    right_val = temp;
                }
                Mat result(features[0].size(), CV_32F);

                // features and GB_sum are continuous
                float *feature_ptr = features[fnum].ptr<float>(0);
                float *score_ptr = GB_sum.ptr<float>(0);


                for (int i = 0; i < roi_size; i++, feature_ptr++, score_ptr++)
                    *score_ptr += ((*feature_ptr <= thresh) ? left_val : right_val);
            }
            // convert to probability
            float *score_ptr = GB_sum.ptr<float>(0);
            for (int i = 0; i < roi_size; i++, score_ptr++)
                *score_ptr = 1 / (1 + exp(- *score_ptr));

            GB_sum.copyTo(prediction(roi));
        }
    }

    if (argc == 3) {
        normalize(prediction, prediction, 0, 1, NORM_MINMAX);
        imshow("result", prediction);
        waitKey(0);
    } else {
        // Write probabilities, original, and all membrane features for next step
        H5File h5fout = create_feature_file(argv[3], prediction);
        write_feature(h5fout, prediction, "probabilities");
        Mat feature;
        read_feature(h5f, feature, "original");
        write_feature(h5fout, feature, "original");
        for (int fnum = 0; fnum < num_features; fnum++) {
            if (names[fnum].find("membrane") != string::npos) {
                read_feature(h5f, feature, names[fnum].c_str());
                write_feature(h5fout, feature, names[fnum].c_str());
            }
        }
    }
}
