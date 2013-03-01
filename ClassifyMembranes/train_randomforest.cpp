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
void read_feature(H5File h5f, Mat &image_out, const char *name, const Rect &roi=Rect(0,0,0,0));

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
    Mat data, labels;
    int base = 0;
    int total_num_negative = 0, total_num_positive = 0;

    for (int i = 1; i < argc; i += 2) {
        // Load the labeled image
        Mat image = imread(argv[i], 1);
        image.convertTo(image, CV_8U);
        vector<Mat> planes;
        split(image, planes);

        // Find training examples
        SparseMat positive_mask((planes[0] == 0) & (planes[1] == 255) & (planes[2] == 0));
        SparseMat negative_mask((planes[0] == 0) & (planes[1] == 0) & (planes[2] == 255));

        // count number of new examples
        int num_positive = positive_mask.nzcount();
        int num_negative = negative_mask.nzcount();
        total_num_positive += num_positive;
        total_num_negative += num_negative;

        cout << argv[i] << " " << num_positive << " positive, " << num_negative << " negative" << endl;

        // Fetch features
        H5File h5f = open_feature_file(argv[i + 1]);
        vector<string> names = feature_names(h5f);

        // allocate new rows in training data
        if (base == 0) {
            data.create(num_positive + num_negative, names.size(), CV_32F);
            labels.create(num_positive + num_negative, 1, CV_32SC1);
        } else {
            data.resize(base + num_positive + num_negative);
            labels.resize(base + num_positive + num_negative);
        }

        float *dataptr = data.ptr<float>(base);
        int num_features = names.size();
        for (int fnum = 0; fnum < num_features; fnum++, dataptr++) {
            Mat feature_im;
            read_feature(h5f, feature_im, names[fnum].c_str());


            float *dest = dataptr;

            SparseMatConstIterator it = positive_mask.begin();
            SparseMatConstIterator it_end = positive_mask.end();
            int count;
            for (count = 0; it != it_end; it++, count++, dest += num_features) {
                const SparseMat::Node *node = it.node();
                *dest = *feature_im.ptr<float>(node->idx[0], node->idx[1]);
                assert (*(positive_mask.ptr(node->idx[0], node->idx[1], 0)) == 255);
                assert (negative_mask.ptr(node->idx[0], node->idx[1], 0) == NULL);
            }
            assert (count == num_positive);

            it = negative_mask.begin();
            it_end = negative_mask.end();
            for (count = 0; it != it_end; it++, count++, dest += num_features) {
                const SparseMat::Node *node = it.node();
                *dest = *feature_im.ptr<float>(node->idx[0], node->idx[1]);
                assert (*(negative_mask.ptr(node->idx[0], node->idx[1], 0)) == 255);
                assert (positive_mask.ptr(node->idx[0], node->idx[1], 0) == NULL);
            }
            assert (count == num_negative);
        }
        // Add labels
        labels(Rect(0, base, 1, num_positive)) = Scalar(1);
        labels(Rect(0, base + num_positive, 1, num_negative)) = Scalar(0);

        base += num_positive + num_negative;
    }

    float priors[] = {total_num_negative, total_num_positive};
    CvRTParams params = CvRTParams(25, // max depth
                                   base / 100, // min sample count
                                   0, // regression accuracy: N/A here
                                   false, // compute surrogate split, no missing data
                                   2, // max number of categories (use sub-optimal algorithm for larger numbers)
                                   priors, // the array of priors
                                   true,  // calculate variable importance
                                   5,       // number of variables randomly selected at node and used to find the best split(s).
                                   500,	 // max number of trees in the forest
                                   0.01f,				// forest accuracy
                                   CV_TERMCRIT_ITER |	CV_TERMCRIT_EPS // termination criteria
                                   );

    Mat var_type = Mat(data.cols + 1, 1, CV_8U);
    var_type.setTo(Scalar(CV_VAR_NUMERICAL)); // all inputs are numerical
    var_type.at<unsigned char>(data.cols, 0) = CV_VAR_CATEGORICAL;

    CvRTrees forest;

    forest.train(data, CV_ROW_SAMPLE, labels,
                 Mat(), Mat(), var_type, Mat(), params);

    cv::FileStorage fs("forest.xml",cv::FileStorage::WRITE);
    forest.write(*fs, "forest");
}
