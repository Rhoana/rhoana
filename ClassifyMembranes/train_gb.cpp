#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/ml.h>
#include <stdint.h>
#include <H5Cpp.h>
#include <assert.h>
#include <fstream>
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
    int total_num_red = 0, total_num_green = 0, total_num_blue = 0;

    vector<string> names = feature_names(open_feature_file(argv[2]));

    for (int i = 1; i < argc - 1; i += 2) {
        // Load the labeled image
        Mat image = imread(argv[i], 1);
        image.convertTo(image, CV_8U);
        vector<Mat> planes;
        split(image, planes);

        // Find training examples
        SparseMat red_mask((planes[0] == 0) & (planes[1] == 0) & (planes[2] == 255));
        SparseMat green_mask((planes[0] == 0) & (planes[1] == 255) & (planes[2] == 0));
        SparseMat blue_mask((planes[0] == 255) & (planes[1] == 0) & (planes[2] == 0));

        // count number of new examples
        int num_red = red_mask.nzcount();
        int num_green = green_mask.nzcount();
        int num_blue = blue_mask.nzcount();
        total_num_red += num_red;
        total_num_green += num_green;
        total_num_blue += num_blue;

        cout << argv[i] << " " << num_red << " red, " << num_green << " green, " << num_blue << " blue" << endl;

        // Fetch features
        H5File h5f = open_feature_file(argv[i + 1]);
        assert(feature_names(h5f) == names);

        // allocate new rows in training data
        if (base == 0) {
            data.create(num_red + num_green + num_blue, names.size(), CV_32F);
            labels.create(num_red + num_green + num_blue, 1, CV_32SC1);
        } else {
            data.resize(base + num_red + num_green + num_blue);
            labels.resize(base + num_red + num_green + num_blue);
        }

        float *dataptr = data.ptr<float>(base);
        int num_features = names.size();
        for (int fnum = 0; fnum < num_features; fnum++, dataptr++) {
            Mat feature_im;
            read_feature(h5f, feature_im, names[fnum].c_str());

            vector<SparseMat *> masks;
            masks.push_back(&red_mask);
            masks.push_back(&green_mask);
            masks.push_back(&blue_mask);
            float *dest = dataptr;

            for(vector<SparseMat *>::iterator mit = masks.begin(); mit != masks.end(); mit++) {
                SparseMatConstIterator it = (*mit)->begin();
                SparseMatConstIterator it_end = (*mit)->end();
                int count;
                for (count = 0; it != it_end; it++, count++, dest += num_features) {
                    const SparseMat::Node *node = it.node();
                    *dest = *feature_im.ptr<float>(node->idx[0], node->idx[1]);
                }
                assert (count == (*mit)->nzcount());
            }
        }

        // Add labels - 0,1,2 = RGB.  NB: match the vector above
        labels(Rect(0, base, 1, num_red)) = Scalar(0);
        labels(Rect(0, base + num_red, 1, num_green)) = Scalar(1);
        labels(Rect(0, base + num_red + num_green, 1, num_blue)) = Scalar(2);

        base += num_red + num_green + num_blue;
    }

    // train a classifier to distinguish green (assuming it is boundary) from red/blue

    float priors[] = {total_num_red + total_num_blue, total_num_green};
    Mat var_type = Mat(data.cols + 1, 1, CV_8U);
    var_type.setTo(Scalar(CV_VAR_NUMERICAL)); // all inputs are numerical
    var_type.at<unsigned char>(data.cols, 0) = CV_VAR_CATEGORICAL;

    Mat border_labels = labels.clone();
    for (int i = 0; i < border_labels.total(); i++)
        border_labels.at<int32_t>(i) = ((border_labels.at<int32_t>(i) == 1) ? 1 : 0);
        
    CvBoost classifier;
    CvBoostParams parms(CvBoost::GENTLE, 2000, 0.85, 1, 0, priors);
    classifier.train(data, CV_ROW_SAMPLE, border_labels,
                     Mat(), Mat(), var_type, Mat(), parms);


    // Write out the classifier in a format that we can easily transform into C code.
    ofstream classifier_out(argv[argc - 1]);

    CvSeq* weak_classifiers = classifier.get_weak_predictors();
    CvSeqReader reader;
    cvStartReadSeq(weak_classifiers, &reader);
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
        classifier_out << "BORDER " << names[fnum] << " <= " << thresh << " ? " << left_val << " : " << right_val << endl;
    }

    // Train a second classifier for Inside/Outside (Red/Blue)
    Mat inout_labels;
    inout_labels.create(total_num_red + total_num_blue, 1, CV_32SC1);
    Mat inout_data;
    inout_data.create(total_num_red + total_num_blue, names.size(), CV_32F);
    int dst = 0;
    for (int i = 0; i < labels.total(); i++) {
        if (labels.at<int32_t>(i) == 1) continue;
        inout_labels.at<int32_t>(dst) = ((labels.at<int32_t>(i) == 0) ? 0 : 1);
        data(Rect(0, i, data.cols, 1)).copyTo(inout_data(Rect(0, dst, data.cols, 1)));
        dst++;
    }
    assert (dst == total_num_blue + total_num_red);
    priors[0] = total_num_red;
    priors[1] = total_num_blue;
    CvBoost classifier_inout;
    classifier_inout.train(inout_data, CV_ROW_SAMPLE, inout_labels,
                           Mat(), Mat(), var_type, Mat(), parms);

    weak_classifiers = classifier_inout.get_weak_predictors();
    cvStartReadSeq(weak_classifiers, &reader);
    cvSetSeqReaderPos(&reader, 0);

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
        classifier_out << "INOUT " << names[fnum] << " <= " << thresh << " ? " << left_val << " : " << right_val << endl;
    }
}
