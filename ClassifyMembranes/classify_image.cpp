#include <stdlib.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <H5Cpp.h>
#include <fstream>
#include <getopt.h>
#include <assert.h>
using namespace cv;
using namespace std;

H5::H5File create_feature_file(char *filename, const Mat &image);
void write_feature(H5::H5File h5file, const Mat &image, const char *name);

void adapthisteq(const Mat &in, Mat &out, float regularizer);
void membranes(Mat &image_in, int windowsize, int membranewidth, void (*feature_callback)(const Mat &image, const char *name));
void local_statistics(Mat &image_in, int windowsize, void (*feature_callback)(const Mat &image, const char *name));
void tensor_gradient(Mat &image_in, void (*feature_callback)(const Mat &image, const char *name));
void vesicles(const Mat &image_in, void (*feature_callback)(const Mat &image, const char *name));
void drawhist(const Mat &src, const char *windowname);

static int verbose;

/* The options we understand. */
static struct option long_options[] = {
  /* These options set a flag. */
  {"verbose", no_argument, &verbose, 1},
  /* These options don't set a flag.
   We distinguish them by their indices. */
  {"windowsize",  required_argument, 0, 'w'},
  {"membranewidth",  required_argument, 0, 'm'},
  {0, 0, 0, 0}
};

/* Struct for weak learners */
struct WL {
    string feature_name;
    float threshold, left_val, right_val;
    bool operator==(const string &rhs) {
        return feature_name == rhs;
    }
};

static Mat_<float> prediction;
static vector<struct WL> weak_learners;
static H5::H5File h5f;

bool should_save(const char *name)
{
    string s = string(name);
    if (s == "original") return true;
    if ((s.find("membrane") == 0) &&
        (s[s.size() - 1] >= '0') &&
        (s[s.size() - 1] <= '9')) return true;
    return false;
}

void add_feature(const Mat &image, const char *name)
{
    assert (prediction.size() == image.size());
    int found = 0;
    string _name = string(name);
    cout << "Got feature " << name << " size " << image.rows << " " << image.cols << endl;
    assert (prediction.size() == image.size());
    Mat _image;
    if (image.depth() != CV_32F) {
      image.convertTo(_image, CV_32F);
    } else {
      _image = image;
    }
    for (int wi = 0; wi < weak_learners.size(); wi++) {
        if (weak_learners[wi].feature_name == _name) {
            found = 1;
            float *score_ptr = prediction.ptr<float>(0);
            const float *feature_ptr = _image.ptr<float>(0);
            float thresh = weak_learners[wi].threshold;
            float left_val = weak_learners[wi].left_val;
            float right_val = weak_learners[wi].right_val;
            for (int i = 0; i < prediction.total(); i++, feature_ptr++, score_ptr++)
                    *score_ptr += ((*feature_ptr <= thresh) ? left_val : right_val);
        }
    }
    if (! found)
        cout << "Didn't find any uses of feature " << name << endl;

    
    // remove old weak learners from consideration
    weak_learners.erase(remove(weak_learners.begin(), weak_learners.end(), name), weak_learners.end());

    if (should_save(name))
        write_feature(h5f, image, name);
}

int main(int argc, char** argv) {
  /* Default values. */
  int windowsize = 19;
  int membranewidth = 3;
  
  while (1) {
    int option_index = 0;
    int c = getopt_long (argc, argv, "w:m:", long_options, &option_index);
    
    /* Detect the end of the options. */
    if (c == -1)
      break;
    switch (c) {
      case 0:
        /* If this option set a flag, do nothing else now. */
        if (long_options[option_index].flag != 0)
          break;
        break;
        
      case 'w':
        windowsize = atoi(optarg);
        assert ((windowsize % 2) == 1);
        break;
        
      case 'm':
        membranewidth = atoi(optarg);
        assert ((membranewidth % 2) == 1);
        break;
        
      case '?':
        /* getopt_long already printed an error message. */
        break;
        
      default:
        abort ();
    }
  }
  
  assert (argc - optind == 3);  /* 3 required arguments */
  char *input_image = argv[optind];
  char *classifier_file = argv[optind + 1];
  char *output_hdf5 = argv[optind + 2];

  cout << "Classifying and storing features from " << input_image << " in " << output_hdf5 << endl;
  
  /* Read input, convert to grayscale */
  Mat image;
  image = imread(input_image, 0);
  image.convertTo(image, CV_8U);
  
  /* allocate the output prediction */
  prediction = Mat::zeros(image.size(), CV_32F);

  /* create the feature file */
  h5f = create_feature_file(output_hdf5, image);

  /* Parse the classifier */
  ifstream classifier(classifier_file);
  while (! classifier.eof()) {
      struct WL temp;
      string ignore;
      classifier >> temp.feature_name >> ignore >> temp.threshold >> ignore >> temp.left_val >> ignore >> temp.right_val;
      if (temp.feature_name == "") break;
      assert (classifier.peek() == '\n');
      weak_learners.push_back(temp);
  }

  /* compute and write features */
  
  /* FEATURE: Original image */
  add_feature(image, "original");

  /* normalize image */
  adapthisteq(image, image, 2);  // max CDF derivative of 2

  /* FEATURE: normxcorr with small circles */
  vesicles(image, add_feature);

  /* FEATURE: Normalized image */
  add_feature(image, "adapthisteq");

  /* FEATURE: normalized cross-correlation with membrane template, with statistics */
  membranes(image, windowsize, membranewidth, add_feature);

  /* FEATURE: local statistics: mean, variance, and pixel counts per-decile */
  local_statistics(image, windowsize, add_feature);

  /* FEATURE: successively smoothed versions of (image, anisotropy, gradient magnitude) */
  tensor_gradient(image, add_feature);

  /* Make sure we've found features for every weak learner */
  assert (weak_learners.empty());

  /* adjust features from logistic to probability */
  float *p = prediction.ptr<float>(0);
  for (int i = 0; i < prediction.total(); i++)
      p[i] = 1.0 / (1.0 + exp(- (p[i])));

  /* write out prediction */
  write_feature(h5f, prediction, "probabilities");
}
