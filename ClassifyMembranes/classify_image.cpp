#include <unistd.h>
#include <stdlib.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <H5Cpp.h>
#include <fstream>
#include <getopt.h>
#include <assert.h>
using namespace cv;
using namespace std;

#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <pthread.h> // for mutex
#include <time.h>

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

// Used in classification and saving callbacks
static Mat_<float> prediction;
static vector<struct WL> weak_learners;
static H5::H5File h5f;
pthread_mutex_t add_feature_lock;
pthread_mutex_t write_feature_lock;
static float add_time = 0;
static float write_time = 0;

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
    time_t time_in, time_out;

    pthread_mutex_lock(&add_feature_lock);
    time(&time_in);

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

    vector<struct WL> cur_learners;
    copy_if(weak_learners.begin(), weak_learners.end(), back_inserter(cur_learners), [&] (struct WL &temp) {return (temp == _name);});

    float *score_ptr = prediction.ptr<float>(0);
    const float *feature_ptr = _image.ptr<float>(0);
    cilk_for (int i = 0; i < prediction.total(); i++) {
        for (vector<struct WL>::iterator cl = cur_learners.begin(); cl != cur_learners.end(); cl++) {
            float thresh = cl->threshold;
            float left_val = cl->left_val;
            float right_val = cl->right_val;
            score_ptr[i] += ((feature_ptr[i] <= thresh) ? left_val : right_val);
        }
    }

    if (cur_learners.empty())
        cout << "Didn't find any uses of feature " << name << endl;

    // remove old weak learners from consideration
    weak_learners.erase(remove(weak_learners.begin(), weak_learners.end(), _name), weak_learners.end());

    time(&time_out);
    add_time += difftime(time_out, time_in);
    pthread_mutex_unlock(&add_feature_lock);

    if (should_save(name)) {
        pthread_mutex_lock(&write_feature_lock);
        time(&time_in);
        write_feature(h5f, image, name);
        pthread_mutex_unlock(&write_feature_lock);
        cout << "Wrote " << name << endl;
        time(&time_out);
        write_time += difftime(time_out, time_in);
    }
}

int main(int argc, char** argv) {
  int numWorkers = __cilkrts_get_nworkers();
  cout << "number of cilk workers = " << numWorkers << endl;

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
  
  assert (argc - optind == 3);  /* 2 required arguments */
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

  pthread_mutex_init(&add_feature_lock, NULL);
  pthread_mutex_init(&write_feature_lock, NULL);

  /* compute and write features */
  
  /* FEATURE: Original image */
  Mat _image;
  image.copyTo(_image);
  cilk_spawn add_feature(_image, "original");

  /* normalize image */
  adapthisteq(image, image, 2);  // max CDF derivative of 2

  /* FEATURE: Normalized image */
  cilk_spawn add_feature(image, "adapthisteq");

  /* FEATURE: normxcorr with small circles */
  cilk_spawn vesicles(image, add_feature);

  /* FEATURE: normalized cross-correlation with membrane template, with statistics */
  cilk_spawn membranes(image, windowsize, membranewidth, add_feature);

  /* FEATURE: local statistics: mean, variance, and pixel counts per-decile */
  cilk_spawn local_statistics(image, windowsize, add_feature);

  /* FEATURE: successively smoothed versions of (image, anisotropy, gradient magnitude) */
  cilk_spawn tensor_gradient(image, add_feature);

  cilk_sync;
  /* Make sure we've found features for every weak learner */
  assert (weak_learners.empty());

  /* adjust features from logistic to probability */
  prediction = -prediction;
  exp(prediction, prediction);
  prediction = 1.0 / (1.0 + prediction);

  /* write out prediction */
  write_feature(h5f, prediction, "probabilities");

  /* close the HDF5 */
  time_t time_in, time_out;
  time(&time_in);
  h5f.close();
  time(&time_out);
  cout << "Time closing HDF5: " << difftime(time_out, time_in) << endl;
  cout << "Time adding features: " << add_time << endl;
  cout << "Time writing: " << write_time << endl;
}
