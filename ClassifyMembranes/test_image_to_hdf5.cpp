#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <H5Cpp.h>
using namespace cv;

H5::H5File create_feature_file(char *filename, const Mat &image);
void write_feature(H5::H5File h5file, const Mat &image, const char *name);

int main(int argc, char** argv) {
  char *input_image = argv[1];
  char *output_hdf5 = argv[2];
  char *suboutput_hdf5 = argv[3];
  
  /* Read input, convert to grayscale */
  Mat image;
  image = imread( input_image, 1 );
  image.convertTo(image, CV_8U);
  cvtColor(image, image, CV_RGB2GRAY);
  
  /* create the dataset */
  H5::H5File featurefile = create_feature_file(output_hdf5, image);
  
  /* write the image */
  write_feature(featurefile, image, "input_image");
  
  /* write subimage */
  image = image(Rect(10, 10, image.size().width - 20, image.size().height - 20));
  /* create a dataset for a subimage */
  featurefile = create_feature_file(suboutput_hdf5, image);
  write_feature(featurefile, image, "subimage");  
}



