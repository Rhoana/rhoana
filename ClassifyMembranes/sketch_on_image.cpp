#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/ml.h>
#include <stdint.h>
#include <H5Cpp.h>
#include <assert.h>
#include <fstream>
#include <algorithm>

using namespace cv;
using namespace H5;
using namespace std;

typedef vector< pair<Point,Point> > lineseg;
typedef vector<lineseg *> vec_lineseg;
lineseg red_vec, green_vec, blue_vec;
vector<string> names;
H5File h5f;
Mat data, labels;

int num_features;
int base = 0;

Mat im, overlay;
bool show_overlay = true;
int value = 0;

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

void button_pushed(int state, void* userdata){

    //Take the rois
    int num_red = 0, num_green = 0, num_blue = 0;
    
    vec_lineseg v_lineseg;
    v_lineseg.push_back(&red_vec);
    v_lineseg.push_back(&green_vec);
    v_lineseg.push_back(&blue_vec);
    int num_sample [3];
        
    for(vector<lineseg *>::iterator vit = v_lineseg.begin(); vit != v_lineseg.end(); vit++) {
        vector< pair<Point,Point> >::iterator it = (*vit)->begin();
        vector< pair<Point,Point> >::iterator it_end = (*vit)->end();
        int sample_num = 0;
        for ( ; it != it_end; it++) {
            Point startpt = (*it).first;
            Point endpt = (*it).second;
            int minx = min(startpt.x, endpt.x), maxx = max(startpt.x, endpt.x);
            int miny = min(startpt.y, endpt.y), maxy = max(startpt.y, endpt.y);
            int width = maxx - minx + 1, height = maxy - miny + 1;
            //cout << " Width " << width << " Height "  << height << endl;
            sample_num += max(width,height);
        }
        num_sample[vit - v_lineseg.begin()] = sample_num;
    }
    
    num_red = num_sample[0]; num_green = num_sample[1]; num_blue = num_sample[2];
    
    cout << "Number of red samples added: " << num_red << endl;
    cout << "Number of green samples added: " << num_green << endl;
    cout << "Number of blue samples added: " << num_blue << endl;
    
    labels.resize(base + num_red + num_green + num_blue);
    labels(Rect(0, base, 1, num_red)) = Scalar(0);
    labels(Rect(0, base + num_red, 1, num_green)) = Scalar(1);
    labels(Rect(0, base + num_red + num_green, 1, num_blue)) = Scalar(2);
    data.resize(base + num_red + num_green + num_blue);
    
    for(vector<lineseg *>::iterator vit = v_lineseg.begin(); vit != v_lineseg.end(); vit++) {
        vector< pair<Point,Point> >::iterator it = (*vit)->begin(), it_end = (*vit)->end();
        for ( ; it != it_end; it++) {
            Point startpt = (*it).first;
            Point endpt = (*it).second;
            int minx = min(startpt.x, endpt.x), maxx = max(startpt.x, endpt.x);
            int miny = min(startpt.y, endpt.y), maxy = max(startpt.y, endpt.y);
            
            Mat mask = Mat::zeros(maxy - miny + 1, maxx - minx + 1, CV_8UC1);
            startpt.x -= minx; startpt.y -= miny;
            endpt.x -= minx; endpt.y -= miny;
            line(mask, startpt, endpt, Scalar(255, 255, 255));
            
            SparseMat smask(mask);
            
            float *dataptr = data.ptr<float>(base);
            for (int fnum = 0; fnum < num_features; fnum++, dataptr++) {
                Mat feature_subim;
                read_feature(h5f, feature_subim, names[fnum].c_str(), Rect(minx, miny, maxx - minx + 1, maxy - miny + 1));
                SparseMatConstIterator it = smask.begin(), it_end = smask.end();
                float *dest = dataptr;
                for ( ; it != it_end; it++, dest += num_features) {
                    const SparseMat::Node *node = it.node();
                    *dest = *feature_subim.ptr<float>(node->idx[0], node->idx[1]);
                }
            }
            base += smask.nzcount();
        }
    }
    assert (base == data.rows);
}


// Implement mouse callback
static void onMouse(int event, int x, int y, int flags, void* ign)
{
    static Point last_draw(0,0);
    switch(event) {
    case CV_EVENT_MOUSEMOVE:
        if (flags & CV_EVENT_FLAG_LBUTTON) {
            if (last_draw != Point(0, 0)) {
                line(im, last_draw, Point(x, y), CV_RGB(255,0,0));
                red_vec.push_back(make_pair(last_draw,Point(x,y)));
            }
            last_draw = Point(x, y);
        }
        else if (flags & CV_EVENT_FLAG_MBUTTON) {
            if (last_draw != Point(0, 0)) {
                line(im, last_draw, Point(x, y), CV_RGB(0,255,0));
                green_vec.push_back(make_pair(last_draw,Point(x,y)));
            }
            last_draw = Point(x, y);
        }
        else if (flags & CV_EVENT_FLAG_RBUTTON) {
            if (last_draw != Point(0, 0)) {
                line(im, last_draw, Point(x, y), CV_RGB(0,0,255));
                blue_vec.push_back(make_pair(last_draw,Point(x,y)));
            }
            last_draw = Point(x, y);
        }
        else {
            last_draw = Point(0, 0);
        }
        break;
    };
    if (! show_overlay) {
        imshow("im", im);
    }
    else {
        Mat tmp;
        add(0.9 * im, overlay, tmp);
        imshow("im", tmp);
    }
}

int main(int argc, char** argv)
{
    assert (argc == 5);
    show_overlay = true;
    
    // Load the image
    im = imread(argv[1]);
    Mat image = imread(argv[1],1);
    image.convertTo(image, CV_8U);
    cout << image << endl;

    vector<Mat> planes;
    split(image, planes);

    // Find training examples
    SparseMat red_mask((planes[0] == 0) & (planes[1] == 0) & (planes[2] == 255));
    SparseMat green_mask((planes[0] == 0) & (planes[1] == 255) & (planes[2] == 0));
    SparseMat blue_mask((planes[0] == 255) & (planes[1] == 0) & (planes[2] == 0));
    
    // Count number of examples
    int num_red = red_mask.nzcount();
    int num_green = green_mask.nzcount();
    int num_blue = blue_mask.nzcount();
    
    cout << argv[1] << " " << num_red << " red, " << num_green << " green, " << num_blue << " blue" << endl;
    
    // Fetch features
    h5f = open_feature_file(argv[2]);
    names = feature_names(h5f);
    
    // Create the original dataset
    data.create(num_red + num_green + num_blue, names.size(), CV_32F);
    labels.create(num_red + num_green + num_blue, 1, CV_32SC1);
    
    // Store the original data
    float *dataptr = data.ptr<float>(base);
    num_features = names.size();
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
    
    // Show interface
    overlay = imread(argv[3]);
    overlay = 0.1 * overlay;
    Mat tmp;
    add(im, overlay, tmp);
    
    imshow("im", tmp);
    cvSetMouseCallback("im",&onMouse, 0 );
    createButton("Train the classifier", button_pushed, NULL, CV_PUSH_BUTTON, 0);
    
    while (true) {
        int k = cvWaitKey(0);
        if (k == ' ') {
            show_overlay = ! show_overlay;
            if (! show_overlay) {
                imshow("im", im);
            } else {
                Mat tmp;
                add(0.9 * im, overlay, tmp);
                imshow("im", tmp);
            }
        }
        if (k == 27)
            break;
    }
    imwrite(argv[4], im);
}