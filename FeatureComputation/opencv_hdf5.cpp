#include <opencv/cv.h>
#include <H5Cpp.h>
#include <assert.h>
using namespace cv;
using namespace H5;
using namespace std;


static Size imsize;
H5File create_feature_file(char *filename, const Mat &base_image)
{
    imsize = base_image.size();
    return H5File(filename, H5F_ACC_TRUNC);
}

H5File open_feature_file(char *filename)
{
    return H5File(filename, H5F_ACC_RDONLY);
}
    

static DataSet create_dataset(H5File h5f, const char *name)
{
    DSetCreatPropList cparms;
    hsize_t chunk_dims[2] = {256, 256};
    hsize_t dims[2];
    cparms.setChunk(2, chunk_dims);
    cparms.setShuffle();
    cparms.setDeflate(5);
    dims[0] = imsize.height;
    dims[1] = imsize.width;
  
    return h5f.createDataSet(name, PredType::NATIVE_FLOAT,
                             DataSpace(2, dims, dims),
                             cparms);
}

void write_feature(H5File h5f, const Mat &image_in, const char *name)
{
    // make sure the sizes match
    assert (imsize == image_in.size());

    // make sure the image is in native float
    Mat image;
    if (image_in.type() !=  CV_32F)
        image_in.convertTo(image, CV_32F);
    else
        image = image_in;
    
    DataSet dataset = create_dataset(h5f, name);

    DataSpace imspace;
    float *imdata;
    if (image.isContinuous()) {
        imspace = dataset.getSpace(); // same size as 
        imspace.selectAll();
        imdata = image.ptr<float>();
    } else {
        // we are working with an ROI
        assert (image.isSubmatrix());
        Size parent_size; Point parent_ofs;
        image.locateROI(parent_size, parent_ofs);
        hsize_t parent_count[2];
        parent_count[0] = parent_size.height; parent_count[1] = parent_size.width;
        imspace.setExtentSimple(2, parent_count);
        hsize_t im_offset[2], im_size[2];
        im_offset[0] = parent_ofs.y; im_offset[1] = parent_ofs.x;
        im_size[0] = image.size().height; im_size[1] = image.size().width;
        imspace.selectHyperslab(H5S_SELECT_SET, im_size, im_offset);
        imdata = image.ptr<float>() - parent_ofs.x - parent_ofs.y * parent_size.width;
    }
    dataset.write(imdata, PredType::NATIVE_FLOAT, imspace);
}

void read_feature(H5File h5f, Mat &image_out, const char *name, const Rect &roi=Rect(0,0,0,0))
{
    DataSet dataset = h5f.openDataSet(name);
    DataSpace dspace = dataset.getSpace();
    assert (dspace.getSimpleExtentNdims() == 2);
    hsize_t dims[2];
    dspace.getSimpleExtentDims(dims);
    if ((roi.width == 0) && (roi.height == 0)) {
        image_out.create(dims[0], dims[1], CV_32F);
        dspace.selectAll();
    } else {
        image_out.create(roi.height, roi.width, CV_32F);
        hsize_t _offset[2], _size[2];
        _offset[0] = roi.y; _offset[1] = roi.x;
        _size[0] = roi.height; _size[1] = roi.width;
        dspace.selectHyperslab(H5S_SELECT_SET, _size, _offset);
    }
    
    DataSpace imspace;
    float *imdata;
    if (image_out.isContinuous()) {
        imspace = dataset.getSpace(); // same size as 
        imspace.selectAll();
        imdata = image_out.ptr<float>();
    } else {
        // we are working with an ROI
        assert (image_out.isSubmatrix());
        Size parent_size; Point parent_ofs;
        image_out.locateROI(parent_size, parent_ofs);
        hsize_t parent_count[2];
        parent_count[0] = parent_size.height; parent_count[1] = parent_size.width;
        imspace.setExtentSimple(2, parent_count);
        hsize_t im_offset[2], im_size[2];
        im_offset[0] = parent_ofs.y; im_offset[1] = parent_ofs.x;
        im_size[0] = image_out.size().height; im_size[1] = image_out.size().width;
        imspace.selectHyperslab(H5S_SELECT_SET, im_size, im_offset);
        imdata = image_out.ptr<float>() - parent_ofs.x - parent_ofs.y * parent_size.width;
    }
    dataset.read(imdata, PredType::NATIVE_FLOAT, imspace, dspace);
}

void read_feature_size(H5File h5f, Size &size_out, const char *name)
{
    DataSet dataset = h5f.openDataSet(name);
    DataSpace dspace = dataset.getSpace();
    assert (dspace.getSimpleExtentNdims() == 2);
    hsize_t dims[2];
    dspace.getSimpleExtentDims(dims);
    size_out.height = dims[0];
    size_out.width = dims[1];
}

vector<string> get_feature_names(H5File h5f)
{
    vector<string> out;
    int num_features;
    for (int i = 0; i < h5f.getNumObjs(); i++)
        out.push_back(h5f.getObjnameByIdx(i));
    return out;
}
