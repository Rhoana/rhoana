OPENCV=/Users/thouis/opencv-install
HDF5_DIR=/Users/thouis/homebrew/Cellar/hdf5/1
CXXFLAGS=-I$(OPENCV)/include -I$(HDF5_DIR)/include -g
LDFLAGS=-L$(OPENCV)/lib -lopencv_highgui -lopencv_imgproc -lopencv_core -L$(HDF5_DIR)/lib -lhdf5_cpp -lhdf5

all: compute_features show_features test_image_to_hdf5 train_randomforest predict_randomforest train_gb predict_gb

membrane.o: quickmedian.h

compute_features: opencv_hdf5.o compute_features.o membrane.o adapthisteq.o local_statistics.o tensor_gradient_features.o drawhist.o vesicles.o
	g++ -o $@ $^ $(LDFLAGS)

show_features: opencv_hdf5.o show_features.o
	g++ -o $@ $^ $(LDFLAGS)

train_randomforest: opencv_hdf5.o train_randomforest.o
	g++ -o $@ $^ $(LDFLAGS) -lopencv_ml

predict_randomforest: opencv_hdf5.o predict_randomforest.o
	g++ -o $@ $^ $(LDFLAGS) -lopencv_ml

train_gb: opencv_hdf5.o train_gb.o
	g++ -o $@ $^ $(LDFLAGS) -lopencv_ml

predict_gb: opencv_hdf5.o predict_gb.o
	g++ -o $@ $^ $(LDFLAGS) -lopencv_ml


test_image_to_hdf5: opencv_hdf5.o test_image_to_hdf5.o 
	g++ -o $@ $^ $(LDFLAGS)

sketch_on_image: sketch_on_image.o
	g++ -o $@ $^ $(LDFLAGS)
