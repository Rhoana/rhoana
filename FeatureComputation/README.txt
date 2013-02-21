Usage:

Training
  compute_features IMAGE_1.tif FEATURES_1.hdf5
  compute_features IMAGE_2.tif FEATURES_2.hdf5
  ...
  sketch_on_image IMAGE_1.tif TRAIN_1.tif
  (Green/LeftButton = foreground, Red/RightButton = Background)
  train_{gb,randomforest} TRAIN_1.tif FEATURES_1.hdf5 TRAIN_2.tif FEATURES_2.hdf5 ...
  predict_{gb,randomforest} {GB/forest}.xml FEATURES_1.hdf5
    (will show output)
  (repeat sketch/train/predict)

  predict_{gb,randomforest} {GB/forest}.xml FEATURES_1.hdf5 PREDICTION_1.hdf5
