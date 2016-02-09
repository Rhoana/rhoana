import re
import logging
from zipfile import ZipFile

import numpy as np
import imread

rlogger = logging.getLogger("rhoana.ingest")

image_pattern = re.compile('^.*\.(png|tif|jpg|jpeg)$')

def zip_imread(zipfile, name):
    imdata = zipfile.read(name)
    extension = name.split('.')[-1]
    im = imread.imread_from_blob(imdata, formatstr=extension)
    # promote grayscale to 3D with a single-channel
    im.shape += (1,) * (3 - im.ndim)
    return im

def ingest_zip(path, storage):
    rlogger.info("Ingesting images from {!s} into {!s}".format(path, storage))

    zipfile = ZipFile(open(path, "rb"))

    names = zipfile.namelist()
    image_names = [n for n in names if image_pattern.match(n)]
    image_names.sort()
    rlogger.info("Found {} images in {}".format(len(image_names), path))

    # Extract size information from the first image
    rlogger.info("Extracting size information from {}".format(image_names[0]))
    im = zip_imread(zipfile, image_names[0])
    rlogger.info("Image size {}".format(im.shape))

    # create dataset
    dataset = storage.new_dataset("raw", (len(image_names),) + im.shape, np.float32)

    # read and store images
    for idx, n in enumerate(image_names):
        im = zip_imread(zipfile, n)
        dataset[idx, ...] = im

    return dataset
