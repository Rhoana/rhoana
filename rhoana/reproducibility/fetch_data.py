import os.path
import requests
import logging

rlogger = logging.getLogger("rhoana")

def fetch(dataset_name,
          output_directory,
          default_source='http://rhoana.s3-website-us-east-1.amazonaws.com'):
    '''fetch a dataset by name over the net and store it.'''

    # output path
    file_path = os.path.join(output_directory, dataset_name)

    # check for data already downloaded
    if os.path.exists(file_path):
        rlogger.info("Not downloading {} because it already exists in directory {}".format(dataset_name, output_directory))
        return file_path

    url = '{}/{}'.format(default_source, dataset_name)
    try:
        r = requests.get(url, stream=True)
        with open(file_path, "wb") as f:
            for block in r.iter_content(chunk_size=(2 ** 20)):
                f.write(block)
    except Exception as err:
        rlogger.exception("Downloading {}".format(url))
    return file_path

# TODO:
#   report bandwidth, download time
