import unittest
import shutil
import tempfile
import os.path
import numpy as np

from rhoana.backends.storage import HDF5Storage

class TestFetch(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_create_storage(self):
        path = os.path.join(self.test_dir, 'test.hdf5')
        storage = HDF5Storage(path)
        del storage  # should close and flush the hdf5 file
        self.assertTrue(os.path.exists(path))

    def test_create_dataset(self):
        path = os.path.join(self.test_dir, 'test.hdf5')
        storage = HDF5Storage(path)
        randvals = np.random.uniform(0, 1, [2, 4, 4, 1]).astype(np.float32)
        ds = storage.new_dataset("tmp", randvals.shape, dtype=np.float32)
        ds[...] = randvals
        self.assertEqual(ds[...].shape, randvals.shape)
        self.assertEqual((ds.depth, ds.width, ds.height, ds.channels), randvals.shape)
        self.assertTrue(np.all(ds[...] == randvals))

if __name__ == '__main__':
    unittest.main()
