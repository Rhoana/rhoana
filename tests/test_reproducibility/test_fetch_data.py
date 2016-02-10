import unittest
import shutil
import tempfile
import os.path
import hashlib

from rhoana.reproducibility import fetch

class TestFetchData(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_download(self):
        testfile = 'small.zip'
        fetch(testfile, self.test_dir)
        path = os.path.join(self.test_dir, testfile)
        self.assertTrue(os.path.exists(path))

        h = hashlib.md5()
        h.update(open(path, 'rb').read())
        md5 = h.hexdigest().lower()
        self.assertEqual(md5, 'ed4e780570b7f99dc5689b0fc5c12981')

if __name__ == '__main__':
    unittest.main()
