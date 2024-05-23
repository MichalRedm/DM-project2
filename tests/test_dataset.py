import unittest
import dm_project2.dataset as ds

class TestDataset(unittest.TestCase):
  
    def test_invalid_dataset(self):
        self.assertRaises(ds.InvalidDatasetException, ds.MovieLensDataset, "lorem ipsum")

if __name__ == '__main__':
    unittest.main()
