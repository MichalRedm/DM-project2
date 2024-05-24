import unittest
import mock
import pandas as pd
from context import dataset as ds


def mock_init(cls):
    cls._ratings = pd.DataFrame({
        'userId': [1],
        'movieId': [1],
        'rating': [1],
        'timestamp': [0]
    }).set_index(['userId', 'movieId'])
    cls._movies = pd.DataFrame({
        'movieId': [1],
        'title': ['Lorem'],
        'genre': ['Ipsum']
    }).set_index('movieId')


class TestDataset(unittest.TestCase):
  
    def test_invalid_dataset(self):
        self.assertRaises(ds.InvalidDatasetException, ds.MovieLensDataset, "lorem ipsum")
    
    def test_get_nonexistent_movie(self):
        with mock.patch.object(ds.MovieLensDataset, '__init__', mock_init):
            dataset = ds.MovieLensDataset()
            self.assertRaises(ds.InvalidMovieException, dataset.get_movie_by_id, 2)
    
    def test_delete_rating(self):
        with mock.patch.object(ds.MovieLensDataset, '__init__', mock_init):
            dataset = ds.MovieLensDataset()
            self.assertTrue(dataset.delete_rating(1, 1))
            self.assertTrue(dataset._ratings.empty)
    
    def test_delete_rating_for_nonexistent_user(self):
        with mock.patch.object(ds.MovieLensDataset, '__init__', mock_init):
            dataset = ds.MovieLensDataset()
            self.assertRaises(ds.InvalidUserException, dataset.delete_rating, 2, 1)
    
    def test_delete_rating_for_nonexistent_movie(self):
        with mock.patch.object(ds.MovieLensDataset, '__init__', mock_init):
            dataset = ds.MovieLensDataset()
            self.assertRaises(ds.InvalidMovieException, dataset.delete_rating, 1, 2)

if __name__ == '__main__':
    unittest.main()
