import unittest
import mock
import pandas as pd
from context import dataset as ds


def mock_init(self: ds.MovieLensDataset) -> None:
    """Mock initialization for the class `MovieLensDataset`."""
    self._ratings = pd.DataFrame({
        'userId': [1],
        'movieId': [1],
        'rating': [1],
        'timestamp': [0]
    }).set_index(['userId', 'movieId'])
    self._movies = pd.DataFrame({
        'movieId': [1],
        'title': ['Lorem'],
        'genres': ['Ipsum']
    }).set_index('movieId')


class TestDataset(unittest.TestCase):
    """Set of test cases for the class `MovieLensDataset`."""
  
    def test_invalid_dataset(self):
        """Check if an exception is thrown when incorrect dataset name is specified."""
        self.assertRaises(ds.InvalidDatasetException, ds.MovieLensDataset, "lorem ipsum")
    
    def test_get_movie_by_id(self):
        """Check if a movie can be correctly retrieved by an id."""
        with mock.patch.object(ds.MovieLensDataset, '__init__', mock_init):
            dataset = ds.MovieLensDataset()
            movie = dataset.get_movie_by_id(1)
            self.assertEqual(1, movie.id)
            self.assertEqual('Lorem', movie.title)
            self.assertEqual(['Ipsum'], movie.genres)

    def test_get_nonexistent_movie(self):
        """Check if attempting to retrieve a nonexistent movie will raise an exception."""
        with mock.patch.object(ds.MovieLensDataset, '__init__', mock_init):
            dataset = ds.MovieLensDataset()
            self.assertRaises(ds.InvalidMovieException, dataset.get_movie_by_id, 2)
    
    def test_delete_rating(self):
        """Check if deleting rating works correctly."""
        with mock.patch.object(ds.MovieLensDataset, '__init__', mock_init):
            dataset = ds.MovieLensDataset()
            self.assertTrue(dataset.delete_rating(1, 1))
            self.assertTrue(dataset._ratings.empty)
    
    def test_delete_rating_for_nonexistent_user(self):
        """Check if deleting rating from a user that does not exist will raise an exception."""
        with mock.patch.object(ds.MovieLensDataset, '__init__', mock_init):
            dataset = ds.MovieLensDataset()
            self.assertRaises(ds.InvalidUserException, dataset.delete_rating, 2, 1)
    
    def test_delete_rating_for_nonexistent_movie(self):
        """Check if deleting rating for a movie that does not exist will raise an exception."""
        with mock.patch.object(ds.MovieLensDataset, '__init__', mock_init):
            dataset = ds.MovieLensDataset()
            self.assertRaises(ds.InvalidMovieException, dataset.delete_rating, 1, 2)


if __name__ == '__main__':
    unittest.main()
