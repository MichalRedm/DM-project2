import pandas as pd
from preprocessing import MovieLensDataset


class BaselinePredictor:
    """Baseline predictor for the Movie Lens dataset."""

    _dataset: MovieLensDataset

    def __init__(self) -> None:
        self._dataset = None
    
    def fit(self, dataset: MovieLensDataset) -> None:
        """Fits the predictor to the data."""
        self._dataset = dataset
    
    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Computes the baseline prediction for a given user-movie pair

        Parameters
        ----------
        user_id : int
            Id of the user in the Movie Lens dataset.

        movie_id : int
            Id of the movie in the Movie Lens dataset.
        
        Returns
        -------
        float
            Prediction of the rating (not rounded).
        """
        ratings = self._dataset.get_ratings()
        movie_avg = ratings[ratings.index.get_level_values('movieId') == movie_id]['rating'].mean()
        if pd.isna(movie_avg):
            movie_avg = 3.5
        return movie_avg
